from __future__ import annotations

import asyncio
import json
import re
import time
import uuid

from fastapi import APIRouter, Request

from src.agents.crew_runtime import CrewRuntime
from src.agents.intake import IntakeAgent
from src.agents.planner import PlannerAgent
from src.agents.retriever_agent import CatalogRetrieverAgent
from src.agents.verifier import VerifierAgent
from src.api.models.requests import AskQueryRequest, PlanQueryRequest, PrereqQueryRequest
from src.api.models.responses import AskQueryResponse, PlanQueryResponse, PrereqQueryResponse
from src.core.config import settings
from src.core.constants import ABSTENTION_MSG, COURSE_CODE_PAT
from src.core.exceptions import GraphNodeNotFoundError
from src.graph.reasoning import check_eligibility, get_available_next_courses
from src.inference.groq_client import call_llm
from src.ops import metrics
from src.ops.tracker import log_trace

router = APIRouter()

intake_agent = IntakeAgent()
retriever_agent = CatalogRetrieverAgent()
planner_agent = PlannerAgent()
verifier_agent = VerifierAgent()
crew_runtime = CrewRuntime(intake_agent, retriever_agent, planner_agent, verifier_agent)

def _trace_id_from_request(request: Request) -> str:
	trace_id = getattr(request.state, "trace_id", "")
	if trace_id:
		return trace_id
	return str(uuid.uuid4())


def _normalize_decision(decision: str, fallback: str = "Need more info") -> str:
	allowed = {"Eligible", "Not eligible", "Need more info"}
	if decision in allowed:
		return decision
	if fallback in allowed:
		return fallback
	return "Need more info"


def _extract_target_course(question: str) -> str:
	match = COURSE_CODE_PAT.search(question.upper())
	if not match:
		return ""
	return match.group(0).replace(" ", "").upper()


def _make_evidence(citations: list[dict]) -> str:
	if not citations:
		return "No cited evidence available in retrieved context."
	lines = []
	for citation in citations:
		lines.append(
			"[SOURCE: "
			f"{citation.get('document_name', '')}, "
			f"Section: {citation.get('section_heading', '')}, "
			f"Chunk: {citation.get('chunk_id', '')}]"
		)
	return "\n".join(lines)


def _next_step(graph_result: dict) -> str:
	missing = graph_result.get("missing_prereqs", [])
	if missing:
		return f"Complete missing prerequisites first: {', '.join(missing)}."
	either_or_options = graph_result.get("either_or_options", [])
	if either_or_options:
		groups: list[str] = []
		for options in either_or_options[:3]:
			if not isinstance(options, list):
				continue
			cleaned = [str(option).strip() for option in options if str(option).strip()]
			if cleaned:
				formatted = [f"({option})" if " and " in option.lower() else option for option in cleaned[:4]]
				groups.append(" OR ".join(formatted))
		if groups:
			return f"Complete one option from each prerequisite group: {'; '.join(groups)}."
	gpa_issues = graph_result.get("gpa_issues", [])
	if gpa_issues:
		if any("student level not provided" in str(issue).lower() for issue in gpa_issues):
			return "Share whether you are an undergraduate or graduate student for accurate GPA policy checks."
		if any("not provided" in str(issue).lower() for issue in gpa_issues):
			return "Share your current cumulative GPA for complete eligibility verification."
		return "Review GPA requirement constraints and contact your advisor if needed."
	if graph_result.get("decision") == "Need more info":
		return "Share additional profile details for a precise eligibility check."
	return "You can proceed with enrollment verification through your department/advisor."


def _default_graph_result() -> dict:
	return {
		"eligible": False,
		"decision": "Need more info",
		"missing_prereqs": [],
		"missing_coreqs": [],
		"grade_issues": [],
		"gpa_issues": [],
		"prereq_path": [],
		"either_or_options": [],
	}


_BARE_DECISION_ANSWERS = {
	"eligible",
	"not eligible",
	"need more info",
	"need more information",
}


def _extract_retrieval_scores(payload: dict) -> list[float]:
	raw_scores = payload.get("retrieval_scores", []) or []
	scores: list[float] = []
	for value in raw_scores:
		try:
			scores.append(float(value))
		except Exception:
			continue
	if scores:
		return scores

	for chunk in payload.get("retrieved_chunks", []) or []:
		try:
			scores.append(float((chunk or {}).get("score", 0.0) or 0.0))
		except Exception:
			continue
	return scores


def _ask_confidence_summary(payload: dict, question: str = "") -> dict:
	scores = _extract_retrieval_scores(payload)
	citation_count = len(payload.get("citations", []) or [])
	chunks_retrieved = int(payload.get("chunks_retrieved", 0) or 0)
	if not chunks_retrieved:
		chunks_retrieved = len(payload.get("retrieved_chunks", []) or [])

	avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
	top_score = round(max(scores), 4) if scores else 0.0

	if citation_count == 0:
		confidence_band = "low"
	elif top_score >= 0.75 and avg_score >= 0.65 and citation_count >= 2:
		confidence_band = "high"
	elif top_score >= 0.6 and avg_score >= 0.5:
		confidence_band = "medium"
	else:
		confidence_band = "low"

	return {
		"question": _compact_text(question, 260),
		"citation_count": citation_count,
		"chunks_retrieved": chunks_retrieved,
		"avg_score": avg_score,
		"top_score": top_score,
		"confidence_band": confidence_band,
	}


def _parse_abstention_json(raw_text: str) -> tuple[bool, float, str]:
	content = str(raw_text or "").strip()
	if not content:
		return False, 0.0, ""

	candidate = content
	if "```" in candidate:
		match = re.search(r"\{.*\}", candidate, re.DOTALL)
		if match:
			candidate = match.group(0)

	try:
		parsed = json.loads(candidate)
		if not isinstance(parsed, dict):
			return False, 0.0, ""
		abstain = bool(parsed.get("abstain", False))
		confidence = float(parsed.get("confidence", 0.0) or 0.0)
		reason = str(parsed.get("reason", "") or "").strip()
		if confidence < 0.0:
			confidence = 0.0
		if confidence > 1.0:
			confidence = 1.0
		return abstain, confidence, reason
	except Exception:
		return False, 0.0, ""


async def _llm_should_abstain_ask(payload: dict, question: str) -> tuple[bool, str]:
	if settings.GROQ_API_KEY.startswith("dev-placeholder"):
		summary = _ask_confidence_summary(payload, question)
		return summary.get("confidence_band") == "low", "heuristic-fallback"

	summary = _ask_confidence_summary(payload, question)
	evidence = {
		"question": summary,
		"current_answer": _compact_text(payload.get("answer", ""), 220),
		"citations": _compact_list(payload.get("citations", []) or [], max_items=5, max_item_chars=100),
		"context_excerpt": _compact_text(payload.get("context_string", ""), 900),
	}

	system_prompt = (
		"You are an abstention policy judge for an academic course-catalog website assistant. "
		"The assistant is only allowed to answer from catalog/policy context provided in evidence. "
		"Abstain if evidence is missing, weak-confidence, indirect, or out-of-scope for catalog/policy data. "
		"Return ONLY strict JSON with keys: abstain (boolean), confidence (0..1), reason (string)."
	)
	user_prompt = (
		f"User question: {question}\n"
		f"Evidence bundle: {json.dumps(evidence, ensure_ascii=True)}\n"
		"Decide whether to abstain."
	)

	try:
		raw = await call_llm(system_prompt, user_prompt, settings.TEMP_REASONING)
		abstain, judge_confidence, reason = _parse_abstention_json(raw)
		if judge_confidence >= 0.6:
			return abstain, reason
	except Exception:
		pass

	# Deterministic fallback for reliability if abstention judge fails/parses low confidence.
	if summary.get("confidence_band") == "low":
		return True, "low-retrieval-confidence"
	return False, "sufficient-retrieval-confidence"


async def _apply_confidence_based_ask_abstention(payload: dict, question: str) -> dict:
	enriched = dict(payload)
	should_abstain, reason = await _llm_should_abstain_ask(enriched, question)
	if should_abstain:
		enriched["abstained"] = True
		enriched["citations"] = []
		enriched["answer"] = ABSTENTION_MSG
		enriched["abstention_reason"] = reason
	return enriched


def _format_or_group(group: list[str]) -> str:
	options = [str(option).strip() for option in group if str(option).strip()]
	formatted = [f"({option})" if " and " in option.lower() else option for option in options[:4]]
	return " OR ".join(formatted)


def _needs_enrichment(answer: str, min_len: int = 60) -> bool:
	clean = " ".join(str(answer or "").split()).strip()
	if not clean:
		return True
	if clean.lower() in _BARE_DECISION_ANSWERS:
		return True
	return len(clean) < min_len


def _compact_text(value: object, max_chars: int = 120) -> str:
	clean = " ".join(str(value or "").split()).strip()
	if len(clean) <= max_chars:
		return clean
	if max_chars <= 3:
		return clean[:max_chars]
	return f"{clean[: max_chars - 3].rstrip()}..."


def _compact_list(values: list[object], max_items: int = 5, max_item_chars: int = 80) -> list[str]:
	items: list[str] = []
	for value in values[:max_items]:
		if isinstance(value, list):
			joined = " | ".join(_compact_text(item, max_item_chars) for item in value[:4])
			items.append(_compact_text(joined, max_item_chars))
		else:
			items.append(_compact_text(value, max_item_chars))
	return items


def _normalize_clarifying_questions(values: list[object], max_items: int = 5) -> list[str]:
	cleaned: list[str] = []
	seen: set[str] = set()
	for value in values:
		question = " ".join(str(value or "").split()).strip()
		if not question:
			continue
		if not question.endswith("?"):
			question = f"{question.rstrip('.')}?"
		normalized = question.lower()
		if normalized in seen:
			continue
		seen.add(normalized)
		cleaned.append(question)
		if len(cleaned) >= max_items:
			break
	return cleaned


def _parse_clarifying_questions(raw_text: str) -> list[str]:
	content = str(raw_text or "").strip()
	if not content:
		return []

	candidate = content
	if "```" in content:
		array_match = re.search(r"\[.*\]", content, re.DOTALL)
		obj_match = re.search(r"\{.*\}", content, re.DOTALL)
		if array_match:
			candidate = array_match.group(0)
		elif obj_match:
			candidate = obj_match.group(0)

	questions: list[str] = []
	try:
		parsed = json.loads(candidate)
		if isinstance(parsed, list):
			questions = [str(item) for item in parsed]
		elif isinstance(parsed, dict):
			values = parsed.get("questions") or parsed.get("clarifying_questions") or []
			if isinstance(values, list):
				questions = [str(item) for item in values]
	except Exception:
		for raw_line in content.splitlines():
			line = re.sub(r"^[\s\-*\d\.)]+", "", raw_line).strip()
			if not line:
				continue
			questions.append(line)

	return _normalize_clarifying_questions(questions)


def _finalize_answer(answer: str, fallback: str) -> str:
	clean = str(answer or "").strip()
	if not clean:
		clean = str(fallback or "").strip()
	if not clean:
		return ""
	lines = [line.strip() for line in clean.splitlines() if line.strip()]
	if lines:
		return "\n".join(lines)
	return " ".join(clean.split())


def _build_prereq_context(payload: dict, question: str = "") -> dict:
	graph_result = payload.get("graph_result", {}) or {}
	missing_prereqs = graph_result.get("missing_prereqs", []) or []
	gpa_issues = graph_result.get("gpa_issues", []) or []
	if missing_prereqs:
		# Keep focus on immediate blockers before secondary policy checks.
		gpa_issues = []
	return {
		"question": _compact_text(question, 260),
		"decision": _normalize_decision(str(payload.get("decision", "Need more info")), fallback="Need more info"),
		"current_answer": _compact_text(payload.get("answer", ""), 180),
		"missing_prereqs": _compact_list(missing_prereqs, max_items=6, max_item_chars=32),
		"missing_coreqs": _compact_list(graph_result.get("missing_coreqs", []) or [], max_items=6, max_item_chars=32),
		"either_or_options": _compact_list(graph_result.get("either_or_options", []) or [], max_items=3, max_item_chars=90),
		"grade_issues": _compact_list(graph_result.get("grade_issues", []) or [], max_items=3, max_item_chars=90),
		"gpa_issues": _compact_list(gpa_issues, max_items=3, max_item_chars=90),
		"next_step": _compact_text(payload.get("next_step", ""), 140),
		"prereq_path": _compact_list(graph_result.get("prereq_path", []) or [], max_items=8, max_item_chars=30),
		"clarifying_questions": _compact_list(payload.get("clarifying_questions", []) or [], max_items=3, max_item_chars=90),
	}


def _build_plan_context(payload: dict, question: str = "") -> dict:
	plan = payload.get("plan", []) or []
	plan_courses: list[str] = []
	for course in plan[:6]:
		if not isinstance(course, dict):
			continue
		code = str(course.get("course_code", "") or "").strip()
		credits = int(course.get("credits", 0) or 0)
		if code:
			plan_courses.append(f"{code} ({credits} credits)")
	return {
		"question": _compact_text(question, 260),
		"current_answer": _compact_text(payload.get("answer", ""), 180),
		"plan_size": len(plan),
		"total_credits": int(sum(int(course.get("credits", 0) or 0) for course in plan if isinstance(course, dict))),
		"plan_courses": plan_courses,
		"recommended_courses": _compact_list(
			[str(course.get("course_code", "")) for course in plan if isinstance(course, dict)],
			max_items=6,
			max_item_chars=24,
		),
		"clarifying_questions": _compact_list(payload.get("clarifying_questions", []) or [], max_items=3, max_item_chars=90),
		"risks": _compact_list(payload.get("risks", []) or [], max_items=3, max_item_chars=90),
	}


def _build_ask_context(payload: dict, question: str = "") -> dict:
	confidence = _ask_confidence_summary(payload, question)
	return {
		"question": _compact_text(question, 260),
		"abstained": bool(payload.get("abstained", False)),
		"current_answer": _compact_text(payload.get("answer", ""), 180),
		"citations": _compact_list(payload.get("citations", []) or [], max_items=5, max_item_chars=80),
		"citation_count": len(payload.get("citations", []) or []),
		"confidence": confidence,
	}


async def _llm_short_answer(endpoint: str, payload: dict, fallback: str, question: str = "") -> str:
	if endpoint == "ask" and bool(payload.get("abstained", False)):
		return _finalize_answer(fallback, fallback)

	if endpoint == "prereq":
		context = _build_prereq_context(payload, question)
	elif endpoint == "plan":
		context = _build_plan_context(payload, question)
	else:
		context = _build_ask_context(payload, question)

	system_prompt = (
		"You are an academic advising assistant generating final API responses. "
		"Use only provided facts and answer the user naturally. "
		"Keep the response brief and direct in a normal conversational tone. "
		"Prefer plain prose over bullet points unless the question explicitly asks for a list. "
		"Never invent course codes, numbers, credits, or policy details not present in the provided facts. "
		"If a detail is missing or uncertain, omit that detail instead of guessing. "
		"Avoid mentioning internal JSON, graph objects, pipelines, or meta-explanations. "
		"Do not use boilerplate phrases such as 'This explanation is tailored to your question'. "
		"Return plain text only."
	)
	user_prompt = (
		f"Endpoint: /query/{endpoint}\n"
		f"User question: {question}\n"
		f"Structured facts: {json.dumps(context, ensure_ascii=True)}\n"
		"Write the final answer now. Keep it concise."
	)

	try:
		generated = await call_llm(system_prompt, user_prompt, settings.TEMP_GENERATION)
	except Exception:
		return _finalize_answer(fallback, fallback)

	return _finalize_answer(generated, fallback)


def _fallback_prereq_answer(payload: dict) -> str:
	decision = _normalize_decision(str(payload.get("decision", "Need more info")), fallback="Need more info")
	answer = str(payload.get("answer", "") or "").strip()
	graph_result = payload.get("graph_result", {}) or {}
	next_step = str(payload.get("next_step", "") or "").strip()

	if not _needs_enrichment(answer, min_len=55):
		return answer

	if decision == "Eligible":
		base = "You are eligible based on the provided profile and catalog prerequisite rules."
	elif decision == "Not eligible":
		base = "You are not currently eligible based on the catalog prerequisite evaluation."
	else:
		base = "More information is needed to determine eligibility with confidence."

	reasons: list[str] = []

	missing_prereqs = graph_result.get("missing_prereqs", []) or []
	if missing_prereqs:
		reasons.append(f"Missing required prerequisites: {', '.join(str(item) for item in missing_prereqs)}.")

	either_or_options = graph_result.get("either_or_options", []) or []
	if either_or_options and not missing_prereqs:
		group_descriptions: list[str] = []
		for group in either_or_options[:3]:
			if isinstance(group, list):
				formatted = _format_or_group(group)
				if formatted:
					group_descriptions.append(formatted)
		if group_descriptions:
			reasons.append(
				"At least one prerequisite option group is not satisfied: "
				+ "; ".join(group_descriptions)
				+ "."
			)

	grade_issues = graph_result.get("grade_issues", []) or []
	if grade_issues and not missing_prereqs:
		reasons.append("Grade-related constraints: " + "; ".join(str(item) for item in grade_issues[:3]) + ".")

	gpa_issues = graph_result.get("gpa_issues", []) or []
	if gpa_issues and not missing_prereqs and not either_or_options:
		reasons.append("GPA/policy constraints: " + "; ".join(str(item) for item in gpa_issues[:3]) + ".")

	if decision == "Not eligible" and not reasons:
		reasons.append("At least one prerequisite constraint is not fully satisfied in the current graph evaluation.")

	message = " ".join([base] + reasons).strip()
	if next_step and next_step.lower() not in message.lower():
		message = f"{message} Next step: {next_step}"

	return message


def _fallback_plan_answer(payload: dict) -> str:
	answer = str(payload.get("answer", "") or "").strip()
	if not _needs_enrichment(answer, min_len=50):
		return answer

	plan = payload.get("plan", []) or []
	if plan:
		total_credits = int(sum(int(course.get("credits", 0) or 0) for course in plan))
		codes = [str(course.get("course_code", "")).strip() for course in plan if str(course.get("course_code", "")).strip()]
		preview = ", ".join(codes[:5])
		msg = f"Generated a term plan with {len(plan)} course(s) totaling {total_credits} credits."
		if preview:
			msg = f"{msg} Recommended courses: {preview}."
		return f"{msg} Recommendations are based on retrieved catalog evidence and prerequisite availability."

	clarifying_questions = payload.get("clarifying_questions", []) or []
	if clarifying_questions:
		preview_questions = "; ".join(str(item) for item in clarifying_questions[:3])
		return (
			"A complete term plan could not be generated from the current profile and retrieved context. "
			f"Please provide more details: {preview_questions}"
		)

	return "A complete term plan could not be generated from the current profile and retrieved context."


def _plan_needs_clarification(payload: dict) -> bool:
	plan = payload.get("plan", []) or []
	if plan:
		return False

	questions = payload.get("clarifying_questions", []) or []
	if any(str(question or "").strip() for question in questions):
		return True

	answer = str(payload.get("answer", "") or "").strip().lower()
	if not answer:
		return True

	need_markers = (
		"need more info",
		"more information",
		"could not be generated",
		"please provide",
		"missing",
	)
	return any(marker in answer for marker in need_markers)


def _sanitize_plan_clarifying_questions(payload: dict) -> list[str]:
	if not _plan_needs_clarification(payload):
		return []
	return _normalize_clarifying_questions(payload.get("clarifying_questions", []) or [])


def _needs_dynamic_clarifying_questions(endpoint: str, payload: dict) -> bool:
	if endpoint == "plan":
		return _plan_needs_clarification(payload)

	if endpoint == "prereq":
		decision = _normalize_decision(str(payload.get("decision", "Need more info")), fallback="Need more info")
		return decision == "Need more info"

	if endpoint == "ask":
		if bool(payload.get("abstained", False)):
			return True
		answer = str(payload.get("answer", "") or "").lower()
		markers = ("need more info", "more information", "insufficient", "unclear")
		return any(marker in answer for marker in markers)

	return False


def _clarifying_context(endpoint: str, payload: dict, question: str) -> dict:
	if endpoint == "prereq":
		return _build_prereq_context(payload, question)
	if endpoint == "plan":
		return _build_plan_context(payload, question)
	return _build_ask_context(payload, question)


async def _llm_dynamic_clarifying_questions(endpoint: str, payload: dict, question: str) -> list[str]:
	if settings.GROQ_API_KEY.startswith("dev-placeholder"):
		return []

	context = _clarifying_context(endpoint, payload, question)
	system_prompt = (
		"You generate clarification questions for academic advising API responses. "
		"Ask only what is necessary to proceed with confidence. "
		"Avoid repeating details already provided. "
		"Return ONLY valid JSON as an array of strings with at most 5 concise questions."
	)
	user_prompt = (
		f"Endpoint: /query/{endpoint}\n"
		f"User question: {question}\n"
		f"Known facts: {json.dumps(context, ensure_ascii=True)}\n"
		"Generate follow-up clarification questions now."
	)

	try:
		raw = await call_llm(system_prompt, user_prompt, settings.TEMP_CLARIFY)
		return _parse_clarifying_questions(raw)
	except Exception:
		return []


async def _ensure_dynamic_clarifying_questions(endpoint: str, payload: dict, question: str = "") -> list[str]:
	existing = _normalize_clarifying_questions(payload.get("clarifying_questions", []) or [])
	if existing:
		return existing

	if not _needs_dynamic_clarifying_questions(endpoint, payload):
		return []

	return await _llm_dynamic_clarifying_questions(endpoint, payload, question)


def _fallback_ask_answer(payload: dict) -> str:
	answer = str(payload.get("answer", "") or "").strip()
	abstained = bool(payload.get("abstained", False))
	if abstained:
		return answer or ABSTENTION_MSG

	if not _needs_enrichment(answer, min_len=30):
		return answer

	if answer.lower() in _BARE_DECISION_ANSWERS:
		return f"Based on the retrieved catalog context, the result is: {answer}."

	citations = payload.get("citations", []) or []
	if citations:
		return (
			"Based on the retrieved catalog context, the available evidence is limited but suggests the above conclusion. "
			"Please review the cited sources for details."
		)

	return ABSTENTION_MSG


def _ground_prereq_assumptions(payload: dict) -> list[str]:
	graph_result = payload.get("graph_result", {}) or {}
	decision = _normalize_decision(str(payload.get("decision", "Need more info")), fallback="Need more info")
	if decision != "Need more info":
		return []

	gpa_issues = [str(issue) for issue in (graph_result.get("gpa_issues", []) or [])]
	if any("not provided" in issue.lower() for issue in gpa_issues):
		return ["Eligibility depends on missing GPA or student-level details."]

	clarifying_questions = payload.get("clarifying_questions", []) or []
	if clarifying_questions:
		return ["Eligibility depends on additional student profile details."]

	return []


async def _enrich_prereq_payload(payload: dict, question: str = "") -> dict:
	enriched = dict(payload)
	enriched["clarifying_questions"] = await _ensure_dynamic_clarifying_questions("prereq", enriched, question)
	fallback_answer = _fallback_prereq_answer(enriched)
	enriched["answer"] = await _llm_short_answer(
		"prereq",
		enriched,
		fallback_answer,
		question,
	)
	enriched["assumptions"] = _ground_prereq_assumptions(enriched)
	return enriched


async def _enrich_plan_payload(payload: dict, question: str = "") -> dict:
	enriched = dict(payload)
	enriched["clarifying_questions"] = _sanitize_plan_clarifying_questions(enriched)
	enriched["clarifying_questions"] = await _ensure_dynamic_clarifying_questions("plan", enriched, question)
	fallback_answer = _fallback_plan_answer(enriched)
	enriched["answer"] = await _llm_short_answer(
		"plan",
		enriched,
		fallback_answer,
		question,
	)
	return enriched


async def _enrich_ask_payload(payload: dict, question: str = "") -> dict:
	enriched = dict(payload)
	enriched["clarifying_questions"] = await _ensure_dynamic_clarifying_questions("ask", enriched, question)
	fallback_answer = _fallback_ask_answer(enriched)
	if bool(enriched.get("abstained", False)):
		enriched["answer"] = _finalize_answer(enriched.get("answer", "") or fallback_answer, fallback_answer)
	else:
		enriched["answer"] = await _llm_short_answer(
			"ask",
			enriched,
			fallback_answer,
			question,
		)
	enriched = await _apply_confidence_based_ask_abstention(enriched, question)
	if bool(enriched.get("abstained", False)):
		enriched["answer"] = _finalize_answer(enriched.get("answer", "") or ABSTENTION_MSG, ABSTENTION_MSG)
		enriched["clarifying_questions"] = []
	return enriched


@router.post(
	"/prereq",
	response_model=PrereqQueryResponse,
	summary="Check Course Eligibility",
	description="Evaluates whether a student is eligible for a target course using profile data, retrieval evidence, and graph reasoning.",
)
async def prereq_query(payload: PrereqQueryRequest, request: Request) -> PrereqQueryResponse:
	started = time.perf_counter()
	trace_id = _trace_id_from_request(request)
	metrics.increment_total_requests()
	metrics.increment_prereq_requests()

	if settings.CREWAI_ENABLED:
		runtime_result = await crew_runtime.run_prereq(payload.model_dump(), trace_id)
		final_payload = dict(runtime_result.get("response", {}))
		final_payload["trace_id"] = trace_id
		final_payload = await _enrich_prereq_payload(final_payload, question=payload.question)

		chunks_retrieved = int(runtime_result.get("chunks_retrieved", 0) or 0)
		citation_count = int(runtime_result.get("citation_count", 0) or 0)
		abstained = bool(runtime_result.get("abstained", False))
		if abstained:
			metrics.increment_abstentions()
		metrics.add_citation_hits(citation_count)

		latency_ms = (time.perf_counter() - started) * 1000
		metrics.observe_latency_ms(latency_ms)
		metrics.observe_chunks_retrieved(chunks_retrieved)
		await log_trace(
			trace_id=trace_id,
			endpoint="/query/prereq",
			agents_called=runtime_result.get(
				"agents_called",
				["IntakeAgent", "CatalogRetrieverAgent", "PlannerAgent", "VerifierAgent", "CrewRuntime"],
			),
			chunks_retrieved=chunks_retrieved,
			graph_used=bool(runtime_result.get("graph_used", True)),
			verifier_passed=bool(runtime_result.get("verifier_passed", False)),
			verifier_issues=runtime_result.get("verifier_issues", []),
			citation_count=citation_count,
			abstained=abstained,
			total_latency_ms=latency_ms,
		)
		return PrereqQueryResponse(**final_payload)

	profile = await intake_agent.run(payload.model_dump(), mode="prereq")
	missing_profile_fields = [
		str(field).strip()
		for field in (profile.get("missing_fields", []) or [])
		if str(field).strip()
	]
	needs_profile_clarification = bool(missing_profile_fields)

	target_course = _extract_target_course(payload.question)
	enforce_gpa_policies = bool(
		profile.get("gpa", None) is not None
		or str(profile.get("student_level", "")).strip()
		or bool(profile.get("include_financial_aid_policies", False))
	)

	async def _graph_task() -> dict:
		if not target_course:
			return _default_graph_result()
		try:
			return await asyncio.to_thread(
				check_eligibility,
				target_course,
				profile.get("completed_courses", []),
				profile.get("grades", {}),
				student_gpa=profile.get("gpa", None),
				student_level=profile.get("student_level", ""),
				enforce_gpa_policies=enforce_gpa_policies,
				include_financial_aid_policies=bool(profile.get("include_financial_aid_policies", False)),
			)
		except GraphNodeNotFoundError:
			return _default_graph_result()

	filters = {"catalog_year": profile.get("catalog_year", "")} if profile.get("catalog_year") else None
	vector_ctx, graph_result = await asyncio.gather(
		retriever_agent.run(payload.question, filters=filters),
		_graph_task(),
	)
	graph_result_for_prereq = graph_result
	if needs_profile_clarification:
		graph_result_for_prereq = _default_graph_result()

	planner_result = await planner_agent.run(
		profile=profile,
		vector_ctx=vector_ctx,
		graph_result=graph_result_for_prereq,
		mode="prereq",
		question=payload.question,
	)
	effective_decision = _normalize_decision(
		str(graph_result_for_prereq.get("decision", "Need more info")),
		fallback="Need more info",
	)

	draft = {
		"decision": effective_decision,
		"answer": str(planner_result.get("answer", ABSTENTION_MSG)),
		"evidence": _make_evidence(planner_result.get("citations", [])),
		"citations": planner_result.get("citations", []),
		"next_step": _next_step(graph_result_for_prereq),
		"clarifying_questions": (
			planner_result.get("clarifying_questions", [])
			if effective_decision == "Need more info"
			else []
		),
		"graph_result": {
			"eligible": bool(graph_result_for_prereq.get("eligible", False)),
			"missing_prereqs": graph_result_for_prereq.get("missing_prereqs", []),
			"missing_coreqs": graph_result_for_prereq.get("missing_coreqs", []),
			"grade_issues": graph_result_for_prereq.get("grade_issues", []),
			"gpa_issues": graph_result_for_prereq.get("gpa_issues", []),
			"prereq_path": graph_result_for_prereq.get("prereq_path", []),
			"either_or_options": graph_result_for_prereq.get("either_or_options", []),
		},
		"assumptions": planner_result.get("assumptions", []),
		"trace_id": trace_id,
	}

	verified = await verifier_agent.run(draft, vector_ctx.get("retrieved_chunks", []), graph_result_for_prereq)
	final_payload = dict(verified["final_response"])
	final_payload["decision"] = _normalize_decision(
		str(final_payload.get("decision", draft["decision"])),
		fallback=draft["decision"],
	)
	final_payload["trace_id"] = trace_id
	final_payload = await _enrich_prereq_payload(final_payload, question=payload.question)

	chunks_retrieved = len(vector_ctx.get("retrieved_chunks", []))
	citation_count = len(final_payload.get("citations", []))
	if ABSTENTION_MSG in str(final_payload.get("answer", "")):
		metrics.increment_abstentions()
	metrics.add_citation_hits(citation_count)

	latency_ms = (time.perf_counter() - started) * 1000
	metrics.observe_latency_ms(latency_ms)
	metrics.observe_chunks_retrieved(chunks_retrieved)
	await log_trace(
		trace_id=trace_id,
		endpoint="/query/prereq",
		agents_called=["IntakeAgent", "CatalogRetrieverAgent", "PlannerAgent", "VerifierAgent"],
		chunks_retrieved=chunks_retrieved,
		graph_used=True,
		verifier_passed=bool(verified.get("passed", False)),
		verifier_issues=verified.get("issues", []),
		citation_count=citation_count,
		abstained=ABSTENTION_MSG in str(final_payload.get("answer", "")),
		total_latency_ms=latency_ms,
	)

	return PrereqQueryResponse(**final_payload)


@router.post(
	"/plan",
	response_model=PlanQueryResponse,
	summary="Generate Course Plan",
	description="Creates a term plan constrained by completed courses, target major/term, and maximum credit load.",
)
async def plan_query(payload: PlanQueryRequest, request: Request) -> PlanQueryResponse:
	started = time.perf_counter()
	trace_id = _trace_id_from_request(request)
	metrics.increment_total_requests()
	metrics.increment_plan_requests()

	if settings.CREWAI_ENABLED:
		runtime_result = await crew_runtime.run_plan(payload.model_dump(), trace_id)
		final_payload = dict(runtime_result.get("response", {}))
		final_payload["trace_id"] = trace_id
		final_payload = await _enrich_plan_payload(final_payload, question=payload.question)

		chunks_retrieved = int(runtime_result.get("chunks_retrieved", 0) or 0)
		citation_count = int(runtime_result.get("citation_count", 0) or 0)
		abstained = bool(runtime_result.get("abstained", False))
		if abstained:
			metrics.increment_abstentions()
		metrics.add_citation_hits(citation_count)

		latency_ms = (time.perf_counter() - started) * 1000
		metrics.observe_latency_ms(latency_ms)
		metrics.observe_chunks_retrieved(chunks_retrieved)
		await log_trace(
			trace_id=trace_id,
			endpoint="/query/plan",
			agents_called=runtime_result.get(
				"agents_called",
				["IntakeAgent", "CatalogRetrieverAgent", "PlannerAgent", "VerifierAgent", "CrewRuntime"],
			),
			chunks_retrieved=chunks_retrieved,
			graph_used=bool(runtime_result.get("graph_used", True)),
			verifier_passed=bool(runtime_result.get("verifier_passed", False)),
			verifier_issues=runtime_result.get("verifier_issues", []),
			citation_count=citation_count,
			abstained=abstained,
			total_latency_ms=latency_ms,
		)
		return PlanQueryResponse(**final_payload)

	profile = await intake_agent.run(payload.model_dump(), mode="plan")

	plan_question = str(payload.question or "").strip()
	if not plan_question:
		plan_question = (
			f"Create a {profile.get('target_term', '')} course plan for {profile.get('target_major', '')} "
			f"with at most {profile.get('max_credits', 0)} credits."
		)
	filters = {"catalog_year": profile.get("catalog_year", "")} if profile.get("catalog_year") else None
	completed_courses = profile.get("completed_courses", [])
	if completed_courses:
		vector_ctx, available_courses = await asyncio.gather(
			retriever_agent.run(plan_question, filters=filters),
			asyncio.to_thread(get_available_next_courses, completed_courses),
		)
	else:
		vector_ctx = await retriever_agent.run(plan_question, filters=filters)
		available_courses = []

	graph_result = {"available_next_courses": available_courses}
	planner_result = await planner_agent.run(
		profile=profile,
		vector_ctx=vector_ctx,
		graph_result=graph_result,
		mode="plan",
		question=plan_question,
	)

	draft = {
		"answer": planner_result.get("answer", ""),
		"plan": planner_result.get("plan", []),
		"citations": planner_result.get("citations", []),
		"clarifying_questions": planner_result.get("clarifying_questions", []),
		"assumptions": planner_result.get("assumptions", []),
		"risks": planner_result.get("risks", []),
	}

	verified = await verifier_agent.run(draft, vector_ctx.get("retrieved_chunks", []), graph_result)
	final_payload = dict(verified["final_response"])
	final_payload["trace_id"] = trace_id
	final_payload["total_credits"] = int(
		sum(int(course.get("credits", 0) or 0) for course in final_payload.get("plan", []))
	)
	final_payload = await _enrich_plan_payload(final_payload, question=plan_question)

	chunks_retrieved = len(vector_ctx.get("retrieved_chunks", []))
	citation_count = len(final_payload.get("citations", []))
	if ABSTENTION_MSG in str(final_payload.get("answer", "")):
		metrics.increment_abstentions()
	metrics.add_citation_hits(citation_count)

	latency_ms = (time.perf_counter() - started) * 1000
	metrics.observe_latency_ms(latency_ms)
	metrics.observe_chunks_retrieved(chunks_retrieved)
	await log_trace(
		trace_id=trace_id,
		endpoint="/query/plan",
		agents_called=["IntakeAgent", "CatalogRetrieverAgent", "PlannerAgent", "VerifierAgent"],
		chunks_retrieved=chunks_retrieved,
		graph_used=True,
		verifier_passed=bool(verified.get("passed", False)),
		verifier_issues=verified.get("issues", []),
		citation_count=citation_count,
		abstained=ABSTENTION_MSG in str(final_payload.get("answer", "")),
		total_latency_ms=latency_ms,
	)

	return PlanQueryResponse(**final_payload)


@router.post(
	"/ask",
	response_model=AskQueryResponse,
	summary="Ask Catalog Question",
	description="Answers a general catalog/policy question with citations when evidence is available, otherwise abstains safely.",
)
async def ask_query(payload: AskQueryRequest, request: Request) -> AskQueryResponse:
	started = time.perf_counter()
	trace_id = _trace_id_from_request(request)
	metrics.increment_total_requests()

	payload_data = payload.model_dump()
	extra_payload = getattr(payload, "__pydantic_extra__", {}) or {}
	raw_filters = payload_data.get("filters", extra_payload.get("filters"))
	filters = raw_filters if isinstance(raw_filters, dict) else None

	if settings.CREWAI_ENABLED:
		runtime_payload = dict(payload_data)
		runtime_payload["filters"] = filters
		runtime_result = await crew_runtime.run_ask(runtime_payload, trace_id)
		final_payload = dict(runtime_result.get("response", {}))
		final_payload["chunks_retrieved"] = int(runtime_result.get("chunks_retrieved", 0) or 0)
		final_payload["trace_id"] = trace_id
		final_payload = await _enrich_ask_payload(final_payload, question=payload.question)

		chunks_retrieved = int(runtime_result.get("chunks_retrieved", 0) or 0)
		citation_count = len(final_payload.get("citations", []) or [])
		abstained = bool(final_payload.get("abstained", False))
		if abstained:
			metrics.increment_abstentions()
		metrics.add_citation_hits(citation_count)

		latency_ms = (time.perf_counter() - started) * 1000
		metrics.observe_latency_ms(latency_ms)
		metrics.observe_chunks_retrieved(chunks_retrieved)
		await log_trace(
			trace_id=trace_id,
			endpoint="/query/ask",
			agents_called=runtime_result.get(
				"agents_called",
				["IntakeAgent", "CatalogRetrieverAgent", "PlannerAgent", "VerifierAgent", "CrewRuntime"],
			),
			chunks_retrieved=chunks_retrieved,
			graph_used=bool(runtime_result.get("graph_used", False)),
			verifier_passed=bool(runtime_result.get("verifier_passed", False)),
			verifier_issues=runtime_result.get("verifier_issues", []),
			citation_count=citation_count,
			abstained=abstained,
			total_latency_ms=latency_ms,
		)
		return AskQueryResponse(**final_payload)

	profile = await intake_agent.run({}, mode="ask")
	vector_ctx = await retriever_agent.run(payload.question, filters=filters)
	planner_result = await planner_agent.run(
		profile=profile,
		vector_ctx=vector_ctx,
		graph_result={},
		mode="ask",
		question=payload.question,
	)

	draft = {
		"answer": str(planner_result.get("answer", ABSTENTION_MSG)),
		"citations": planner_result.get("citations", []),
		"abstained": ABSTENTION_MSG in str(planner_result.get("answer", "")),
	}
	verified = await verifier_agent.run(draft, vector_ctx.get("retrieved_chunks", []), {})
	final_payload = dict(verified["final_response"])
	final_payload["retrieved_chunks"] = vector_ctx.get("retrieved_chunks", []) or []
	final_payload["retrieval_scores"] = [
		float((chunk or {}).get("score", 0.0) or 0.0)
		for chunk in (vector_ctx.get("retrieved_chunks", []) or [])
	]
	final_payload["context_string"] = str(vector_ctx.get("context_string", "") or "")
	final_payload["abstained"] = bool(final_payload.get("abstained", False)) or (
		ABSTENTION_MSG in str(final_payload.get("answer", ""))
	)
	final_payload["trace_id"] = trace_id
	final_payload = await _enrich_ask_payload(final_payload, question=payload.question)

	chunks_retrieved = len(vector_ctx.get("retrieved_chunks", []))
	citation_count = len(final_payload.get("citations", []))
	if final_payload["abstained"]:
		metrics.increment_abstentions()
	metrics.add_citation_hits(citation_count)

	latency_ms = (time.perf_counter() - started) * 1000
	metrics.observe_latency_ms(latency_ms)
	metrics.observe_chunks_retrieved(chunks_retrieved)
	await log_trace(
		trace_id=trace_id,
		endpoint="/query/ask",
		agents_called=["IntakeAgent", "CatalogRetrieverAgent", "PlannerAgent", "VerifierAgent"],
		chunks_retrieved=chunks_retrieved,
		graph_used=False,
		verifier_passed=bool(verified.get("passed", False)),
		verifier_issues=verified.get("issues", []),
		citation_count=citation_count,
		abstained=bool(final_payload["abstained"]),
		total_latency_ms=latency_ms,
	)

	return AskQueryResponse(**final_payload)
