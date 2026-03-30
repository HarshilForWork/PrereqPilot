from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.constants import ABSTENTION_MSG, COURSE_CODE_PAT, normalize_course_code
from src.core.logger import get_logger
from src.graph.store import get_graph
from src.inference.chains import render_plan_prompt, render_prereq_prompt
from src.inference.groq_client import call_llm

log = get_logger(__name__)

_MAX_PROMPT_LIST_ITEMS = 10
_MAX_PROMPT_VALUE_CHARS = 220


def _compact_text(value: Any, max_chars: int = _MAX_PROMPT_VALUE_CHARS) -> str:
	text = " ".join(str(value or "").split())
	if max_chars <= 3:
		return text[:max_chars]
	if len(text) <= max_chars:
		return text
	return f"{text[: max_chars - 3].rstrip()}..."


def _compact_list(values: list[Any], max_items: int = _MAX_PROMPT_LIST_ITEMS) -> list[str]:
	compacted: list[str] = []
	for item in values[:max_items]:
		if isinstance(item, dict):
			compacted.append(_compact_text(json.dumps(item, ensure_ascii=True)))
		else:
			compacted.append(_compact_text(item))
	return compacted


def _compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
	if not isinstance(profile, dict):
		return {}

	compacted: dict[str, Any] = {}
	for key in (
		"target_major",
		"target_term",
		"max_credits",
		"catalog_year",
		"student_level",
		"gpa",
		"include_financial_aid_policies",
	):
		value = profile.get(key)
		if value in (None, "", []):
			continue
		if isinstance(value, str):
			compacted[key] = _compact_text(value)
		else:
			compacted[key] = value

	completed_courses = profile.get("completed_courses", [])
	if isinstance(completed_courses, list) and completed_courses:
		compacted["completed_courses"] = _compact_list(completed_courses)
		if len(completed_courses) > _MAX_PROMPT_LIST_ITEMS:
			compacted["completed_courses_omitted_count"] = len(completed_courses) - _MAX_PROMPT_LIST_ITEMS

	grades = profile.get("grades", {})
	if isinstance(grades, dict) and grades:
		grade_pairs = list(grades.items())[:_MAX_PROMPT_LIST_ITEMS]
		compacted["grades"] = {
			_compact_text(course, 32): _compact_text(grade, 12) for course, grade in grade_pairs
		}
		if len(grades) > _MAX_PROMPT_LIST_ITEMS:
			compacted["grades_omitted_count"] = len(grades) - _MAX_PROMPT_LIST_ITEMS

	for key in ("missing_fields", "clarifying_questions"):
		values = profile.get(key, [])
		if isinstance(values, list) and values:
			compacted[key] = _compact_list(values)
			if len(values) > _MAX_PROMPT_LIST_ITEMS:
				compacted[f"{key}_omitted_count"] = len(values) - _MAX_PROMPT_LIST_ITEMS

	return compacted


def _compact_graph_result(graph_result: dict[str, Any]) -> dict[str, Any]:
	if not isinstance(graph_result, dict):
		return {}

	compacted: dict[str, Any] = {}
	if "decision" in graph_result:
		compacted["decision"] = _compact_text(graph_result.get("decision", ""), 64)
	if "eligible" in graph_result:
		compacted["eligible"] = bool(graph_result.get("eligible", False))

	for key in (
		"missing_prereqs",
		"missing_coreqs",
		"grade_issues",
		"gpa_issues",
		"prereq_path",
		"either_or_options",
		"available_next_courses",
	):
		value = graph_result.get(key, [])
		if isinstance(value, list) and value:
			compacted[key] = _compact_list(value)
			if len(value) > _MAX_PROMPT_LIST_ITEMS:
				compacted[f"{key}_omitted_count"] = len(value) - _MAX_PROMPT_LIST_ITEMS

	return compacted


def _pick_citation_for_course(course_code: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, str]:
	course_code = normalize_course_code(course_code)
	for chunk in retrieved_chunks:
		metadata = chunk.get("metadata") or {}
		mentioned = str(metadata.get("course_codes_mentioned", ""))
		if course_code in [normalize_course_code(code) for code in mentioned.split(",") if code.strip()]:
			return {
				"chunk_id": str(metadata.get("chunk_id", "")),
				"document_name": str(metadata.get("document_name", "")),
				"section_heading": str(metadata.get("section_heading", "")),
			}

	return {"chunk_id": "", "document_name": "", "section_heading": ""}


def _coerce_positive_int(value: Any, default: int = 0) -> int:
	try:
		parsed = int(value)
		return parsed if parsed > 0 else default
	except Exception:
		return default


def _build_course_metadata_index(graph_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
	index: dict[str, dict[str, Any]] = {}

	nodes = graph_result.get("nodes") or []
	for node in nodes:
		code = normalize_course_code(str(node.get("id", "") or ""))
		if not code:
			continue
		index[code] = {
			"course_name": str(node.get("course_name", "") or "").strip(),
			"credits": _coerce_positive_int(node.get("credits", 0), 0),
		}

	try:
		graph = get_graph()
		for node_code, attrs in graph.nodes(data=True):
			code = normalize_course_code(str(node_code))
			if not code:
				continue

			existing = index.get(code, {})
			existing_name = str(existing.get("course_name", "") or "").strip()
			existing_credits = _coerce_positive_int(existing.get("credits", 0), 0)

			incoming_name = str(attrs.get("course_name", "") or "").strip()
			incoming_credits = _coerce_positive_int(attrs.get("credits", 0), 0)

			index[code] = {
				"course_name": incoming_name if incoming_name else existing_name,
				"credits": incoming_credits if incoming_credits > 0 else existing_credits,
			}
	except Exception:
		# Keep planner robust if graph access fails; caller falls back to conservative defaults.
		pass

	return index


def _extract_major_prefixes(target_major: str) -> set[str]:
	if not target_major:
		return set()

	major = target_major.strip().upper()
	prefixes: set[str] = set()
	for match in re.finditer(r"(?<!\d)(\d{1,2})(?:-\d+)?(?!\d)", major):
		prefixes.add(str(int(match.group(1))))

	if "EECS" in major or "COURSE 6" in major or "COMPUTER SCIENCE" in major or major == "6":
		prefixes.add("6")

	return prefixes


def _course_alias_key(course_code: str) -> str:
	normalized = normalize_course_code(course_code)
	match = re.fullmatch(r"([A-Z0-9]{1,4})\.(\d+)(\[J\])?", normalized)
	if not match:
		return normalized

	prefix, suffix, joint = match.groups()
	if len(suffix) >= 2 and set(suffix) == {"0"}:
		joint_suffix = joint or ""
		return f"{prefix}.0{joint_suffix}"

	return normalized


def _course_family_key(course_code: str) -> str:
	normalized = normalize_course_code(course_code)
	if "." not in normalized:
		return normalized
	prefix, suffix = normalized.split(".", 1)
	suffix = suffix.replace("[J]", "")
	if not suffix:
		return normalized
	return f"{prefix}.{suffix[:3]}"


def _is_placeholder_code(course_code: str, course_name: str, credits: int) -> bool:
	if course_name or credits > 0:
		return False

	normalized = normalize_course_code(course_code)
	if re.fullmatch(r"\d{1,2}\.0{2,}", normalized):
		return True
	if re.fullmatch(r"\d{1,2}\.C0\d+(?:\[J\])?", normalized):
		return True
	return False


def _is_special_or_experiential(course_code: str, course_name: str) -> bool:
	normalized = normalize_course_code(course_code)
	upper_name = (course_name or "").upper()
	if ".S" in normalized:
		return True
	keywords = (
		"SPECIAL SUBJECT",
		"PRACTICAL",
		"INTERNSHIP",
		"INDEPENDENT STUDY",
		"THESIS",
		"SEMINAR",
	)
	return any(keyword in upper_name for keyword in keywords)


def _extract_course_codes(answer_text: str) -> list[str]:
	ordered: list[str] = []
	seen: set[str] = set()
	for match in COURSE_CODE_PAT.finditer(answer_text or ""):
		code = normalize_course_code(match.group(0))
		if code and code not in seen:
			seen.add(code)
			ordered.append(code)
	return ordered


def _dedupe_text_items(values: list[str], max_items: int = 5) -> list[str]:
	cleaned: list[str] = []
	seen: set[str] = set()
	for value in values:
		text = str(value or "").strip()
		if not text:
			continue
		normalized = text.lower()
		if normalized in seen:
			continue
		seen.add(normalized)
		cleaned.append(text)
		if len(cleaned) >= max_items:
			break
	return cleaned


def _extract_course_codes_set(text: str) -> set[str]:
	return {
		normalize_course_code(match.group(0))
		for match in COURSE_CODE_PAT.finditer(text or "")
		if normalize_course_code(match.group(0))
	}


def _sanitize_plan_notes(values: list[str], plan_codes: set[str], max_items: int = 4) -> list[str]:
	cleaned: list[str] = []
	for text in _dedupe_text_items(values, max_items=max_items * 2):
		mentioned_codes = _extract_course_codes_set(text)
		if mentioned_codes:
			if not plan_codes:
				continue
			if not mentioned_codes.issubset(plan_codes):
				continue
		cleaned.append(text)
		if len(cleaned) >= max_items:
			break
	return cleaned


def _build_course_justification(course_code: str, target_major: str, major_match: bool) -> str:
	if target_major and major_match:
		return (
			f"Recommended because it is currently available based on completed prerequisites and aligns with your "
			f"{target_major} focus."
		)
	return "Recommended because it is currently available based on completed prerequisites."


def _build_plan_from_graph(
	available_courses: list[str],
	retrieved_chunks: list[dict[str, Any]],
	max_credits: int,
	answer_text: str,
	graph_result: dict[str, Any],
	target_major: str,
	completed_courses: list[str],
) -> list[dict[str, Any]]:
	metadata_index = _build_course_metadata_index(graph_result)
	major_prefixes = _extract_major_prefixes(target_major)
	completed_families = {
		_course_family_key(course)
		for course in completed_courses
		if str(course).strip()
	}

	if not available_courses:
		return []
	else:
		codes = [normalize_course_code(code) for code in available_courses if str(code).strip()]

	candidates: list[dict[str, Any]] = []
	for course_code in codes:
		if _course_family_key(course_code) in completed_families:
			continue

		metadata = metadata_index.get(course_code, {})
		course_name = str(metadata.get("course_name", "") or "").strip()
		catalog_credits = _coerce_positive_int(metadata.get("credits", 0), 0)
		effective_credits = catalog_credits if catalog_credits > 0 else 3

		if _is_placeholder_code(course_code, course_name, catalog_credits):
			continue

		major_match = False
		prefix = course_code.split(".", 1)[0] if "." in course_code else ""
		if major_prefixes and prefix:
			major_match = prefix in major_prefixes

		citation = _pick_citation_for_course(course_code, retrieved_chunks)
		citation_hit = bool(citation.get("chunk_id") and citation.get("document_name"))
		special_topic = _is_special_or_experiential(course_code, course_name)

		candidates.append(
			{
				"course_code": course_code,
				"course_name": course_name or course_code,
				"credits": effective_credits,
				"major_match": major_match,
				"has_name": bool(course_name),
				"has_catalog_credits": catalog_credits > 0,
				"is_special_topic": special_topic,
				"is_low_credit": effective_credits < 6,
				"citation": citation,
				"citation_hit": citation_hit,
			}
		)

	if major_prefixes:
		target_slots = max(1, max_credits // 3) if max_credits > 0 else 4
		major_candidate_count = sum(1 for item in candidates if item["major_match"])
		if major_candidate_count >= min(3, target_slots):
			candidates = [item for item in candidates if item["major_match"]]

	candidates.sort(
		key=lambda item: (
			not item["major_match"],
			not item["has_name"],
			not item["has_catalog_credits"],
			item["is_special_topic"],
			item["is_low_credit"],
			not item["citation_hit"],
			item["course_code"],
		)
	)

	plan: list[dict[str, Any]] = []
	total = 0
	seen_aliases: set[str] = set()
	for candidate in candidates:
		course_code = str(candidate["course_code"])
		alias_key = _course_alias_key(course_code)
		if alias_key in seen_aliases:
			continue

		credits = int(candidate["credits"])
		if max_credits > 0 and total + credits > max_credits:
			continue

		seen_aliases.add(alias_key)
		plan.append(
			{
				"course_code": course_code,
				"course_name": str(candidate["course_name"]),
				"credits": credits,
				"justification": _build_course_justification(
					course_code,
					target_major=target_major,
					major_match=bool(candidate["major_match"]),
				),
				"citation": candidate["citation"],
			}
		)
		total += credits

	return plan


class PlannerAgent:
	def __init__(self) -> None:
		self._parser = PydanticOutputParser(pydantic_object=PlannerStructuredOutput)
		self._prereq_chain = (
			RunnableLambda(self._build_prereq_prompt_payload)
			| RunnableLambda(self._invoke_structured_llm)
			| RunnableLambda(self._parse_structured_payload)
		)
		self._plan_chain = (
			RunnableLambda(self._build_plan_prompt_payload)
			| RunnableLambda(self._invoke_structured_llm)
			| RunnableLambda(self._parse_structured_payload)
		)

	def _build_prereq_prompt_payload(self, chain_input: dict[str, Any]) -> dict[str, Any]:
		question = str(chain_input.get("question", "") or "")
		graph_result = chain_input.get("graph_result", {})
		context_string = str(chain_input.get("context_string", "") or "")
		profile = chain_input.get("profile", {}) or {}
		missing_fields = profile.get("missing_fields", [])
		clarifying_questions = profile.get("clarifying_questions", [])
		compacted_graph_result = _compact_graph_result(graph_result)

		system_prompt, user_prompt = render_prereq_prompt(
			context=context_string,
			question=question,
			graph_result=json.dumps(compacted_graph_result, ensure_ascii=True),
		)
		if missing_fields:
			user_prompt = (
				f"{user_prompt}\n\n"
				f"Known missing student fields: {json.dumps(missing_fields, ensure_ascii=True)}\n"
				"If these missing fields prevent a reliable decision, ask concise clarifying questions."
			)
		if clarifying_questions:
			user_prompt = (
				f"{user_prompt}\n"
				f"Suggested clarifying questions: {json.dumps(clarifying_questions, ensure_ascii=True)}"
			)
		return {
			"system_prompt": system_prompt,
			"user_prompt": user_prompt,
			"temperature": settings.TEMP_REASONING,
			"fallback_answer": "",
		}

	def _build_plan_prompt_payload(self, chain_input: dict[str, Any]) -> dict[str, Any]:
		profile = chain_input.get("profile", {})
		graph_result = chain_input.get("graph_result", {})
		context_string = str(chain_input.get("context_string", "") or "")
		compacted_profile = _compact_profile(profile)
		compacted_graph_result = _compact_graph_result(graph_result)

		system_prompt, user_prompt = render_plan_prompt(
			profile=json.dumps(compacted_profile, ensure_ascii=True),
			context=context_string,
			graph_result=json.dumps(compacted_graph_result, ensure_ascii=True),
		)
		return {
			"system_prompt": system_prompt,
			"user_prompt": user_prompt,
			"temperature": settings.TEMP_GENERATION,
			"fallback_answer": "",
		}

	async def _invoke_structured_llm(self, payload: dict[str, Any]) -> dict[str, Any]:
		structured_user_prompt = (
			f"{payload.get('user_prompt', '')}\n\n"
			"Return ONLY valid JSON (no markdown fences) with keys: "
			"answer (string), clarifying_questions (array of strings), assumptions (array of strings), risks (array of strings)."
		)
		raw_output = await call_llm(
			str(payload.get("system_prompt", "")),
			structured_user_prompt,
			float(payload.get("temperature", settings.TEMP_REASONING)),
		)
		return {
			"raw_output": raw_output,
			"fallback_answer": str(payload.get("fallback_answer", "") or ""),
		}

	def _parse_structured_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
		raw_output = str(payload.get("raw_output", "") or "").strip()
		fallback_answer = str(payload.get("fallback_answer", "") or "")

		candidate = raw_output
		if "```" in raw_output:
			match = re.search(r"\{.*\}", raw_output, re.DOTALL)
			if match:
				candidate = match.group(0)

		try:
			parsed = self._parser.parse(candidate)
		except Exception:
			answer = raw_output or fallback_answer
			parsed = PlannerStructuredOutput(answer=answer)

		if not parsed.answer:
			parsed.answer = raw_output or fallback_answer

		return parsed.model_dump()

	async def run(
		self,
		profile: dict[str, Any],
		vector_ctx: dict[str, Any],
		graph_result: dict[str, Any],
		mode: str,
		question: str,
	) -> dict[str, Any]:
		citations = vector_ctx.get("citations", [])
		retrieved_chunks = vector_ctx.get("retrieved_chunks", [])
		context_string = vector_ctx.get("context_string", "")

		if not context_string:
			return {
				"answer": ABSTENTION_MSG,
				"plan": [],
				"citations": [],
				"clarifying_questions": profile.get("clarifying_questions", []),
				"assumptions": ["No relevant catalog context retrieved."],
				"risks": ["Response may be incomplete due to missing context."],
			}

		structured_result: dict[str, Any] = {
			"answer": "",
			"clarifying_questions": [],
			"assumptions": [],
			"risks": [],
		}
		try:
			if settings.GROQ_API_KEY.startswith("dev-placeholder"):
				raise RuntimeError("placeholder key")

			if mode == "plan":
				structured_result = await self._plan_chain.ainvoke(
					{
						"profile": profile,
						"context_string": context_string,
						"graph_result": graph_result,
					}
				)
			else:
				structured_result = await self._prereq_chain.ainvoke(
					{
						"question": question,
						"context_string": context_string,
						"graph_result": graph_result,
						"profile": profile,
					}
				)
		except Exception:
			log.exception("Planner LLM call failed; using deterministic fallback")
			if mode == "prereq":
				decision = str(graph_result.get("decision", "Need more info"))
				evidence_lines = []
				for citation in citations[:3]:
					evidence_lines.append(
						"[SOURCE: "
						f"{citation.get('document_name', '')}, "
						f"Section: {citation.get('section_heading', '')}, "
						f"Chunk: {citation.get('chunk_id', '')}]"
					)
				next_step = "Share missing prerequisite details with your advisor if any requirement is unclear."
				answer = (
					f"DECISION: {decision}\n"
					f"EVIDENCE: {'; '.join(evidence_lines) if evidence_lines else 'No high-confidence evidence retrieved.'}\n"
					f"NEXT STEP: {next_step}"
				)
				structured_result = PlannerStructuredOutput(answer=answer).model_dump()
			elif mode == "ask":
				structured_result = PlannerStructuredOutput(answer=ABSTENTION_MSG).model_dump()
			else:
				answer = (
					"Answer / Plan: Suggested next courses based on available prerequisites.\n"
					"Why (requirements/prereqs satisfied): Requirements appear satisfied for listed options.\n"
					"Citations: Use retrieved source entries below."
				)
				structured_result = PlannerStructuredOutput(answer=answer).model_dump()

		answer = str(structured_result.get("answer", "") or "")

		available_courses = graph_result.get("available_next_courses", [])
		plan = _build_plan_from_graph(
			available_courses=available_courses,
			retrieved_chunks=retrieved_chunks,
			max_credits=int(profile.get("max_credits", 0) or 0),
			answer_text=answer,
			graph_result=graph_result,
			target_major=str(profile.get("target_major", "") or "").strip(),
			completed_courses=profile.get("completed_courses", []),
		)
		plan_codes = {
			normalize_course_code(str(course.get("course_code", "") or ""))
			for course in plan
			if isinstance(course, dict)
			and str(course.get("course_code", "") or "").strip()
		}

		assumptions = [str(item) for item in structured_result.get("assumptions", []) if str(item).strip()]
		risks = [str(item) for item in structured_result.get("risks", []) if str(item).strip()]
		if mode == "plan":
			assumptions = _sanitize_plan_notes(assumptions, plan_codes, max_items=4)
			risks = _sanitize_plan_notes(risks, plan_codes, max_items=4)
		else:
			assumptions = _dedupe_text_items(assumptions, max_items=5)
			risks = _dedupe_text_items(risks, max_items=5)
		if not citations:
			risks.append("No citations were retrieved for this response.")
		if mode == "plan":
			risks = _sanitize_plan_notes(risks, plan_codes, max_items=4)
		else:
			risks = _dedupe_text_items(risks, max_items=5)

		clarifying = structured_result.get("clarifying_questions", [])
		if not clarifying:
			clarifying = profile.get("clarifying_questions", [])

		answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
		return {
			"answer": answer,
			"plan": plan,
			"citations": citations,
			"clarifying_questions": [
				str(item)
				for item in clarifying
				if str(item).strip()
			],
			"assumptions": assumptions,
			"risks": risks,
		}


class PlannerStructuredOutput(BaseModel):
	answer: str = ""
	clarifying_questions: list[str] = Field(default_factory=list)
	assumptions: list[str] = Field(default_factory=list)
	risks: list[str] = Field(default_factory=list)
