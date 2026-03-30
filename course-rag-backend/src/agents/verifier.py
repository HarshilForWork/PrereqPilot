from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.constants import ABSTENTION_MSG
from src.core.logger import get_logger
from src.inference.chains import render_verifier_prompt
from src.inference.groq_client import call_llm

log = get_logger(__name__)

_MAX_VERIFIER_CONTEXT_CHUNKS = 4
_MAX_VERIFIER_CHARS_PER_CHUNK = 320
_MAX_VERIFIER_CONTEXT_CHARS = 2800
_MAX_VERIFIER_LIST_ITEMS = 8
_MAX_VERIFIER_VALUE_CHARS = 200
_MAX_VERIFIER_ANSWER_CHARS = 900


def _compact_text(value: Any, max_chars: int) -> str:
	text = " ".join(str(value or "").split())
	if max_chars <= 3:
		return text[:max_chars]
	if len(text) <= max_chars:
		return text
	return f"{text[: max_chars - 3].rstrip()}..."


def _compact_list(values: list[Any], max_items: int = _MAX_VERIFIER_LIST_ITEMS) -> list[str]:
	compacted: list[str] = []
	for item in values[:max_items]:
		if isinstance(item, dict):
			compacted.append(_compact_text(json.dumps(item, ensure_ascii=True), _MAX_VERIFIER_VALUE_CHARS))
		else:
			compacted.append(_compact_text(item, _MAX_VERIFIER_VALUE_CHARS))
	return compacted


def _compact_citation(citation: dict[str, Any]) -> dict[str, str]:
	return {
		"chunk_id": _compact_text(citation.get("chunk_id", ""), 48),
		"document_name": _compact_text(citation.get("document_name", ""), 80),
		"section_heading": _compact_text(citation.get("section_heading", ""), 80),
	}


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
			if len(value) > _MAX_VERIFIER_LIST_ITEMS:
				compacted[f"{key}_omitted_count"] = len(value) - _MAX_VERIFIER_LIST_ITEMS

	return compacted


def _compact_draft(draft: dict[str, Any]) -> dict[str, Any]:
	if not isinstance(draft, dict):
		return {}

	compacted: dict[str, Any] = {}
	for key in ("decision", "next_step"):
		if key in draft and draft.get(key) not in (None, ""):
			compacted[key] = _compact_text(draft.get(key, ""), _MAX_VERIFIER_VALUE_CHARS)

	if "abstained" in draft:
		compacted["abstained"] = bool(draft.get("abstained", False))

	answer = draft.get("answer", "")
	if str(answer).strip():
		compacted["answer"] = _compact_text(answer, _MAX_VERIFIER_ANSWER_CHARS)

	evidence = draft.get("evidence", "")
	if str(evidence).strip():
		compacted["evidence"] = _compact_text(evidence, _MAX_VERIFIER_ANSWER_CHARS)

	citations = draft.get("citations", [])
	if isinstance(citations, list) and citations:
		compacted["citations"] = [
			_compact_citation(citation) for citation in citations[:_MAX_VERIFIER_LIST_ITEMS] if isinstance(citation, dict)
		]
		if len(citations) > _MAX_VERIFIER_LIST_ITEMS:
			compacted["citations_omitted_count"] = len(citations) - _MAX_VERIFIER_LIST_ITEMS

	plan = draft.get("plan", [])
	if isinstance(plan, list) and plan:
		plan_preview: list[dict[str, Any]] = []
		for item in plan[:_MAX_VERIFIER_LIST_ITEMS]:
			if isinstance(item, dict):
				plan_preview.append(
					{
						"course_code": _compact_text(item.get("course_code", ""), 24),
						"credits": item.get("credits", 0),
					}
				)
			else:
				plan_preview.append({"course_code": _compact_text(item, 24), "credits": 0})
		compacted["plan_preview"] = plan_preview
		if len(plan) > _MAX_VERIFIER_LIST_ITEMS:
			compacted["plan_omitted_count"] = len(plan) - _MAX_VERIFIER_LIST_ITEMS

	for key in ("clarifying_questions", "assumptions", "risks"):
		values = draft.get(key, [])
		if isinstance(values, list) and values:
			compacted[key] = _compact_list(values)
			if len(values) > _MAX_VERIFIER_LIST_ITEMS:
				compacted[f"{key}_omitted_count"] = len(values) - _MAX_VERIFIER_LIST_ITEMS

	return compacted


def _build_compact_context(retrieved_docs: list[dict[str, Any]]) -> str:
	context_lines: list[str] = []
	total_chars = 0
	limited_docs = retrieved_docs[:_MAX_VERIFIER_CONTEXT_CHUNKS]

	for doc in limited_docs:
		metadata = doc.get("metadata") or {}
		content = _compact_text(doc.get("content", ""), _MAX_VERIFIER_CHARS_PER_CHUNK)
		line = (
			f"{content}\n"
			"[SOURCE: "
			f"{metadata.get('document_name', '')}, "
			f"Section: {metadata.get('section_heading', '')}, "
			f"Chunk: {metadata.get('chunk_id', '')}]"
		)

		remaining = _MAX_VERIFIER_CONTEXT_CHARS - total_chars
		if remaining <= 0:
			break
		if len(line) > remaining:
			line = _compact_text(line, remaining)

		context_lines.append(line)
		total_chars += len(line) + 2

	omitted = len(retrieved_docs) - len(limited_docs)
	if omitted > 0:
		context_lines.append(f"[Token budget note: omitted {omitted} additional retrieved chunks.]")

	return "\n\n".join(context_lines)


def _extract_json_block(text: str) -> dict[str, Any] | None:
	text = text.strip()
	try:
		return json.loads(text)
	except Exception:
		pass

	match = re.search(r"\{.*\}", text, re.DOTALL)
	if not match:
		return None

	try:
		return json.loads(match.group(0))
	except Exception:
		return None


def _has_required_citation_fields(citation: dict[str, Any]) -> bool:
	return bool(citation.get("chunk_id") and citation.get("document_name"))


class VerifierAgent:
	def __init__(self) -> None:
		self._parser = PydanticOutputParser(pydantic_object=VerifierStructuredOutput)
		self._verifier_chain = (
			RunnableLambda(self._build_prompt_payload)
			| RunnableLambda(self._invoke_structured_llm)
			| RunnableLambda(self._parse_structured_output)
		)

	def _build_prompt_payload(self, chain_input: dict[str, Any]) -> dict[str, Any]:
		draft = chain_input.get("draft", {})
		retrieved_docs = chain_input.get("retrieved_docs", [])
		graph_result = chain_input.get("graph_result", {})

		compacted_draft = _compact_draft(draft)
		compacted_graph_result = _compact_graph_result(graph_result)
		context_string = _build_compact_context(retrieved_docs)

		system_prompt, user_prompt = render_verifier_prompt(
			draft=json.dumps(compacted_draft, ensure_ascii=True),
			context=context_string,
			graph_result=json.dumps(compacted_graph_result, ensure_ascii=True),
		)

		return {
			"system_prompt": system_prompt,
			"user_prompt": user_prompt,
		}

	async def _invoke_structured_llm(self, payload: dict[str, Any]) -> dict[str, Any]:
		structured_user_prompt = (
			f"{payload.get('user_prompt', '')}\n\n"
			"Return ONLY valid JSON (no markdown fences) with keys: "
			"passed (boolean), issues (array of strings), corrected_response (string or null)."
		)
		raw_output = await call_llm(
			str(payload.get("system_prompt", "")),
			structured_user_prompt,
			settings.TEMP_REASONING,
		)
		return {"raw_output": raw_output}

	def _parse_structured_output(self, payload: dict[str, Any]) -> dict[str, Any]:
		raw_output = str(payload.get("raw_output", "") or "").strip()
		candidate = raw_output
		if "```" in raw_output:
			match = re.search(r"\{.*\}", raw_output, re.DOTALL)
			if match:
				candidate = match.group(0)

		try:
			parsed = self._parser.parse(candidate)
		except Exception:
			fallback = _extract_json_block(raw_output)
			if fallback:
				parsed = VerifierStructuredOutput.model_validate(fallback)
			else:
				parsed = VerifierStructuredOutput(passed=False, issues=["Verifier model output was not valid structured JSON."])

		return parsed.model_dump()

	async def run(
		self,
		draft: dict[str, Any],
		retrieved_docs: list[dict[str, Any]],
		graph_result: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		graph_result = graph_result or {}
		issues: list[str] = []

		answer = str(draft.get("answer", ""))
		citations = draft.get("citations", []) or []

		if answer and ABSTENTION_MSG not in answer and not citations:
			issues.append("Response contains factual content without citations.")

		for citation in citations:
			if not _has_required_citation_fields(citation):
				issues.append("Citation missing required fields: chunk_id and document_name.")
				break

		if "decision" in draft and graph_result:
			graph_decision = str(graph_result.get("decision", ""))
			draft_decision = str(draft.get("decision", ""))
			if graph_decision and draft_decision and graph_decision != draft_decision:
				issues.append("Decision is inconsistent with graph reasoning result.")

		corrected = dict(draft)
		if issues:
			if "decision" in corrected and graph_result.get("decision"):
				corrected["decision"] = str(graph_result["decision"])
			if not corrected.get("answer"):
				corrected["answer"] = ABSTENTION_MSG

		llm_issues = list(issues)
		if not settings.GROQ_API_KEY.startswith("dev-placeholder"):
			try:
				parsed = await self._verifier_chain.ainvoke(
					{
						"draft": draft,
						"retrieved_docs": retrieved_docs,
						"graph_result": graph_result,
					}
				)
				llm_passed = bool(parsed.get("passed", False))
				llm_issues.extend([str(issue) for issue in parsed.get("issues", []) if issue])
				corrected_response = parsed.get("corrected_response")
				if corrected_response:
					corrected["answer"] = str(corrected_response)
				if llm_passed and not llm_issues:
					return {
						"passed": True,
						"issues": [],
						"final_response": draft,
					}
			except Exception:
				log.exception("LLM verifier pass failed; using heuristic verification result")

		final_issues = sorted(set(llm_issues))
		return {
			"passed": not final_issues,
			"issues": final_issues,
			"final_response": corrected if final_issues else draft,
		}


class VerifierStructuredOutput(BaseModel):
	passed: bool = False
	issues: list[str] = Field(default_factory=list)
	corrected_response: str | None = None
