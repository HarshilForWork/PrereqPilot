from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.constants import COURSE_CODE_PAT, normalize_course_code
from src.core.logger import get_logger
from src.inference.groq_client import call_llm

log = get_logger(__name__)
_MAX_CLARIFY_QUESTIONS = 5


def _normalize_course_codes(courses: list[str] | None) -> list[str]:
	if not courses:
		return []
	return sorted({course.upper().strip().replace(" ", "") for course in courses if course})


def _normalize_grades(grades: dict[str, str] | None) -> dict[str, str]:
	if not grades:
		return {}
	return {
		course.upper().strip().replace(" ", ""): grade.upper().strip()
		for course, grade in grades.items()
		if course
	}


def _normalize_gpa(gpa: Any) -> float | None:
	if gpa in (None, ""):
		return None
	try:
		return float(gpa)
	except (TypeError, ValueError):
		return None


def _normalize_bool(value: Any, default: bool = False) -> bool:
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		lowered = value.strip().lower()
		if lowered in {"true", "1", "yes", "y", "on"}:
			return True
		if lowered in {"false", "0", "no", "n", "off"}:
			return False
	return default


def _extract_course_codes_from_text(text: str) -> list[str]:
	if not text:
		return []
	ordered: list[str] = []
	seen: set[str] = set()
	for match in COURSE_CODE_PAT.finditer(text):
		code = normalize_course_code(match.group(0))
		if code and code not in seen:
			seen.add(code)
			ordered.append(code)
	return ordered


def _question_has_profile_signal(question: str) -> bool:
	lowered = question.lower()
	signals = (
		"completed",
		"after",
		"with",
		"without",
		"taken",
		"took",
		"having completed",
	)
	return any(signal in lowered for signal in signals)


def _infer_completed_courses_from_question(question: str) -> list[str]:
	all_codes = _extract_course_codes_from_text(question)
	if len(all_codes) <= 1:
		return []

	target_course = all_codes[0]

	positive_patterns = [
		r"\bhave\s+completed\b(?P<segment>[^?.!;]*)",
		r"\bcompleted\b(?P<segment>[^?.!;]*)",
		r"\bhaving\s+completed\b(?P<segment>[^?.!;]*)",
		r"\bafter\b(?P<segment>[^?.!;]*)",
		r"\bwith\b(?P<segment>[^?.!;]*)",
		r"\btaken\b(?P<segment>[^?.!;]*)",
		r"\btook\b(?P<segment>[^?.!;]*)",
	]
	negative_patterns = [
		r"\bwithout\b(?P<segment>[^?.!;]*)",
	]

	inferred: list[str] = []
	for pattern in positive_patterns:
		for match in re.finditer(pattern, question, flags=re.IGNORECASE):
			inferred.extend(_extract_course_codes_from_text(match.group("segment") or ""))

	excluded: set[str] = set()
	for pattern in negative_patterns:
		for match in re.finditer(pattern, question, flags=re.IGNORECASE):
			excluded.update(_extract_course_codes_from_text(match.group("segment") or ""))

	ordered: list[str] = []
	seen: set[str] = set()
	for code in inferred:
		if code == target_course or code in excluded:
			continue
		if code not in seen:
			seen.add(code)
			ordered.append(code)

	if ordered:
		return ordered

	lowered = question.lower()
	if any(token in lowered for token in ("completed", "after", "with", "taken", "took")) and "without" not in lowered:
		for code in all_codes[1:]:
			if code != target_course and code not in excluded and code not in seen:
				seen.add(code)
				ordered.append(code)

	return ordered


def _infer_plan_profile_from_question(question: str) -> dict[str, Any]:
	if not question:
		return {}

	positive_patterns = [
		r"\bafter\b(?P<segment>[^?.!;]*)",
		r"\bcompleted\b(?P<segment>[^?.!;]*)",
		r"\bwith\b(?P<segment>[^?.!;]*)",
		r"\btaken\b(?P<segment>[^?.!;]*)",
		r"\btook\b(?P<segment>[^?.!;]*)",
	]

	completed: list[str] = []
	for pattern in positive_patterns:
		for match in re.finditer(pattern, question, flags=re.IGNORECASE):
			completed.extend(_extract_course_codes_from_text(match.group("segment") or ""))

	if not completed and "after" in question.lower():
		completed = _extract_course_codes_from_text(question)

	max_credits = 0
	credit_patterns = [
		r"\bmax(?:imum)?\s*(?P<value>\d{1,2})\s*(?:credits?|units?)\b",
		r"\b(?P<value>\d{1,2})\s*(?:credits?|units?)\b",
	]
	for pattern in credit_patterns:
		match = re.search(pattern, question, flags=re.IGNORECASE)
		if match:
			try:
				max_credits = int(match.group("value"))
				break
			except (TypeError, ValueError):
				pass

	target_term = ""
	term_match = re.search(r"\b(?P<term>fall|spring|summer|winter)\s+(?P<year>20\d{2})\b", question, re.IGNORECASE)
	if term_match:
		target_term = f"{term_match.group('term').title()} {term_match.group('year')}"

	target_major = ""
	major_patterns = [
		r"\bfor\s+(?P<major>[A-Za-z0-9\-]+)\s+major\b",
		r"\bmajor\s*(?:is|:)?\s*(?P<major>[A-Za-z0-9\-]+)\b",
	]
	for pattern in major_patterns:
		match = re.search(pattern, question, flags=re.IGNORECASE)
		if match:
			target_major = str(match.group("major") or "").strip()
			break

	return {
		"completed_courses": _normalize_course_codes(completed),
		"max_credits": max_credits,
		"target_term": target_term,
		"target_major": target_major,
	}


class IntakeAgent:
	def __init__(self) -> None:
		self._chain = RunnableLambda(self._normalize_profile)

	def _normalize_profile(self, chain_input: dict[str, Any]) -> dict[str, Any]:
		raw_input = dict(chain_input.get("raw_input", {}) or {})
		mode = str(chain_input.get("mode", "prereq") or "prereq")
		question = str(raw_input.get("question", "") or "")
		inferred_plan_profile = _infer_plan_profile_from_question(question) if mode == "plan" else {}

		profile = raw_input.get("student_profile", raw_input) or {}
		completed_courses = _normalize_course_codes(profile.get("completed_courses", []))
		if mode == "prereq" and not completed_courses:
			completed_courses = _infer_completed_courses_from_question(question)
		if mode == "plan" and not completed_courses:
			completed_courses = inferred_plan_profile.get("completed_courses", [])
		grades = _normalize_grades(profile.get("grades", {}))

		target_major = str(profile.get("target_major", "") or "").strip() or str(
			inferred_plan_profile.get("target_major", "") or ""
		).strip()
		target_term = str(profile.get("target_term", "") or "").strip() or str(
			inferred_plan_profile.get("target_term", "") or ""
		).strip()
		profile_max_credits = int(profile.get("max_credits", 0) or 0)
		max_credits = profile_max_credits if profile_max_credits > 0 else int(
			inferred_plan_profile.get("max_credits", 0) or 0
		)

		payload = {
			"student_id": str(profile.get("student_id", "anonymous")),
			"completed_courses": completed_courses,
			"grades": grades,
			"gpa": _normalize_gpa(profile.get("gpa", None)),
			"student_level": str(profile.get("student_level", "")),
			"include_financial_aid_policies": _normalize_bool(
				profile.get("include_financial_aid_policies", False),
				default=False,
			),
			"target_major": target_major,
			"target_term": target_term,
			"max_credits": max_credits,
			"catalog_year": str(profile.get("catalog_year", "")),
			"missing_fields": [],
			"clarifying_questions": [],
		}

		required: list[str] = []
		if mode == "prereq":
			required = ["completed_courses"]
		elif mode == "plan":
			required = ["completed_courses", "target_major", "target_term", "max_credits"]

		missing: list[str] = []
		for field in required:
			value = payload.get(field)
			if field == "completed_courses" and mode == "prereq" and _question_has_profile_signal(question):
				continue
			if value in (None, "", [], {}, 0):
				missing.append(field)

		if not payload["catalog_year"]:
			payload["catalog_year"] = ""

		payload["missing_fields"] = missing
		payload["clarifying_questions"] = []
		return IntakeChainOutput.model_validate(payload).model_dump()

	async def run(self, raw_input: dict[str, Any], mode: str = "prereq") -> dict[str, Any]:
		result = await self._chain.ainvoke({"raw_input": raw_input, "mode": mode})
		normalized = IntakeChainOutput.model_validate(result).model_dump()
		question = str((raw_input or {}).get("question", "") or "")
		normalized["clarifying_questions"] = await self._generate_clarifying_questions(
			question=question,
			mode=mode,
			missing_fields=normalized.get("missing_fields", []),
			profile=normalized,
		)
		return IntakeChainOutput.model_validate(normalized).model_dump()

	def _fallback_clarifying_questions(self, missing_fields: list[str]) -> list[str]:
		questions: list[str] = []
		for field in missing_fields:
			readable = str(field or "").strip().replace("_", " ")
			if not readable:
				continue
			questions.append(f"Could you share your {readable}?")
			if len(questions) >= _MAX_CLARIFY_QUESTIONS:
				break
		return questions

	def _parse_llm_questions(self, text: str) -> list[str]:
		content = str(text or "").strip()
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

		cleaned: list[str] = []
		seen: set[str] = set()
		for value in questions:
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
			if len(cleaned) >= _MAX_CLARIFY_QUESTIONS:
				break

		return cleaned

	async def _generate_clarifying_questions(
		self,
		question: str,
		mode: str,
		missing_fields: list[str],
		profile: dict[str, Any],
	) -> list[str]:
		if not missing_fields:
			return []

		system_prompt = (
			"You generate concise clarification questions for academic advising API requests. "
			"Ask only for information needed to proceed. "
			"Do not ask about fields already provided. "
			"Return ONLY valid JSON as an array of strings with at most 5 items."
		)
		user_prompt = (
			f"Endpoint mode: {mode}\n"
			f"User question: {question}\n"
			f"Missing fields: {json.dumps(missing_fields, ensure_ascii=True)}\n"
			f"Known profile: {json.dumps(profile, ensure_ascii=True)}\n"
			"Generate targeted follow-up questions now."
		)

		try:
			if settings.GROQ_API_KEY.startswith("dev-placeholder"):
				raise RuntimeError("placeholder key")
			raw = await call_llm(system_prompt, user_prompt, settings.TEMP_CLARIFY)
			parsed = self._parse_llm_questions(raw)
			if parsed:
				return parsed
		except Exception:
			log.exception("Intake clarifying-question generation failed; using field-based fallback")

		return self._fallback_clarifying_questions(missing_fields)


class IntakeChainOutput(BaseModel):
	student_id: str = "anonymous"
	completed_courses: list[str] = Field(default_factory=list)
	grades: dict[str, str] = Field(default_factory=dict)
	gpa: float | None = None
	student_level: str = ""
	include_financial_aid_policies: bool = False
	target_major: str = ""
	target_term: str = ""
	max_credits: int = 0
	catalog_year: str = ""
	missing_fields: list[str] = Field(default_factory=list)
	clarifying_questions: list[str] = Field(default_factory=list)
