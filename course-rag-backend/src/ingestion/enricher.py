import re
from typing import Any

from src.core.constants import (
	CATALOG_YEAR_REGEX,
	COREQ_KW,
	COURSE_CODE_PAT,
	CREDIT_KW,
	GRADE_KW,
	is_valid_course_code,
	normalize_course_code,
	PREREQ_KW,
)


def _contains_any(text: str, keywords: list[str]) -> bool:
	lowered = text.lower()
	return any(keyword in lowered for keyword in keywords)


def _coerce_scalar(value: Any) -> str | int | float | bool:
	if value is None:
		return ""
	if isinstance(value, (str, int, float, bool)):
		return value
	if isinstance(value, list):
		return ",".join(str(item) for item in value)
	if isinstance(value, dict):
		return ""
	return str(value)


def _guess_document_type(content_type: str, text: str) -> str:
	if content_type == "table":
		return "table"

	lowered = text.lower()
	if "program requirement" in lowered or "degree requirement" in lowered:
		return "program_requirements"
	if "policy" in lowered or "academic standing" in lowered:
		return "academic_policy"
	return "course_description"


def enrich_chunk_metadata(chunk: dict[str, Any], parser_result: dict[str, Any]) -> dict[str, Any]:
	"""
	Build enriched chunk payload without mutating the original chunk.
	"""
	chunk_meta = dict(chunk.get("metadata", {}))
	content = str(chunk.get("content", ""))
	content_type = str(chunk_meta.get("content_type", "text"))

	course_codes = sorted(
		{
			normalize_course_code(code)
			for code in COURSE_CODE_PAT.findall(content)
			if is_valid_course_code(code)
		}
	)
	has_either_or = bool(
		re.search(
			r"\b(?:\d{1,2}\.[A-Z0-9]+(?:\[J\])?|[A-Z]{1,4}\s?\d{3,4}[A-Z]?)\b\s+or\s+\b(?:\d{1,2}\.[A-Z0-9]+(?:\[J\])?|[A-Z]{1,4}\s?\d{3,4}[A-Z]?)\b",
			content,
			re.IGNORECASE,
		)
	)
	catalog_year_match = CATALOG_YEAR_REGEX.search(content) or CATALOG_YEAR_REGEX.search(
		str(parser_result.get("content", ""))
	)

	parser_metadata = parser_result.get("metadata") or {}
	institution = parser_metadata.get("institution", "") if isinstance(parser_metadata, dict) else ""

	section_heading = ""
	for line in content.splitlines()[:8]:
		candidate = line.strip()
		if 3 <= len(candidate) <= 80 and candidate == candidate.title():
			section_heading = candidate
			break

	metadata = {
		"chunk_id": str(chunk.get("chunk_id", "")),
		"document_name": str(chunk.get("document_name", "")),
		"chunk_index": int(chunk.get("chunk_index", -1)),
		"char_count": int(chunk.get("char_count", len(content))),
		"chunking_method": str(chunk_meta.get("chunking_method", "")),
		"content_type": content_type,
		"source_document": str(chunk_meta.get("source_document", chunk.get("document_name", ""))),
		"chunk_size_config": int(chunk_meta.get("chunk_size_config", 0) or 0),
		"overlap_config": int(chunk_meta.get("overlap_config", 0) or 0),
		"section_heading": section_heading,
		"page_number": -1,
		"document_type": _guess_document_type(content_type, content),
		"course_codes_mentioned": ",".join(course_codes),
		"has_prerequisites": _contains_any(content, PREREQ_KW),
		"has_grade_requirements": _contains_any(content, GRADE_KW),
		"has_credit_info": _contains_any(content, CREDIT_KW),
		"has_corequisite": _contains_any(content, COREQ_KW),
		"has_either_or": has_either_or,
		"institution": str(institution),
		"catalog_year": catalog_year_match.group(1) if catalog_year_match else "",
	}

	metadata = {key: _coerce_scalar(value) for key, value in metadata.items()}

	return {
		"chunk_id": str(chunk.get("chunk_id", "")),
		"document_name": str(chunk.get("document_name", "")),
		"content": content,
		"chunk_index": int(chunk.get("chunk_index", -1)),
		"char_count": int(chunk.get("char_count", len(content))),
		"metadata": metadata,
	}
