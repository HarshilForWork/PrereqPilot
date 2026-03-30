from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class IngestRequest(BaseModel):
	force_reingest: bool = False


def _normalize_query_payload(data: Any) -> Any:
	if isinstance(data, str):
		return {"question": data}
	if not isinstance(data, dict):
		return data

	normalized = dict(data)
	if not normalized.get("question"):
		for key in ("query", "q", "input"):
			value = normalized.get(key)
			if isinstance(value, str) and value.strip():
				normalized["question"] = value.strip()
				break
	return normalized


class StudentProfilePrereqModel(BaseModel):
	completed_courses: list[str] = Field(default_factory=list)
	grades: dict[str, str] = Field(default_factory=dict)
	gpa: float | None = None
	student_level: str = ""
	include_financial_aid_policies: bool = False
	catalog_year: str = ""


class PrereqQueryRequest(BaseModel):
	question: str = Field(validation_alias=AliasChoices("question", "query", "q", "input"), min_length=1)
	model_config = ConfigDict(
		extra="allow",
		json_schema_extra={
			"examples": [
				{"query": "Can I take 6.1010 if I have completed 6.1000?"},
			]
		},
	)

	@model_validator(mode="before")
	@classmethod
	def _coerce_query_input(cls, data: Any) -> Any:
		return _normalize_query_payload(data)


class StudentProfilePlanModel(BaseModel):
	completed_courses: list[str] = Field(default_factory=list)
	grades: dict[str, str] = Field(default_factory=dict)
	gpa: float | None = None
	student_level: str = ""
	target_major: str = ""
	target_term: str = ""
	max_credits: int = 0
	catalog_year: str = ""


class PlanQueryRequest(BaseModel):
	question: str = Field(validation_alias=AliasChoices("question", "query", "q", "input"), min_length=1)
	model_config = ConfigDict(
		extra="allow",
		json_schema_extra={
			"examples": [
				{"query": "Plan for Fall 2026 for 6-3 major after 6.1000 with max 12 credits"},
			]
		},
	)

	@model_validator(mode="before")
	@classmethod
	def _coerce_query_input(cls, data: Any) -> Any:
		return _normalize_query_payload(data)


class AskQueryRequest(BaseModel):
	question: str = Field(validation_alias=AliasChoices("question", "query", "q", "input"), min_length=1)
	model_config = ConfigDict(
		extra="allow",
		json_schema_extra={
			"examples": [
				{"query": "How many GIR subjects are required?"},
			]
		},
	)

	@model_validator(mode="before")
	@classmethod
	def _coerce_query_input(cls, data: Any) -> Any:
		return _normalize_query_payload(data)
