from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CitationModel(BaseModel):
	chunk_id: str
	document_name: str
	section_heading: str = ""


class IngestStartResponse(BaseModel):
	job_id: str
	message: str


class IngestStatusResponse(BaseModel):
	job_id: str
	status: str
	progress: float
	detail: str
	started_at: float
	completed_at: float | None


class GraphResultModel(BaseModel):
	eligible: bool
	missing_prereqs: list[str] = Field(default_factory=list)
	missing_coreqs: list[str] = Field(default_factory=list)
	grade_issues: list[str] = Field(default_factory=list)
	gpa_issues: list[str] = Field(default_factory=list)
	prereq_path: list[str] = Field(default_factory=list)
	either_or_options: list[list[str]] = Field(default_factory=list)


class PrereqQueryResponse(BaseModel):
	decision: Literal["Eligible", "Not eligible", "Need more info"]
	answer: str
	evidence: str
	citations: list[CitationModel] = Field(default_factory=list)
	next_step: str
	clarifying_questions: list[str] = Field(default_factory=list)
	graph_result: GraphResultModel
	assumptions: list[str] = Field(default_factory=list)
	trace_id: str


class PlanCourseModel(BaseModel):
	course_code: str
	course_name: str
	credits: int
	justification: str
	citation: CitationModel


class PlanQueryResponse(BaseModel):
	plan: list[PlanCourseModel] = Field(default_factory=list)
	total_credits: int = 0
	citations: list[CitationModel] = Field(default_factory=list)
	clarifying_questions: list[str] = Field(default_factory=list)
	assumptions: list[str] = Field(default_factory=list)
	risks: list[str] = Field(default_factory=list)
	trace_id: str


class AskQueryResponse(BaseModel):
	answer: str
	citations: list[CitationModel] = Field(default_factory=list)
	clarifying_questions: list[str] = Field(default_factory=list)
	abstained: bool
	trace_id: str


class GraphCourseResponse(BaseModel):
	course_code: str
	course_name: str
	credits: int
	prerequisites: list[str] = Field(default_factory=list)
	enables: list[str] = Field(default_factory=list)
	full_prereq_chain: list[str] = Field(default_factory=list)
	node_attributes: dict[str, Any] = Field(default_factory=dict)


class GraphAllResponse(BaseModel):
	nodes: list[dict[str, Any]] = Field(default_factory=list)
	edges: list[dict[str, Any]] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
	status: Literal["ok", "degraded"]
	chroma_document_count: int
	graph_nodes: int
	graph_edges: int
	uptime_seconds: float


class MetricsResponse(BaseModel):
	total_requests: int
	prereq_requests: int
	plan_requests: int
	abstentions: int
	verifier_rewrites: int
	citation_hits: int
	avg_latency_ms: float
	avg_chunks_retrieved: float
