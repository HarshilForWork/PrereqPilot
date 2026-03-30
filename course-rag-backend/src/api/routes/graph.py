from __future__ import annotations

import time

from fastapi import APIRouter, Query, Request

from src.api.models.responses import GraphAllResponse, GraphCourseResponse, HealthResponse, MetricsResponse
from src.core.exceptions import GraphNodeNotFoundError
from src.graph.reasoning import find_path_to_course, get_full_prereq_chain
from src.graph.store import get_graph
from src.ops.metrics import snapshot
from src.retrieval.chroma_store import get_collection

router = APIRouter()


@router.get(
	"/course/{course_code}",
	response_model=GraphCourseResponse,
	summary="Get Course Graph Details",
	description="Returns node metadata, direct prerequisites, enabled courses, and transitive prerequisite chain for a course code.",
)
async def graph_course(course_code: str) -> GraphCourseResponse:
	graph = get_graph()
	code = course_code.upper().strip().replace(" ", "")
	if code not in graph:
		raise GraphNodeNotFoundError(f"Course {code} not found in graph")

	node_attrs = dict(graph.nodes[code])
	prerequisites = sorted(list(graph.predecessors(code)))
	enables = sorted(list(graph.successors(code)))
	full_chain = get_full_prereq_chain(code)

	return GraphCourseResponse(
		course_code=code,
		course_name=str(node_attrs.get("course_name", code)),
		credits=int(node_attrs.get("credits", 0) or 0),
		prerequisites=prerequisites,
		enables=enables,
		full_prereq_chain=full_chain,
		node_attributes=node_attrs,
	)


@router.get(
	"/path",
	summary="Find Prerequisite Path",
	description="Builds a prerequisite path payload from an optional starting course to a target course, with completed courses marked.",
)
async def graph_path(
	from_code: str = Query(alias="from"),
	to_code: str = Query(alias="to"),
	completed: str = "",
) -> dict:
	completed_courses = [course.strip().upper() for course in completed.split(",") if course.strip()]
	from_clean = from_code.strip().upper().replace(" ", "")
	if from_clean and from_clean not in completed_courses:
		completed_courses.append(from_clean)
	return find_path_to_course(to_code, completed_courses)


@router.get(
	"/all",
	response_model=GraphAllResponse,
	summary="Get Full Graph Snapshot",
	description="Returns all graph nodes, edges, and graph-level metadata for visualization or debugging.",
)
async def graph_all() -> GraphAllResponse:
	graph = get_graph()
	nodes = [{"id": node, **attrs} for node, attrs in graph.nodes(data=True)]
	edges = [{"from": src, "to": dst, **attrs} for src, dst, attrs in graph.edges(data=True)]
	return GraphAllResponse(nodes=nodes, edges=edges, metadata=dict(graph.graph))


@router.get(
	"/health",
	response_model=HealthResponse,
	summary="Service Health",
	description="Reports API health with Chroma collection size, graph size, and service uptime.",
)
async def health(request: Request) -> HealthResponse:
	graph = get_graph()
	chroma_count = get_collection().count()
	started_at = float(getattr(request.app.state, "started_at", time.time()))
	uptime = max(0.0, time.time() - started_at)
	status = "ok" if chroma_count >= 0 else "degraded"

	return HealthResponse(
		status=status,
		chroma_document_count=int(chroma_count),
		graph_nodes=graph.number_of_nodes(),
		graph_edges=graph.number_of_edges(),
		uptime_seconds=uptime,
	)


@router.get(
	"/ops/metrics",
	response_model=MetricsResponse,
	summary="Get Operational Metrics",
	description="Returns in-memory request, latency, abstention, and citation counters used for runtime monitoring.",
)
async def ops_metrics() -> MetricsResponse:
	return MetricsResponse(**snapshot())
