from fastapi.testclient import TestClient

from app import app
from src.api.routes import query as query_routes


class _FakeRetriever:
	async def run(self, query: str, filters=None):
		_ = query
		_ = filters
		return {
			"retrieved_chunks": [
				{
					"content": "CS201 requires CS101",
					"metadata": {
						"chunk_id": "chunk_1",
						"document_name": "catalog.pdf",
						"section_heading": "Prerequisites",
					},
				}
			],
			"citations": [
				{
					"chunk_id": "chunk_1",
					"document_name": "catalog.pdf",
					"section_heading": "Prerequisites",
				}
			],
			"context_string": "CS201 requires CS101",
		}


class _FakePlanner:
	async def run(self, profile, vector_ctx, graph_result, mode, question):
		_ = profile
		_ = vector_ctx
		_ = graph_result
		_ = mode
		_ = question
		return {
			"answer": "DECISION: Eligible",
			"plan": [
				{
					"course_code": "CS201",
					"course_name": "Data Structures",
					"credits": 3,
					"justification": "Prerequisite satisfied.",
					"citation": {
						"chunk_id": "chunk_1",
						"document_name": "catalog.pdf",
						"section_heading": "Prerequisites",
					},
				}
			],
			"citations": [
				{
					"chunk_id": "chunk_1",
					"document_name": "catalog.pdf",
					"section_heading": "Prerequisites",
				}
			],
			"clarifying_questions": [],
			"assumptions": [],
			"risks": [],
		}


class _FakeVerifier:
	async def run(self, draft, retrieved_docs, graph_result):
		_ = retrieved_docs
		_ = graph_result
		return {"passed": True, "issues": [], "final_response": draft}


def test_query_prereq_route(monkeypatch):
	monkeypatch.setattr(query_routes, "retriever_agent", _FakeRetriever())
	monkeypatch.setattr(query_routes, "planner_agent", _FakePlanner())
	monkeypatch.setattr(query_routes, "verifier_agent", _FakeVerifier())
	monkeypatch.setattr(
		query_routes,
		"check_eligibility",
		lambda course, completed, grades: {
			"eligible": True,
			"decision": "Eligible",
			"missing_prereqs": [],
			"missing_coreqs": [],
			"grade_issues": [],
			"prereq_path": ["CS101"],
			"either_or_options": [],
		},
	)

	client = TestClient(app)
	response = client.post(
		"/query/prereq",
		json={
			"question": "Can I take CS201 after CS101?",
			"student_profile": {"completed_courses": ["CS101"], "grades": {"CS101": "B"}},
		},
	)

	assert response.status_code == 200
	body = response.json()
	assert body["decision"] in {"Eligible", "Not eligible", "Need more info"}
	assert "trace_id" in body


def test_query_ask_route(monkeypatch):
	monkeypatch.setattr(query_routes, "retriever_agent", _FakeRetriever())
	monkeypatch.setattr(query_routes, "planner_agent", _FakePlanner())
	monkeypatch.setattr(query_routes, "verifier_agent", _FakeVerifier())

	client = TestClient(app)
	response = client.post("/query/ask", json={"question": "Tell me about CS201", "filters": None})
	assert response.status_code == 200
	body = response.json()
	assert "answer" in body
	assert "abstained" in body
	assert "trace_id" in body
