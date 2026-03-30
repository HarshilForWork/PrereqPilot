from fastapi.testclient import TestClient
from app import app
from src.core.config import settings

settings.CREWAI_ENABLED = True
settings.CREWAI_VERBOSE = False
settings.CREWAI_MEMORY = True

cases = [
    (
        "prereq",
        "/query/prereq",
        {
            "question": "Can I take 6.1040 if I completed 6.1020 and 6.1200[J]?",
            "student_profile": {
                "completed_courses": ["6.1020", "6.1200[J]"],
                "grades": {},
            },
        },
    ),
    (
        "plan",
        "/query/plan",
        {
            "student_profile": {
                "completed_courses": ["6.1020", "6.1200[J]"],
                "grades": {},
                "target_major": "6-3",
                "target_term": "Fall 2026",
                "max_credits": 12,
            }
        },
    ),
    (
        "ask",
        "/query/ask",
        {
            "question": "What are the prerequisites for 6.1040?",
            "filters": None,
        },
    ),
]

with TestClient(app) as client:
    for name, path, payload in cases:
        r = client.post(path, json=payload)
        body = r.json()
        if isinstance(body, dict):
            citations = body.get("citations", [])
            print(
                "SMOKE_RESULT",
                name,
                "status=", r.status_code,
                "decision=", body.get("decision"),
                "abstained=", body.get("abstained"),
                "citations=", len(citations) if isinstance(citations, list) else "n/a",
                "trace=", bool(body.get("trace_id")),
            )
        else:
            print("SMOKE_RESULT", name, "status=", r.status_code, "non_dict")
