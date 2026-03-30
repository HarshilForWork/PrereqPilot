from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.constants import ABSTENTION_MSG
from src.ops.evaluator import run_eval


def _ensure_prereq_profile(profile: dict[str, Any]) -> dict[str, Any]:
    copy = dict(profile or {})
    completed = copy.get("completed_courses", [])
    if not completed:
        copy["completed_courses"] = ["6.100A"]
    copy.setdefault("grades", {})
    return copy


def _ensure_plan_profile(profile: dict[str, Any]) -> dict[str, Any]:
    copy = dict(profile or {})
    completed = copy.get("completed_courses", [])
    if not completed:
        copy["completed_courses"] = ["6.100A", "18.06"]
    copy.setdefault("grades", {})
    if not str(copy.get("target_major", "")).strip():
        copy["target_major"] = "Course 6-3"
    if not str(copy.get("target_term", "")).strip():
        copy["target_term"] = "Fall 2026"
    if int(copy.get("max_credits", 0) or 0) <= 0:
        copy["max_credits"] = 12
    return copy


def _invoke_query(client: httpx.Client, item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    query_type = str(item.get("type", ""))
    question = str(item.get("question", ""))
    profile = dict(item.get("student_profile") or {})

    if query_type in {"prereq_check", "multi_hop_chain"}:
        payload = {
            "question": question,
            "student_profile": _ensure_prereq_profile(profile),
        }
        response = client.post("/query/prereq", json=payload)
        if response.status_code != 200:
            return {
                "decision": None,
                "citations": [],
                "abstained": True,
                "answer": "",
            }, {
                "error": f"HTTP {response.status_code}",
                "endpoint": "/query/prereq",
            }
        body = response.json()
        return {
            "decision": body.get("decision"),
            "citations": body.get("citations", []),
            "abstained": ABSTENTION_MSG in str(body.get("answer", "")),
            "answer": str(body.get("answer", "") or ""),
        }, body

    if query_type == "program_requirement":
        payload = {
            "student_profile": _ensure_plan_profile(profile),
        }
        response = client.post("/query/plan", json=payload)
        if response.status_code != 200:
            return {
                "decision": None,
                "citations": [],
                "abstained": True,
                "answer": "",
            }, {
                "error": f"HTTP {response.status_code}",
                "endpoint": "/query/plan",
            }
        body = response.json()
        return {
            "decision": None,
            "citations": body.get("citations", []),
            "abstained": ABSTENTION_MSG in str(body.get("answer", "")),
            "answer": str(body.get("answer", "") or ""),
        }, body

    payload = {
        "question": question,
        "filters": None,
    }
    response = client.post("/query/ask", json=payload)
    if response.status_code != 200:
        return {
            "decision": None,
            "citations": [],
            "abstained": True,
            "answer": "",
        }, {
            "error": f"HTTP {response.status_code}",
            "endpoint": "/query/ask",
        }
    body = response.json()
    return {
        "decision": None,
        "citations": body.get("citations", []),
        "abstained": bool(body.get("abstained", False)),
        "answer": str(body.get("answer", "") or ""),
    }, body


def _pick_id(eval_set: list[dict[str, Any]], prefix: str) -> str:
    for item in eval_set:
        item_id = str(item.get("id", ""))
        if item_id.startswith(prefix):
            return item_id
    return ""


def _build_transcripts(
    eval_set: list[dict[str, Any]],
    raw_responses: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    items_by_id = {str(item.get("id", "")): item for item in eval_set}

    eligibility_id = _pick_id(eval_set, "prereq_real_03") or _pick_id(eval_set, "prereq_real")
    plan_id = _pick_id(eval_set, "program_real_01") or _pick_id(eval_set, "program_real")
    abstention_id = _pick_id(eval_set, "trick_real_01") or _pick_id(eval_set, "trick_real")

    eligibility_item = items_by_id.get(eligibility_id, {})
    plan_item = items_by_id.get(plan_id, {})
    abstention_item = items_by_id.get(abstention_id, {})

    eligibility_response = raw_responses.get(eligibility_id, {})
    plan_response = raw_responses.get(plan_id, {})
    abstention_response = raw_responses.get(abstention_id, {})

    return {
        "eligibility_with_citations": {
            "id": eligibility_id,
            "question": eligibility_item.get("question", ""),
            "assistant": {
                "decision": eligibility_response.get("decision", ""),
                "answer": eligibility_response.get("answer", ""),
                "citations": eligibility_response.get("citations", [])[:3],
            },
        },
        "plan_with_citations": {
            "id": plan_id,
            "question": plan_item.get("question", ""),
            "assistant": {
                "plan": plan_response.get("plan", [])[:5],
                "answer": plan_response.get("answer", ""),
                "citations": plan_response.get("citations", [])[:3],
            },
        },
        "abstention_with_guidance": {
            "id": abstention_id,
            "question": abstention_item.get("question", ""),
            "assistant": {
                "abstained": bool(abstention_response.get("abstained", False)),
                "answer": abstention_response.get("answer", ABSTENTION_MSG),
                "guidance": "For instructor/schedule/facility details, check official MIT registrar and department sites.",
                "citations": abstention_response.get("citations", [])[:3],
            },
        },
    }


def main() -> None:
    args = _parse_args()
    base = Path(__file__).resolve().parent
    eval_set_path = str(base / "eval_set_realistic.json")
    eval_set = json.loads(Path(eval_set_path).read_text(encoding="utf-8"))

    actual_results: dict[str, dict[str, Any]] = {}
    raw_responses: dict[str, dict[str, Any]] = {}

    try:
        with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
            for item in eval_set:
                query_id = str(item.get("id", ""))
                actual, raw = _invoke_query(client, item)
                actual_results[query_id] = actual
                raw_responses[query_id] = raw
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Failed to reach evaluation API at {args.base_url}. Start the backend first, then retry."
        ) from exc

    result = run_eval(eval_set_path, actual_results=actual_results)
    transcripts = _build_transcripts(eval_set, raw_responses)

    report_payload = {
        "summary": {
            "total_queries": result["total_queries"],
            "citation_coverage_rate": result["citation_coverage_rate"],
            "eligibility_correctness_rate": result["eligibility_correctness_rate"],
            "abstention_accuracy_rate": result["abstention_accuracy_rate"],
        },
        "details": result.get("details", []),
        "raw_responses": raw_responses,
        "transcripts": transcripts,
    }

    report_path = base / "eval_report_realistic.json"
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print("Evaluation Summary (Realistic Set)")
    print(json.dumps(report_payload["summary"], indent=2))
    print(f"Report saved to: {report_path}")
    print(f"API base URL: {args.base_url}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realistic evaluation against a running PrereqPilot API.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8000"),
        help="Base URL of running backend API, e.g. http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("EVAL_HTTP_TIMEOUT", "60")),
        help="HTTP timeout in seconds for each request.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()