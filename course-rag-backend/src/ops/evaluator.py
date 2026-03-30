from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _safe_rate(numerator: int, denominator: int) -> float:
	if denominator <= 0:
		return 0.0
	return round((numerator / denominator) * 100.0, 2)


def run_eval(eval_set_path: str, actual_results: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
	path = Path(eval_set_path)
	if not path.exists():
		return {
			"total_queries": 0,
			"citation_coverage_rate": 0.0,
			"eligibility_correctness_rate": 0.0,
			"abstention_accuracy_rate": 0.0,
			"details": [],
		}

	actual_results = actual_results or {}
	eval_set = json.loads(path.read_text(encoding="utf-8"))

	total = len(eval_set)
	citation_hits = 0
	prereq_total = 0
	prereq_correct = 0
	abstention_total = 0
	abstention_correct = 0
	details: list[dict[str, Any]] = []

	for item in eval_set:
		query_id = str(item.get("id", ""))
		question = str(item.get("question", "") or "")
		expected_decision = item.get("expected_decision")
		expected_citations = bool(item.get("expected_citations_present", False))
		expected_abstained = item.get("expected_abstained")

		actual = actual_results.get(query_id, {})
		actual_citations = actual.get("citations", [])
		actual_decision = actual.get("decision")
		actual_abstained = actual.get("abstained")
		actual_answer = str(actual.get("answer", "") or "")

		has_citation = bool(actual_citations)
		if expected_citations and has_citation:
			citation_hits += 1

		if expected_decision is not None:
			prereq_total += 1
			if actual_decision == expected_decision:
				prereq_correct += 1

		if expected_abstained is not None:
			abstention_total += 1
			if bool(actual_abstained) == bool(expected_abstained):
				abstention_correct += 1

		details.append(
			{
				"id": query_id,
				"type": item.get("type"),
				"question": question,
				"actual_answer": actual_answer,
				"expected_decision": expected_decision,
				"actual_decision": actual_decision,
				"expected_abstained": expected_abstained,
				"actual_abstained": actual_abstained,
				"expected_citations_present": expected_citations,
				"actual_citations_present": has_citation,
			}
		)

	return {
		"total_queries": total,
		"citation_coverage_rate": _safe_rate(citation_hits, len([q for q in eval_set if q.get("expected_citations_present")])),
		"eligibility_correctness_rate": _safe_rate(prereq_correct, prereq_total),
		"abstention_accuracy_rate": _safe_rate(abstention_correct, abstention_total),
		"details": details,
	}
