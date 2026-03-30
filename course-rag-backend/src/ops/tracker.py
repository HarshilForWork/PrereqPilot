import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.config import settings


async def log_trace(
	trace_id: str,
	endpoint: str,
	agents_called: list[str],
	chunks_retrieved: int,
	graph_used: bool,
	verifier_passed: bool,
	verifier_issues: list[str],
	citation_count: int,
	abstained: bool,
	total_latency_ms: float,
) -> None:
	payload: dict[str, Any] = {
		"trace_id": trace_id,
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"endpoint": endpoint,
		"agents_called": agents_called,
		"chunks_retrieved": chunks_retrieved,
		"graph_used": graph_used,
		"verifier_passed": verifier_passed,
		"verifier_issues": verifier_issues,
		"citation_count": citation_count,
		"abstained": abstained,
		"total_latency_ms": round(total_latency_ms, 3),
	}
	path = Path(settings.REQUEST_LOGS_DIR)
	path.mkdir(parents=True, exist_ok=True)
	(path / f"{trace_id}.json").write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
