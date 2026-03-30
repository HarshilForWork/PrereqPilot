import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.core.config import settings


class RequestLoggerMiddleware(BaseHTTPMiddleware):
	async def dispatch(self, request: Request, call_next):
		started = time.perf_counter()
		trace_id = str(uuid.uuid4())
		request.state.trace_id = trace_id

		response: Response = await call_next(request)

		elapsed_ms = (time.perf_counter() - started) * 1000
		payload = {
			"trace_id": trace_id,
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"method": request.method,
			"path": request.url.path,
			"status_code": response.status_code,
			"latency_ms": round(elapsed_ms, 3),
		}

		directory = Path(settings.REQUEST_LOGS_DIR)
		directory.mkdir(parents=True, exist_ok=True)
		path = directory / f"{trace_id}.json"
		existing = {}
		if path.exists():
			try:
				existing = json.loads(path.read_text(encoding="utf-8"))
			except Exception:
				existing = {}

		merged = {**existing, **payload}
		path.write_text(json.dumps(merged, ensure_ascii=True), encoding="utf-8")
		response.headers["X-Trace-Id"] = trace_id
		return response
