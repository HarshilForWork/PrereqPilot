from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.core.exceptions import AppError, GraphNodeNotFoundError, LLMServiceError, RetrievalError
from src.core.logger import get_logger
from src.inference.prompts import ABSTENTION_RESPONSE

log = get_logger(__name__)


def _error_payload(error: str, detail: str, code: int) -> dict:
	return {"error": error, "detail": detail, "code": code}


def register_error_handlers(app: FastAPI) -> None:
	@app.exception_handler(GraphNodeNotFoundError)
	async def graph_not_found_handler(_: Request, exc: GraphNodeNotFoundError):
		return JSONResponse(status_code=404, content=_error_payload(exc.__class__.__name__, exc.detail, 404))

	@app.exception_handler(RetrievalError)
	async def retrieval_handler(_: Request, exc: RetrievalError):
		return JSONResponse(status_code=503, content=_error_payload(exc.__class__.__name__, exc.detail, 503))

	@app.exception_handler(LLMServiceError)
	async def llm_handler(_: Request, exc: LLMServiceError):
		return JSONResponse(status_code=503, content=_error_payload(exc.__class__.__name__, exc.detail, 503))

	@app.exception_handler(AppError)
	async def app_error_handler(_: Request, exc: AppError):
		return JSONResponse(
			status_code=exc.status_code,
			content=_error_payload(exc.__class__.__name__, exc.detail, exc.status_code),
		)

	@app.exception_handler(Exception)
	async def unhandled_handler(_: Request, exc: Exception):
		detail = str(exc) or "Unhandled server error"
		if "empty retrieval" in detail.lower():
			return JSONResponse(
				status_code=200,
				content={
					"answer": ABSTENTION_RESPONSE,
					"citations": [],
					"abstained": True,
					"trace_id": "",
				},
			)
		log.exception("Unhandled exception")
		return JSONResponse(status_code=500, content=_error_payload(exc.__class__.__name__, detail, 500))
