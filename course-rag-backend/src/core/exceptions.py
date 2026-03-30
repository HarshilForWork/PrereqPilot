class AppError(Exception):
	status_code: int = 500

	def __init__(self, detail: str, status_code: int | None = None) -> None:
		super().__init__(detail)
		self.detail = detail
		if status_code is not None:
			self.status_code = status_code


class RetrievalError(AppError):
	status_code = 503


class IngestionError(AppError):
	status_code = 503


class GraphError(AppError):
	status_code = 500


class GraphNodeNotFoundError(GraphError):
	status_code = 404


class LLMServiceError(AppError):
	status_code = 503


class ValidationError(AppError):
	status_code = 400
