import asyncio
import math
import re
import threading
import time
from collections import deque
from typing import List, Sequence

from pinecone import Pinecone

from src.core.config import settings
from src.core.exceptions import RetrievalError
from src.core.logger import get_logger

log = get_logger(__name__)

_client: Pinecone | None = None
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_RATE_WINDOW_SECONDS = 60.0
_TPM_LIMIT_RE = re.compile(r"max tokens per minute \((\d+)\)", re.IGNORECASE)

_rate_lock = threading.Lock()
_token_window: deque[tuple[float, int]] = deque()
_token_window_total = 0
_observed_tpm_limit: int | None = None


def _get_client() -> Pinecone:
	global _client
	if _client is None:
		_client = Pinecone(api_key=settings.PINECONE_API_KEY)
	return _client


def _task_type(value: str) -> str:
	mapping = {
		"retrieval_document": "passage",
		"retrieval_query": "query",
		"passage": "passage",
		"query": "query",
	}
	key = value.strip().lower()
	return mapping.get(key, "passage")


def _response_data(response: object) -> list[object]:
	if isinstance(response, dict):
		return list(response.get("data", []) or [])
	return list(getattr(response, "data", []) or [])


def _vector_values(item: object) -> list[float]:
	if isinstance(item, dict):
		values = item.get("values")
	else:
		values = getattr(item, "values", None)

	if values is None:
		raise RetrievalError("Pinecone embedding response missing vector values.")

	return [float(value) for value in values]


def _error_status_code(exc: Exception) -> int | None:
	for attr in ("status", "status_code", "code"):
		value = getattr(exc, attr, None)
		if isinstance(value, int):
			return value
		if isinstance(value, str) and value.isdigit():
			return int(value)

	message = str(exc)
	for code in _RETRYABLE_STATUS_CODES:
		if f" {code}" in message or f"({code})" in message:
			return code
	return None


def _configured_max_retries() -> int:
	return max(1, int(getattr(settings, "EMBEDDING_MAX_RETRIES", 4)))


def _chars_per_token() -> int:
	return max(1, int(getattr(settings, "EMBEDDING_ESTIMATED_CHARS_PER_TOKEN", 4)))


def _effective_tpm_limit() -> int:
	configured = max(1, int(getattr(settings, "EMBEDDING_MAX_TOKENS_PER_MINUTE", 200000)))
	if _observed_tpm_limit is None:
		return configured
	# Keep a safety margin when we observed a provider-side model limit.
	return max(1, min(configured, int(_observed_tpm_limit * 0.9)))


def _max_tokens_per_request() -> int:
	return max(1, int(getattr(settings, "EMBEDDING_MAX_TOKENS_PER_REQUEST", 45000)))


def _estimate_tokens(text: str) -> int:
	return max(1, math.ceil(len(text) / _chars_per_token()))


def _estimate_batch_tokens(inputs: Sequence[str]) -> int:
	return sum(_estimate_tokens(text) for text in inputs)


def _record_observed_tpm_limit(exc: Exception) -> None:
	global _observed_tpm_limit
	match = _TPM_LIMIT_RE.search(str(exc))
	if not match:
		return

	try:
		observed = int(match.group(1))
	except ValueError:
		return

	with _rate_lock:
		if _observed_tpm_limit is None or observed < _observed_tpm_limit:
			_observed_tpm_limit = observed
			log.info(f"Observed Pinecone TPM limit={observed}; applying adaptive throttle.")


def _reserve_tpm_budget(estimated_tokens: int) -> None:
	global _token_window_total

	limit = _effective_tpm_limit()
	estimated_tokens = min(limit, max(1, estimated_tokens))

	while True:
		wait_seconds = 0.0
		with _rate_lock:
			now = time.monotonic()
			while _token_window and (now - _token_window[0][0]) >= _RATE_WINDOW_SECONDS:
				_, expired_tokens = _token_window.popleft()
				_token_window_total -= expired_tokens

			if _token_window_total + estimated_tokens <= limit:
				_token_window.append((now, estimated_tokens))
				_token_window_total += estimated_tokens
				return

			oldest_ts, _ = _token_window[0]
			wait_seconds = max(0.05, (oldest_ts + _RATE_WINDOW_SECONDS) - now)

		log.info(
			f"Embedding throttle wait {wait_seconds:.2f}s | est_tokens={estimated_tokens} "
			f"window_tokens={_token_window_total} limit={limit}"
		)
		time.sleep(wait_seconds)


def _split_inputs_for_request_limits(inputs: Sequence[str], max_items: int) -> list[list[str]]:
	if not inputs:
		return []

	token_cap = _max_tokens_per_request()
	groups: list[list[str]] = []
	current: list[str] = []
	current_tokens = 0

	for text in inputs:
		text_tokens = _estimate_tokens(text)
		would_overflow = bool(current) and (
			len(current) >= max_items or (current_tokens + text_tokens) > token_cap
		)

		if would_overflow:
			groups.append(current)
			current = []
			current_tokens = 0

		current.append(text)
		current_tokens += text_tokens

	if current:
		groups.append(current)

	return groups


def _is_retryable(exc: Exception) -> bool:
	code = _error_status_code(exc)
	if code in _RETRYABLE_STATUS_CODES:
		return True

	message = str(exc).lower()
	return "rate limit" in message or "too many requests" in message or "temporarily unavailable" in message


def _retry_delay_seconds(attempt: int, exc: Exception) -> int:
	delay_seconds = min(2 ** (attempt - 1), 8)
	if _error_status_code(exc) == 429:
		# 429s usually indicate rolling per-minute quota exhaustion.
		delay_seconds = max(delay_seconds, min(10 * attempt, 40))
	return delay_seconds


def _embed_with_retry(inputs: Sequence[str], task_type: str) -> list[list[float]]:
	last_error: Exception | None = None
	input_type = _task_type(task_type)
	max_retries = _configured_max_retries()
	estimated_tokens = _estimate_batch_tokens(inputs)

	for attempt in range(1, max_retries + 1):
		try:
			_reserve_tpm_budget(estimated_tokens)
			response = _get_client().inference.embed(
				model=settings.EMBEDDING_MODEL,
				inputs=list(inputs),
				parameters={"input_type": input_type, "truncate": "END"},
			)
			vectors = [_vector_values(item) for item in _response_data(response)]

			if len(vectors) != len(inputs):
				raise RetrievalError(
					f"Embedding count mismatch for model {settings.EMBEDDING_MODEL}: expected {len(inputs)}, got {len(vectors)}"
				)
			return vectors
		except Exception as exc:
			last_error = exc
			if _error_status_code(exc) == 429:
				_record_observed_tpm_limit(exc)

			if attempt >= max_retries or not _is_retryable(exc):
				break

			delay_seconds = _retry_delay_seconds(attempt, exc)
			log.warning(
				f"Pinecone embedding retry {attempt}/{max_retries} in {delay_seconds}s: {exc}"
			)
			time.sleep(delay_seconds)

	raise RetrievalError(
		f"Pinecone embedding failed after {max_retries} attempt(s). Last error: {last_error}"
	)


def _embed_batch_sync(texts: List[str], task_type: str) -> List[List[float]]:
	max_items = max(1, int(settings.EMBEDDING_BATCH_SIZE))
	batches = _split_inputs_for_request_limits(texts, max_items=max_items)
	if len(batches) > 1:
		log.info(
			f"Embedding request split into {len(batches)} sub-batch(es) "
			f"for token/request cap={_max_tokens_per_request()}"
		)

	vectors: list[list[float]] = []
	for batch in batches:
		vectors.extend(_embed_with_retry(inputs=batch, task_type=task_type))

	log.info(
		f"Embedding batch used Pinecone model={settings.EMBEDDING_MODEL} input_type={_task_type(task_type)}"
	)
	return vectors


def _embed_single_sync(text: str, task_type: str) -> List[float]:
	vectors = _embed_with_retry(inputs=[text], task_type=task_type)
	log.info(
		f"Embedding query used Pinecone model={settings.EMBEDDING_MODEL} input_type={_task_type(task_type)}"
	)
	return vectors[0]


async def embed_documents(texts: List[str]) -> List[List[float]]:
	if not texts:
		return []

	batch_size = settings.EMBEDDING_BATCH_SIZE
	all_embeddings: List[List[float]] = []
	loop = asyncio.get_event_loop()

	log.info(f"Embedding {len(texts)} document(s) | batch_size={batch_size}")
	started = time.perf_counter()

	for idx in range(0, len(texts), batch_size):
		batch = texts[idx : idx + batch_size]
		vectors = await loop.run_in_executor(
			None,
			_embed_batch_sync,
			batch,
			settings.EMBEDDING_DOCUMENT_TASK_TYPE,
		)
		all_embeddings.extend(vectors)

	log.info(
		f"Embedding completed | vectors={len(all_embeddings)} elapsed={time.perf_counter() - started:.2f}s"
	)
	return all_embeddings


async def embed_query(query: str) -> List[float]:
	loop = asyncio.get_event_loop()
	return await loop.run_in_executor(
		None,
		_embed_single_sync,
		query,
		settings.EMBEDDING_QUERY_TASK_TYPE,
	)
