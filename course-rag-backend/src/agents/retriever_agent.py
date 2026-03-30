from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.logger import get_logger
from src.retrieval.retriever import retrieve_chunks

log = get_logger(__name__)

_MAX_CONTEXT_CHUNKS = 4
_MAX_CONTEXT_CHARS_PER_CHUNK = 420
_MAX_CONTEXT_TOTAL_CHARS = 3600


def _compact_text(value: Any, max_chars: int) -> str:
	text = " ".join(str(value or "").split())
	if max_chars <= 3:
		return text[:max_chars]
	if len(text) <= max_chars:
		return text
	return f"{text[: max_chars - 3].rstrip()}..."


class CatalogRetrieverAgent:
	def __init__(self) -> None:
		self._retrieve_chain = RunnableLambda(self._retrieve_chunks)
		self._format_chain = RunnableParallel(
			retrieved_chunks=RunnableLambda(lambda payload: payload.get("chunks", [])),
			citations=RunnableLambda(self._build_citations),
			context_string=RunnableLambda(self._build_context_string),
		)

	async def _retrieve_chunks(self, chain_input: dict[str, Any]) -> dict[str, Any]:
		query = str(chain_input.get("query", "") or "")
		filters = chain_input.get("filters")

		# Skeleton-safe fallback: avoid external retrieval when placeholder keys are used.
		if settings.PINECONE_API_KEY.startswith("dev-placeholder"):
			chunks: list[dict[str, Any]] = []
		else:
			try:
				chunks = await retrieve_chunks(query=query, filters=filters)
			except Exception:
				log.exception("Retriever failed, using empty context fallback")
				chunks = []

		return {"chunks": chunks}

	def _build_citations(self, payload: dict[str, Any]) -> list[dict[str, str]]:
		chunks = payload.get("chunks", [])
		citations: list[dict[str, str]] = []
		for chunk in chunks:
			metadata = chunk.get("metadata") or {}
			citations.append(
				{
					"chunk_id": str(metadata.get("chunk_id", "")),
					"document_name": str(metadata.get("document_name", "")),
					"section_heading": str(metadata.get("section_heading", "")),
				}
			)
		return citations

	def _build_context_string(self, payload: dict[str, Any]) -> str:
		chunks = payload.get("chunks", [])
		context_lines: list[str] = []
		total_chars = 0
		limited_chunks = chunks[:_MAX_CONTEXT_CHUNKS]
		for chunk in limited_chunks:
			metadata = chunk.get("metadata") or {}
			citation = (
				"[SOURCE: "
				f"{metadata.get('document_name', '')}, "
				f"Section: {metadata.get('section_heading', '')}, "
				f"Chunk: {metadata.get('chunk_id', '')}]"
			)
			content = _compact_text(chunk.get("content", ""), _MAX_CONTEXT_CHARS_PER_CHUNK)
			line = f"{content}\n{citation}"

			remaining = _MAX_CONTEXT_TOTAL_CHARS - total_chars
			if remaining <= 0:
				break
			if len(line) > remaining:
				content_budget = max(80, remaining - len(citation) - 1)
				line = f"{_compact_text(content, content_budget)}\n{citation}"

			context_lines.append(line)
			total_chars += len(line) + 2

		omitted = len(chunks) - len(limited_chunks)
		if omitted > 0:
			context_lines.append(f"[Token budget note: omitted {omitted} additional chunks.]")
		return "\n\n".join(context_lines)

	async def run(self, query: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
		retrieval_payload = await self._retrieve_chain.ainvoke({"query": query, "filters": filters})
		formatted = await self._format_chain.ainvoke(retrieval_payload)
		return RetrieverChainOutput.model_validate(formatted).model_dump()


class RetrieverChainOutput(BaseModel):
	retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
	citations: list[dict[str, str]] = Field(default_factory=list)
	context_string: str = ""
