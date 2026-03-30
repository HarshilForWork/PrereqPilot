from typing import Any

from src.core.config import settings
from src.retrieval.chroma_store import query_collection
from src.retrieval.embeddings import embed_query


def build_where_filter(filters: dict[str, Any] | None) -> dict[str, Any] | None:
	if not filters:
		return None

	normalized: dict[str, Any] = {}
	for key, value in filters.items():
		if value is None:
			continue
		if isinstance(value, list):
			normalized[key] = ",".join(str(item) for item in value)
		elif isinstance(value, dict):
			continue
		else:
			normalized[key] = value

	return normalized or None


def _to_chunks(payload: dict[str, Any]) -> list[dict[str, Any]]:
	docs = payload.get("documents", [[]])
	metas = payload.get("metadatas", [[]])
	dists = payload.get("distances", [[]])

	rows: list[dict[str, Any]] = []
	for index, document in enumerate(docs[0] if docs else []):
		metadata = (metas[0] if metas else [])[index] if metas and metas[0] else {}
		distance = (dists[0] if dists else [])[index] if dists and dists[0] else 1.0
		score = 1.0 - float(distance)
		if score < settings.SCORE_THRESHOLD:
			continue
		rows.append(
			{
				"content": document,
				"metadata": metadata,
				"score": score,
			}
		)
	return rows


async def retrieve_chunks(query: str, filters: dict[str, Any] | None = None, k: int | None = None) -> list[dict[str, Any]]:
	vector = await embed_query(query)
	payload = await query_collection(
		query_embedding=vector,
		k=k or settings.MAX_RETRIEVAL_K,
		where=build_where_filter(filters),
	)
	return _to_chunks(payload)
