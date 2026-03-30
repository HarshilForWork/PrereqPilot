import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.config import settings


def _sanitize_value(value):
	if value is None:
		return ""
	if isinstance(value, (str, int, float, bool)):
		return value
	if isinstance(value, list):
		return ",".join(str(item) for item in value)
	if isinstance(value, dict):
		return ",".join(f"{k}={v}" for k, v in value.items())
	return str(value)


def _sanitize_metadata(metadata: dict) -> dict:
	flat: dict = {}
	for key, value in metadata.items():
		if isinstance(value, dict):
			for nested_key, nested_value in value.items():
				flat[f"{key}_{nested_key}"] = _sanitize_value(nested_value)
		else:
			flat[key] = _sanitize_value(value)
	return flat


def get_chroma_client() -> chromadb.PersistentClient:
	return chromadb.PersistentClient(
		path=settings.CHROMA_PERSIST_DIR,
		settings=ChromaSettings(anonymized_telemetry=False),
	)


def get_collection() -> chromadb.Collection:
	client = get_chroma_client()
	return client.get_or_create_collection(
		name=settings.CHROMA_COLLECTION_NAME,
		metadata={"hnsw:space": "cosine"},
	)


async def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
	collection = get_collection()
	metadatas = [_sanitize_metadata(chunk.get("metadata", {})) for chunk in chunks]
	collection.upsert(
		ids=[chunk["chunk_id"] for chunk in chunks],
		documents=[chunk["content"] for chunk in chunks],
		embeddings=embeddings,
		metadatas=metadatas,
	)


async def query_collection(query_embedding: list[float], k: int, where: dict | None = None) -> dict:
	collection = get_collection()
	kwargs = {
		"query_embeddings": [query_embedding],
		"n_results": k,
		"include": ["documents", "metadatas", "distances"],
	}
	if where:
		kwargs["where"] = where
	return collection.query(**kwargs)
