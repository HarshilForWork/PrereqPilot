import pytest

from src.retrieval import embeddings


@pytest.mark.asyncio
async def test_embed_documents_preserves_order(monkeypatch) -> None:
	def fake_batch(texts: list[str], task_type: str) -> list[list[float]]:
		assert task_type == "retrieval_document"
		return [[float(index)] for index, _ in enumerate(texts)]

	monkeypatch.setattr(embeddings, "_embed_batch_sync", fake_batch)
	result = await embeddings.embed_documents(["one", "two", "three"])
	assert result == [[0.0], [1.0], [2.0]]


@pytest.mark.asyncio
async def test_embed_query_uses_query_task_type(monkeypatch) -> None:
	def fake_single(text: str, task_type: str) -> list[float]:
		assert text == "Can I take CS301?"
		assert task_type == "retrieval_query"
		return [0.1, 0.2, 0.3]

	monkeypatch.setattr(embeddings, "_embed_single_sync", fake_single)
	result = await embeddings.embed_query("Can I take CS301?")
	assert result == [0.1, 0.2, 0.3]
