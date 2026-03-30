from pathlib import Path

import pytest

from src.ingestion import pipeline


@pytest.mark.asyncio
async def test_run_ingestion_job_completes(monkeypatch, tmp_path: Path) -> None:
	docs_dir = tmp_path / "catalog_docs"
	docs_dir.mkdir(parents=True, exist_ok=True)
	(docs_dir / "sample.pdf").write_bytes(b"%PDF-test")

	monkeypatch.setattr(pipeline.settings, "CATALOG_DOCS_DIR", str(docs_dir))

	def fake_parse(pdf_path: str, save_parsed_text: bool = False) -> dict:
		_ = save_parsed_text
		return {
			"document_name": Path(pdf_path).name,
			"content": "CS201 requires CS101",
			"total_pages": 1,
			"parsing_method": "fake",
			"processing_time": 0.01,
			"metadata": {},
		}

	def fake_chunker(parsed_content: list[dict], chunk_size: int = 800, chunk_overlap: int = 150) -> list[dict]:
		_ = chunk_size
		_ = chunk_overlap
		_ = parsed_content
		return [
			{
				"chunk_id": "chunk_1",
				"document_name": "sample.pdf",
				"content": "CS201 prerequisite CS101",
				"chunk_index": 0,
				"char_count": 24,
				"metadata": {
					"chunking_method": "optimized_text",
					"content_type": "text",
					"source_document": "sample.pdf",
					"chunk_size_config": 800,
					"overlap_config": 150,
				},
			}
		]

	async def fake_embed(texts: list[str]) -> list[list[float]]:
		return [[0.1, 0.2] for _ in texts]

	upsert_calls: list[tuple[list[dict], list[list[float]]]] = []

	async def fake_upsert(chunks: list[dict], embeddings: list[list[float]]) -> None:
		upsert_calls.append((chunks, embeddings))

	monkeypatch.setattr(pipeline, "parse_document_hybrid", fake_parse)
	monkeypatch.setattr(pipeline, "chunk_documents_optimized", fake_chunker)
	monkeypatch.setattr(pipeline, "embed_documents", fake_embed)
	monkeypatch.setattr(pipeline, "upsert_chunks", fake_upsert)
	monkeypatch.setattr(pipeline, "build_graph_from_chunks", lambda chunks: None)
	monkeypatch.setattr(pipeline.graph_store, "save", lambda: None)

	job_id = "job_test_001"
	await pipeline.run_ingestion_job(job_id)
	status = pipeline.get_job_status(job_id)

	assert status is not None
	assert status["status"] == "complete"
	assert len(upsert_calls) == 1
	assert len(upsert_calls[0][0]) == 1
	assert len(upsert_calls[0][1]) == 1
