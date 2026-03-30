import time
from pathlib import Path
from typing import Any

from src.core.config import settings
from src.core.exceptions import IngestionError
from src.core.logger import get_logger
from src.graph import store as graph_store
from src.ingestion.enricher import enrich_chunk_metadata
from src.ingestion.graph_builder import build_graph_from_chunks
from src.processing.chunker import chunk_documents_optimized
from src.processing.parser import parse_document_hybrid
from src.retrieval.chroma_store import upsert_chunks
from src.retrieval.embeddings import embed_documents

log = get_logger(__name__)

ingestion_jobs: dict[str, dict[str, Any]] = {}


def _set_job(job_id: str, **updates: Any) -> None:
	if job_id not in ingestion_jobs:
		ingestion_jobs[job_id] = {
			"status": "running",
			"progress": 0.0,
			"detail": "queued",
			"started_at": time.time(),
			"completed_at": None,
		}
	ingestion_jobs[job_id].update(updates)


def get_job_status(job_id: str) -> dict[str, Any] | None:
	return ingestion_jobs.get(job_id)


async def run_ingestion_job(job_id: str, force_reingest: bool = False) -> None:
	started_at = time.time()
	_set_job(job_id, status="running", progress=0.0, detail="Discovering PDF files", started_at=started_at)

	try:
		docs_dir = Path(settings.CATALOG_DOCS_DIR)
		docs_dir.mkdir(parents=True, exist_ok=True)
		pdf_files = sorted(path for path in docs_dir.glob("*.pdf") if path.is_file())

		if not pdf_files:
			_set_job(
				job_id,
				status="complete",
				progress=1.0,
				detail="No PDFs found in catalog_docs directory",
				completed_at=time.time(),
			)
			return

		_set_job(job_id, progress=0.1, detail=f"Found {len(pdf_files)} PDF(s)")
		_ = force_reingest

		all_enriched_chunks: list[dict[str, Any]] = []
		table_chunk_count = 0

		for index, pdf_path in enumerate(pdf_files, start=1):
			_set_job(job_id, detail=f"Parsing {pdf_path.name}", progress=min(0.15 + 0.2 * (index / len(pdf_files)), 0.35))
			parsed = parse_document_hybrid(str(pdf_path), save_parsed_text=False)

			_set_job(job_id, detail=f"Chunking {pdf_path.name}")
			raw_chunks = chunk_documents_optimized(
				[parsed],
				chunk_size=settings.CHUNK_SIZE,
				chunk_overlap=settings.CHUNK_OVERLAP,
			)

			_set_job(job_id, detail=f"Enriching chunks for {pdf_path.name}")
			for chunk in raw_chunks:
				enriched = enrich_chunk_metadata(chunk, parsed)
				all_enriched_chunks.append(enriched)
				if str(enriched.get("metadata", {}).get("content_type", "")) == "table":
					table_chunk_count += 1

		_set_job(job_id, progress=0.55, detail="Embedding chunks")
		embeddings = await embed_documents([chunk["content"] for chunk in all_enriched_chunks])

		if len(embeddings) != len(all_enriched_chunks):
			raise IngestionError("Embedding count mismatch during ingestion.")

		_set_job(job_id, progress=0.75, detail="Storing vectors in ChromaDB")
		await upsert_chunks(all_enriched_chunks, embeddings)

		if force_reingest or graph_store.get_graph().number_of_nodes() > 0:
			_set_job(job_id, progress=0.85, detail="Resetting in-memory graph before rebuild")
			graph_store.reset_graph()

		_set_job(job_id, progress=0.9, detail="Building prerequisite graph")
		build_graph_from_chunks(all_enriched_chunks)
		graph_store.save()

		elapsed = time.time() - started_at
		detail = (
			f"Ingestion complete | docs={len(pdf_files)} chunks={len(all_enriched_chunks)} "
			f"table_chunks={table_chunk_count} time={elapsed:.2f}s"
		)
		_set_job(job_id, status="complete", progress=1.0, detail=detail, completed_at=time.time())
		log.info(detail)
	except Exception as exc:
		log.exception("Ingestion failed")
		_set_job(
			job_id,
			status="failed",
			progress=1.0,
			detail=f"Ingestion failed: {exc}",
			completed_at=time.time(),
		)
