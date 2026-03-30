from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.api.models.requests import IngestRequest
from src.api.models.responses import IngestStartResponse, IngestStatusResponse
from src.ingestion.pipeline import get_job_status, run_ingestion_job

router = APIRouter()


@router.post(
	"",
	response_model=IngestStartResponse,
	summary="Start Ingestion Job",
	description="Starts a background ingestion run that parses catalog documents, rebuilds retrieval storage, and updates the prerequisite graph.",
)
async def start_ingestion(payload: IngestRequest, background_tasks: BackgroundTasks) -> IngestStartResponse:
	job_id = str(uuid4())
	background_tasks.add_task(run_ingestion_job, job_id, payload.force_reingest)
	return IngestStartResponse(job_id=job_id, message="Ingestion started")


@router.get(
	"/status/{job_id}",
	response_model=IngestStatusResponse,
	summary="Get Ingestion Job Status",
	description="Returns progress and lifecycle timestamps for a previously started ingestion job.",
)
async def ingestion_status(job_id: str) -> IngestStatusResponse:
	status = get_job_status(job_id)
	if not status:
		raise HTTPException(status_code=404, detail=f"Ingestion job {job_id} not found")

	return IngestStatusResponse(
		job_id=job_id,
		status=str(status.get("status", "unknown")),
		progress=float(status.get("progress", 0.0) or 0.0),
		detail=str(status.get("detail", "")),
		started_at=float(status.get("started_at", 0.0) or 0.0),
		completed_at=status.get("completed_at"),
	)
