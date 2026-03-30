import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.api.middleware.error_handler import register_error_handlers
from src.api.middleware.request_logger import RequestLoggerMiddleware
from src.api.routes import graph as graph_routes
from src.api.routes import ingest, query
from src.core.config import settings
from src.core.logger import get_logger
from src.graph.store import get_graph, load as load_graph
from src.retrieval.chroma_store import get_collection

log = get_logger("app")
_started_at = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
	assert settings.GROQ_API_KEY, "GROQ_API_KEY missing from .env"
	assert settings.PINECONE_API_KEY, "PINECONE_API_KEY missing from .env"

	collection = get_collection()
	chroma_count = collection.count()

	load_graph()
	graph = get_graph()

	log.info(
		"Backend ready | "
		f"Chroma: {chroma_count} chunks | "
		f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
	)
	app.state.started_at = _started_at
	yield
	log.info("Shutting down.")


app = FastAPI(title="Course Planning RAG API", lifespan=lifespan)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)
app.add_middleware(RequestLoggerMiddleware)
register_error_handlers(app)

app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(graph_routes.router, prefix="/graph", tags=["Graph"])


if __name__ == "__main__":
	uvicorn.run(
		"app:app",
		host=settings.SERVER_HOST,
		port=settings.SERVER_PORT,
		reload=settings.SERVER_RELOAD,
	)
