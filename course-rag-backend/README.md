# Course RAG Backend (Skeleton)

This repository is the Phase A runnable skeleton for an Agentic RAG Course Planning Assistant.

## Quick start

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies from requirements.txt.
3. Set GROQ_API_KEY and PINECONE_API_KEY in .env.
4. Run: uvicorn app:app --reload --host 0.0.0.0 --port 8000

## Current status

- Folder layout matches the required bounded contexts.
- FastAPI app starts with async route stubs and structured middleware.
- Agent, retrieval, graph, ingestion, and ops modules are scaffolded.
- src/processing/chunker.py is copied from the provided immutable file.
- src/processing/parser.py is a temporary placeholder and must be replaced with the provided immutable parser implementation before production ingestion.

## Phase B

Full ingestion, retrieval, graph reasoning, planner generation, verifier auditing, and evaluation workflows will be implemented in the next phase.
