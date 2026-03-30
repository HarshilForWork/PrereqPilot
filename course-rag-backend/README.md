# 🎓 Prerequisite Pilot — Agentic RAG Course Planning Backend

An agentic RAG-powered backend for an intelligent academic advisor. Combines **semantic vector
retrieval** (ChromaDB + Pinecone/E5 embeddings) with **structured graph reasoning** (NetworkX DAG)
and **multi-agent orchestration** (CrewAI) to deliver grounded, citation-backed course
eligibility checks, automated term planning, and catalog Q&A.

> **Core guarantee:** Every answer is either cited to a specific catalog chunk or explicitly
> abstained. No hallucination. No guessing.

---

## 📑 Table of Contents

1. [System Architecture](#-system-architecture)
2. [Request Lifecycle](#-request-lifecycle)
3. [Prerequisite DAG Structure](#-prerequisite-dag-structure)
4. [Project Structure](#-project-structure)
5. [Configuration](#-configuration)
6. [API Reference](#-api-reference)
7. [Getting Started](#-getting-started)
8. [Ingestion Pipeline](#-ingestion-pipeline)
9. [Agent Roles](#-agent-roles)
10. [Evaluation](#-evaluation)
11. [Observability](#-observability)
12. [Troubleshooting](#-troubleshooting)

---

## 🏗️ System Architecture

The hybrid knowledge engine is what separates this from a standard RAG system.
Vector search answers *"what does the catalog say?"* — the prerequisite DAG answers
*"can the student actually reach this course?"* — both are required for every eligibility decision.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PREREQUISITE PILOT                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Student / Client                                                              │
│         │                                                                       │
│         ▼                                                                       │
│   ┌─────────────┐                                                               │
│   │  FastAPI     │  ← Request logging middleware (trace_id injected here)       │
│   │  Gateway     │  ← Global error handler                                      │
│   └──────┬──────┘                                                               │
│          │                                                                      │
│          ▼                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐             │
│   │                  CrewAI Orchestration Layer                   │             │
│   │                                                              │             │
│   │   ┌──────────────┐                                           │             │
│   │   │ Intake Agent  │  Validates & normalises student profile   │             │
│   │   └──────┬───────┘  Generates clarifying questions if needed │             │
│   │          │                                                   │             │
│   │          ▼  (parallel via asyncio.gather)                    │             │
│   │   ┌──────────────────┐     ┌───────────────────────┐        │             │
│   │   │ Retriever Agent   │     │   Graph Reasoning     │        │             │
│   │   │ Semantic Search   │     │   DAG Traversal       │        │             │
│   │   │ ChromaDB + E5     │     │   NetworkX DiGraph    │        │             │
│   │   └──────┬───────────┘     └────────────┬──────────┘        │             │
│   │          │                              │                    │             │
│   │          └──────────────┬───────────────┘                    │             │
│   │                         ▼                                    │             │
│   │                ┌────────────────┐                            │             │
│   │                │ Planner Agent  │  Merges vector context      │             │
│   │                │                │  + graph eligibility        │             │
│   │                │                │  → draft course plan        │             │
│   │                └───────┬────────┘                            │             │
│   │                        │                                     │             │
│   │                        ▼                                     │             │
│   │                ┌────────────────┐                            │             │
│   │                │ Verifier Agent │  Citation audit             │             │
│   │                │ (Hallucination │  Prereq logic check         │             │
│   │                │   Shield)      │  Max 1 rewrite attempt      │             │
│   │                └───────┬────────┘                            │             │
│   └────────────────────────┼─────────────────────────────────────┘             │
│                            │                                                   │
│                            ▼                                                   │
│                    JSON Response                                                │
│             (answer + citations + graph_result)                                 │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  KNOWLEDGE LAYER                         │  INTELLIGENCE LAYER                 │
│                                          │                                     │
│  ┌──────────────────────────────────┐    │  ┌──────────────────────────────┐   │
│  │  ChromaDB (Persistent)           │    │  │  Groq API                    │   │
│  │  Collection: course_catalog      │    │  │  Model: llama-3.1-8b         │   │
│  │  Metric: cosine similarity       │    │  │  Client: AsyncGroq           │   │
│  │  ~800 char chunks, 150 overlap   │    │  │  All calls async/await       │   │
│  └──────────────────────────────────┘    │  └──────────────────────────────┘   │
│                                          │                                     │
│  ┌──────────────────────────────────┐    │  ┌──────────────────────────────┐   │
│  │  NetworkX DiGraph                │    │  │ Multilingual E5 Large (via Pinecone) │   │
│  │  Nodes: course codes             │    │  │  Batch size: 100 per call    │   │
│  │  Edges: prerequisite flows       │    │  │  task_type: retrieval_doc    │   │
│  │  Persisted: graph_store.json     │    │  │            retrieval_query   │   │
│  └──────────────────────────────────┘    │  └──────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  INGESTION PIPELINE (runs once, triggered via POST /ingest)                     │
│                                                                                 │
│  catalog_docs/*.pdf                                                             │
│         │                                                                       │
│         ▼                                                                       │
│  [parser.py] Hybrid PyMuPDF + pdfplumber → tables extracted as TABLE:\n blocks  │
│         │                                                                       │
│         ▼                                                                       │
│  [chunker.py] Paragraph → sentence → character fallback, 800 char target        │
│         │                                                                       │
│         ▼                                                                       │
│  [enricher.py] Adds: section_heading, course_codes_mentioned, has_prerequisites │
│                      document_type, catalog_year, has_grade_requirements, etc.  │
│         │                                                                       │
│         ├──────────────────────────────────────────────────────┐               │
│         ▼                                                       ▼               │
│  [embeddings.py]                                      [graph_builder.py]        │
│  Pinecone batch embed                                 Parse prereq rules        │
│  → ChromaDB upsert                                    → NetworkX DiGraph        │
│                                                       → graph_store.json        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Request Lifecycle

Every query follows this exact sequence. No shortcuts, no bypasses.

```
POST /query/prereq
  │
  ├─ 1. Middleware injects trace_id, starts latency timer
  │
  ├─ 2. INTAKE AGENT
  │      • LLM extracts course codes and student profile from natural language
  │      • Normalises course codes (e.g. "cs 301" → "CS301")
  │      • If required fields missing → returns clarifying_questions immediately
  │
  ├─ 3. PARALLEL EXECUTION  (asyncio.gather)
  │      │
  │      ├─ RETRIEVER AGENT ──────────────────────────────────────────────────────┐
  │      │    • embed_query(question) using retrieval_query task type              │
  │      │    • ChromaDB cosine search, top-k=8, score_threshold=0.35             │
  │      │    • Formats retrieved chunks into context_string + citations list      │
  │      │                                                                         │
  │      └─ GRAPH REASONING ───────────────────────────────────────────────────────┤
  │           • check_eligibility(course, completed_courses, grades)               │
  │           • Traverses NetworkX DAG: resolves AND / OR / corequisite rules      │
  │           • Returns: eligible bool, missing_prereqs, prereq_path list          │
  │                                                                                 │
  ├─ 4. PLANNER AGENT  ←───────────────────────── receives both results above ─────┘
  │      • Merges vector context + graph eligibility result
  │      • LLM drafts: Answer / Plan / Justifications / Citations / Risks
  │      • Uses temperature=0.2 for generation
  │
  ├─ 5. VERIFIER AGENT
  │      • LLM audits draft for uncited claims and prereq logic errors
  │      • Uses temperature=0.0 for deterministic checking
  │      • If issues found → triggers one rewrite attempt
  │      • If still failing → returns response with issues flagged
  │
  ├─ 6. OPS TRACKER
  │      • Writes trace JSON to request_logs/{trace_id}.json
  │      • Updates in-memory Metrics counters
  │
  └─ 7. Response returned to client
```

---

## 🌐 Prerequisite DAG Structure

Academic requirements are modelled as a **Directed Acyclic Graph**.

```
Node = Course Code            Edge = Prerequisite Flow

   CS101 ──────────────────────────────► CS301
     │          min_grade: C              │
     │                                    │
   MATH120 ────────────────────────────►  │
     │          requirement_type:         │
     │          "required"                │
     │                                    ▼
   CS101 ──── (either group 1) ────► CS401
     │
   MATH220 ── (either group 1) ────► CS401
     │
     └──── co-req: CS201 ────────────────►  CS301
```

| Component | Detail |
|---|---|
| **Node** | Course code e.g. `CS301`. Attributes: `course_name`, `credits`, `description_chunk_id` |
| **Edge** | Prerequisite flow A → B. Attributes: `requirement_type`, `min_grade`, `source_chunk_id` |
| **requirement_type** | `"required"` · `"either_or"` · `"corequisite"` |
| **Persistence** | Serialised to `data/graph_store.json` after every ingestion run |

---

## 📂 Project Structure

```
course-rag-backend/
│
├── app.py                          # FastAPI app factory, lifespan, router registration
├── config.yaml                     # ALL tunables — models, thresholds, paths, batch sizes
├── requirements.txt
├── .env                            # Secret keys only — never commit this
├── .env.example                    # Template for required env vars
├── Dockerfile
├── README.md
│
├── src/
│   │
│   ├── core/                       # Shared foundation — imported by every layer
│   │   ├── config.py               # Loads config.yaml + .env, exposes settings singleton
│   │   ├── constants.py            # ABSTENTION_MSG, COURSE_CODE_REGEX, keyword lists
│   │   ├── exceptions.py           # RetrievalError, GraphNodeNotFoundError, etc.
│   │   └── logger.py               # Structured JSON logger
│   │
│   ├── processing/                 # PROVIDED — do not modify these files
│   │   ├── parser.py               # Hybrid PyMuPDF + pdfplumber PDF parser
│   │   └── chunker.py              # Paragraph/sentence/character chunker with overlap
│   │
│   ├── ingestion/                  # Document pipeline
│   │   ├── pipeline.py             # Orchestrates parse → chunk → enrich → embed → store
│   │   ├── enricher.py             # Adds domain metadata (prereq flags, course codes, etc.)
│   │   └── graph_builder.py        # Parses prereq rules from chunks, builds NetworkX DAG
│   │
│   ├── retrieval/                  # Vector search layer
│   │   ├── embeddings.py           # Pinecone batch embedder (batching via settings)
│   │   ├── chroma_store.py         # ChromaDB client, upsert, cosine query
│   │   └── retriever.py            # LangChain retriever wrapper + metadata filter builder
│   │
│   ├── inference/                  # LLM layer
│   │   ├── groq_client.py          # AsyncGroq singleton + call_llm() helper
│   │   ├── prompts.py              # All ChatPromptTemplates (citation-enforcing)
│   │   └── chains.py               # LangChain chains wired to prompts + retriever
│   │
│   ├── agents/                     # CrewAI agent definitions
│   │   ├── crew_runtime.py         # CrewAI Crew setup, task wiring, kickoff
│   │   ├── intake.py               # IntakeAgent — profile extraction + validation
│   │   ├── retriever_agent.py      # CatalogRetrieverAgent — vector search + citations
│   │   ├── planner.py              # PlannerAgent — merges graph + vector → course plan
│   │   └── verifier.py             # VerifierAgent — citation audit, hallucination shield
│   │
│   ├── graph/                      # NetworkX prerequisite graph
│   │   ├── store.py                # Singleton DiGraph, load/save graph_store.json
│   │   └── reasoning.py            # Eligibility checks, path traversal, multi-hop logic
│   │
│   ├── api/                        # HTTP layer — zero business logic here
│   │   ├── middleware/
│   │   │   ├── request_logger.py   # Injects trace_id, writes to request_logs/
│   │   │   └── error_handler.py    # Global exception → structured JSON error
│   │   ├── models/
│   │   │   ├── requests.py         # Pydantic v2 request bodies
│   │   │   └── responses.py        # Pydantic v2 response models
│   │   └── routes/
│   │       ├── ingest.py           # POST /ingest, GET /ingest/status/{job_id}
│   │       ├── query.py            # POST /query/prereq, /query/plan, /query/ask
│   │       └── graph.py            # GET /graph/course/{code}, /graph/path, /graph/all
│   │
│   └── ops/                        # LLMOps
│       ├── metrics.py              # In-memory counters (latency, abstentions, rewrites)
│       ├── evaluator.py            # Runs eval_set.json, scores citation + eligibility
│       └── tracker.py              # Per-request JSON trace writer
│
├── data/
│   ├── catalog_docs/               # Drop your catalog PDFs here
│   ├── chroma_db/                  # ChromaDB persistent storage (auto-created)
│   └── graph_store.json            # Serialised NetworkX graph (auto-created)
│
├── logs/                           # System-level alerts and eval results
├── request_logs/                   # One JSON file per API request
│
└── tests/
    ├── unit/
    │   ├── test_enricher.py
    │   ├── test_graph_reasoning.py
    │   └── test_embeddings.py
    ├── integration/
    │   ├── test_ingestion_pipeline.py
    │   └── test_query_routes.py
    └── eval/
        ├── eval_set.json           # 25 test queries across 4 categories
        └── run_eval.py             # Evaluation runner
```

---

## ⚙️ Configuration

All tunables live in `config.yaml`. **Never hardcode values in application code.**

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  reload: true

paths:
  catalog_docs_dir: "./data/catalog_docs"
  chroma_persist_dir: "./data/chroma_db"
  graph_store_path: "./data/graph_store.json"
  logs_dir: "./logs"
  request_logs_dir: "./request_logs"

chroma:
  collection_name: "course_catalog"
  similarity_metric: "cosine"

embeddings:
  model: "multilingual-e5-large"
  batch_size: 50                    # Pinecone-managed batch size
  document_task_type: "retrieval_document"
  query_task_type: "retrieval_query"

llm:
  model: "llama-3.1-8b-instant"
  temperature_reasoning: 0.0        # verifier, eligibility checks
  temperature_generation: 0.2       # planner, course plans
  temperature_clarify: 0.1          # intake agent
  max_tokens: 4096

retrieval:
  top_k: 8
  score_threshold: 0.35             # chunks below this are discarded

chunking:
  chunk_size: 800
  chunk_overlap: 150

ops:
  track_latency: true
  log_every_request: true
```

Secret keys go in `.env` only — never in `config.yaml`:

```bash
# .env.example
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [Groq API Key](https://console.groq.com/) — free tier is sufficient for development
- [Pinecone API Key](https://www.pinecone.io/) — for E5-model embeddings via Inference API

### 1. Install

```bash
git clone <repo-url>
cd course-rag-backend

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Open .env and fill in GROQ_API_KEY and GOOGLE_API_KEY
```

### 3. Add Catalog Documents

Drop your university catalog PDFs into `data/catalog_docs/`:

```bash
data/
└── catalog_docs/
    ├── cs_undergraduate_catalog_2024.pdf
    ├── program_requirements_2024.pdf
    └── academic_policies_2024.pdf
```

Minimum required (per assessment spec):
- At least 20 course description pages
- At least 2 program/degree requirement pages
- At least 1 academic policy page

### 4. Run Ingestion

```bash
uvicorn app:app --reload

# In another terminal — trigger the ingestion pipeline
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"force_reingest": false}'

# Check ingestion status
curl http://localhost:8000/ingest/status/{job_id}
```

Ingestion does: PDF parsing → chunking → metadata enrichment → Pinecone batch embedding →
ChromaDB upsert → NetworkX graph build. Runs once. Re-run with `force_reingest: true` if
you update your PDFs.

### 5. Query the System

```bash
# Prerequisite eligibility check
curl -X POST http://localhost:8000/query/prereq \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can I take CS301 if I have completed CS101 with a B and MATH120?",
    "student_profile": {
      "completed_courses": ["CS101", "MATH120"],
      "grades": {"CS101": "B", "MATH120": "A"}
    }
  }'

# Generate a term plan
curl -X POST http://localhost:8000/query/plan \
  -H "Content-Type: application/json" \
  -d '{
    "student_profile": {
      "completed_courses": ["CS101", "MATH120", "CS201"],
      "grades": {"CS101": "A", "CS201": "B+"},
      "target_major": "Computer Science",
      "target_term": "Fall 2025",
      "max_credits": 15
    }
  }'

# General catalog Q&A
curl -X POST http://localhost:8000/query/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the minimum GPA required to graduate?", "filters": null}'

# Get prerequisite path for a course
curl "http://localhost:8000/graph/path?from=CS101&to=CS401&completed=CS101,MATH120"

# System health
curl http://localhost:8000/graph/health
```

---

## 🔄 Ingestion Pipeline

```
Step 1  Discover   Scan data/catalog_docs/ for all .pdf files

Step 2  Parse      parser.parse_document_hybrid(pdf_path)
                   → Hybrid PyMuPDF + pdfplumber
                   → Tables extracted as TABLE:\n... blocks

Step 3  Chunk      chunker.chunk_documents_optimized([parser_result])
                   → Paragraph-aware → sentence-aware → character fallback
                   → 800 char target, 150 char overlap
                   → Tables get dedicated chunks

Step 4  Enrich     enricher.enrich_chunk_metadata(chunk, parser_result)
                   → Adds: section_heading, document_type, course_codes_mentioned
                            has_prerequisites, has_grade_requirements, has_credit_info
                            has_corequisite, has_either_or, catalog_year, institution

Step 5  Embed      embeddings.embed_documents(all_texts)
                   → Multilingual E5 Large (via Pinecone)
                   → task_type: retrieval_document

Step 6  Store      chroma_store.upsert_chunks(chunks, embeddings)
                   → ChromaDB persistent collection, cosine space

Step 7  Build DAG  graph_builder.build_graph_from_chunks(chunks)
                   → Parses prerequisite language from enriched chunks
                   → NetworkX DiGraph, saved to data/graph_store.json
```

---

## 🤖 Agent Roles

| Agent | Responsibility | LLM Temperature |
|---|---|---|
| **IntakeAgent** | Extracts and normalises student profile from natural language. Generates 1–5 clarifying questions if required info is missing. | 0.1 |
| **CatalogRetrieverAgent** | Embeds query with `retrieval_query` task type, queries ChromaDB top-k=8, formats retrieved chunks into citations. | — |
| **PlannerAgent** | Merges graph eligibility result + vector context into a structured course plan with per-course justifications and citations. | 0.2 |
| **VerifierAgent** | Audits every factual claim for a `[SOURCE: ...]` citation. Checks prereq logic against graph data. Forces one rewrite if issues found. | 0.0 |

### Citation Format

Every factual claim in every LLM response must end with:

```
[SOURCE: {document_name}, Section: {section_heading}, Chunk: {chunk_id}]
```

If a claim cannot be cited → it must not be stated. If the answer is not in the catalog:

```
I don't have that information in the provided catalog/policies.
Please check: your academic advisor, the department's official website,
or the current schedule of classes.
```

---

## 📊 API Reference

### Ingestion

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Trigger ingestion pipeline as background task |
| `GET` | `/ingest/status/{job_id}` | Poll ingestion job status and progress |

### Query

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query/prereq` | Eligibility check with graph + vector reasoning |
| `POST` | `/query/plan` | Generate a term course plan |
| `POST` | `/query/ask` | General catalog Q&A |

### Graph

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/graph/course/{code}` | Course node info, prereqs, unlocked courses |
| `GET` | `/graph/path` | DAG path from source to target (frontend visualiser payload) |
| `GET` | `/graph/all` | Full graph as nodes + edges (frontend initial load) |
| `GET` | `/graph/health` | ChromaDB count, graph node/edge counts, uptime |

### Ops

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/ops/metrics` | Latency, abstention rate, verifier rewrite count |

### Response Shape — `/query/prereq`

```json
{
  "decision": "Eligible",
  "answer": "You may enrol in CS301. [SOURCE: cs_catalog.pdf, Section: CS301 Prerequisites, Chunk: cs_catalog_12_ab3f]",
  "evidence": "...",
  "citations": [
    { "chunk_id": "cs_catalog_12_ab3f", "document_name": "cs_catalog.pdf", "section_heading": "CS301 Prerequisites" }
  ],
  "next_step": "Register for CS301 in the upcoming term.",
  "clarifying_questions": [],
  "graph_result": {
    "eligible": true,
    "missing_prereqs": [],
    "prereq_path": ["CS101", "MATH120", "CS301"],
    "either_or_options": []
  },
  "assumptions": [],
  "trace_id": "3f4a1c2b-..."
}
```

---

## 🧪 Evaluation

The eval suite is in `tests/eval/` and covers 25 queries across 4 categories:

| Category | Count | What it tests |
|---|---|---|
| Prerequisite checks | 10 | Eligible / Not eligible decisions |
| Multi-hop chain questions | 5 | Needs 2+ courses of evidence |
| Program requirement questions | 5 | Credits, electives, required categories |
| Not-in-docs / trick questions | 5 | Availability, professor approval, schedule info |

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests (requires running server + ingested data)
pytest tests/integration/

# Run full eval set and get scored report
python tests/eval/run_eval.py
```

Eval report outputs:
- **Citation coverage rate** — % of responses with at least one `[SOURCE: ...]`
- **Eligibility correctness** — correct decision / total prereq checks (manual rubric)
- **Abstention accuracy** — % of not-in-docs queries correctly abstained

---

## 📡 Observability

### Request Traces

Every API call writes a trace file to `request_logs/{trace_id}.json`:

```json
{
  "trace_id": "3f4a1c2b-...",
  "timestamp": "2025-03-30T14:22:11Z",
  "endpoint": "/query/prereq",
  "agents_called": ["IntakeAgent", "CatalogRetrieverAgent", "PlannerAgent", "VerifierAgent"],
  "chunks_retrieved": 8,
  "graph_used": true,
  "verifier_passed": true,
  "verifier_issues": [],
  "citation_count": 3,
  "abstained": false,
  "total_latency_ms": 412
}
```

### Metrics Snapshot

```bash
curl http://localhost:8000/ops/metrics
```

```json
{
  "total_requests": 142,
  "abstentions": 18,
  "verifier_rewrites": 4,
  "citation_hits": 138,
  "avg_latency_ms": 387.4,
  "avg_chunks_retrieved": 7.2
}
```

---

## 🐛 Troubleshooting

**Ingestion fails with PyMuPDF error**
```bash
pip install --upgrade PyMuPDF pdfplumber
```

**ChromaDB collection is empty after ingestion**
Check the job status endpoint — ingestion runs as a background task. Wait for `"status": "complete"`.
Also verify PDFs are in `data/catalog_docs/` before triggering ingestion.

**Groq 429 rate limit errors**
The system retries once after 2 seconds automatically. If it persists, reduce the number of
concurrent requests or upgrade your Groq plan.

**Graph has 0 nodes after ingestion**
Your catalog PDFs may not contain explicit prerequisite language the parser recognises.
Check `logs/` for graph builder output. The system still works with vector-only retrieval —
graph reasoning degrades gracefully to "Need more info" decisions.

**Embeddings are slow**
Multilingual E5 Large batches according to batch_size in config.yaml. For large catalogs (1000+
chunks) the full embed takes ~30–60 seconds. This is a one-time cost at ingestion.

**`GOOGLE_API_KEY` not found**
Ensure `.env` exists in the project root (not inside `src/`). Copy from `.env.example` and
fill in both `GROQ_API_KEY` and `GOOGLE_API_KEY`.

---

## 📚 Data Sources

When curating your catalog corpus, include at minimum:

| Type | Minimum Count | Example |
|---|---|---|
| Course description pages | 20 | Individual course pages from university catalog |
| Program/degree requirement pages | 2 | CS major requirements, minor requirements |
| Academic policy pages | 1 | Grading policy, repeat policy, credit limits |

**Total: 30,000+ words or 25+ distinct documents.**

Include a `data/sources.md` listing each source URL and date accessed.

---

## 🏃 Quick Reference

```bash
# Full startup sequence
pip install -r requirements.txt
cp .env.example .env           # fill in API keys
uvicorn app:app --reload        # start server
curl -X POST localhost:8000/ingest -d '{}' -H "Content-Type: application/json"
# wait for ingestion to complete, then:
curl -X POST localhost:8000/query/prereq -H "Content-Type: application/json" \
  -d '{"question":"Can I take CS301?","student_profile":{"completed_courses":["CS101"]}}'
```

---

> Built for the Purple Merit Technologies — AI/ML Engineer Intern Assessment.
> Architecture: FastAPI + CrewAI + ChromaDB + E5 Embeddings + NetworkX + Groq.