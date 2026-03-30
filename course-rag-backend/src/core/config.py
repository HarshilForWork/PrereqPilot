import os
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml() -> dict:
	path = Path(__file__).parents[2] / "config.yaml"
	with path.open("r", encoding="utf-8") as file:
		return yaml.safe_load(file)


_project_root = Path(__file__).resolve().parents[2]
_env_file = _project_root / ".env"
_yaml = _load_yaml()
_embeddings_cfg = _yaml["embeddings"]
_crewai_cfg = _yaml.get("crewai", {})


class Settings(BaseSettings):
	GROQ_API_KEY: str
	PINECONE_API_KEY: str
	GOOGLE_API_KEY: str = ""
	SSL_CERT_FILE: str = ""
	REQUESTS_CA_BUNDLE: str = ""
	CURL_CA_BUNDLE: str = ""

	CHROMA_PERSIST_DIR: str = _yaml["paths"]["chroma_persist_dir"]
	CHROMA_COLLECTION_NAME: str = _yaml["chroma"]["collection_name"]
	EMBEDDING_MODEL: str = _embeddings_cfg["model"]
	EMBEDDING_BATCH_SIZE: int = _embeddings_cfg["batch_size"]
	EMBEDDING_DOCUMENT_TASK_TYPE: str = _embeddings_cfg["document_task_type"]
	EMBEDDING_QUERY_TASK_TYPE: str = _embeddings_cfg["query_task_type"]
	EMBEDDING_MAX_TOKENS_PER_MINUTE: int = _embeddings_cfg.get("max_tokens_per_minute", 200000)
	EMBEDDING_MAX_TOKENS_PER_REQUEST: int = _embeddings_cfg.get("max_tokens_per_request", 45000)
	EMBEDDING_ESTIMATED_CHARS_PER_TOKEN: int = _embeddings_cfg.get("estimated_chars_per_token", 4)
	EMBEDDING_MAX_RETRIES: int = _embeddings_cfg.get("max_retries", 4)
	LLM_MODEL: str = _yaml["llm"]["model"]
	LLM_MAX_TOKENS: int = _yaml["llm"]["max_tokens"]
	TEMP_REASONING: float = _yaml["llm"]["temperature_reasoning"]
	TEMP_GENERATION: float = _yaml["llm"]["temperature_generation"]
	TEMP_CLARIFY: float = _yaml["llm"]["temperature_clarify"]
	MAX_RETRIEVAL_K: int = _yaml["retrieval"]["top_k"]
	SCORE_THRESHOLD: float = _yaml["retrieval"]["score_threshold"]
	CATALOG_DOCS_DIR: str = _yaml["paths"]["catalog_docs_dir"]
	GRAPH_STORE_PATH: str = _yaml["paths"]["graph_store_path"]
	CHUNK_SIZE: int = _yaml["chunking"]["chunk_size"]
	CHUNK_OVERLAP: int = _yaml["chunking"]["chunk_overlap"]
	REQUEST_LOGS_DIR: str = _yaml["paths"]["request_logs_dir"]
	LOGS_DIR: str = _yaml["paths"]["logs_dir"]
	SERVER_HOST: str = _yaml["server"]["host"]
	SERVER_PORT: int = _yaml["server"]["port"]
	SERVER_RELOAD: bool = _yaml["server"]["reload"]
	OPS_TRACK_LATENCY: bool = _yaml["ops"]["track_latency"]
	OPS_LOG_EVERY_REQUEST: bool = _yaml["ops"]["log_every_request"]
	CREWAI_ENABLED: bool = _crewai_cfg.get("enabled", False)
	CREWAI_VERBOSE: bool = _crewai_cfg.get("verbose", False)
	CREWAI_MEMORY: bool = _crewai_cfg.get("memory", False)
	CREWAI_TIMEOUT_SECONDS: int = _crewai_cfg.get("timeout_seconds", 45)

	model_config = SettingsConfigDict(env_file=_env_file, env_file_encoding="utf-8", extra="ignore")


settings = Settings()


def _apply_ssl_env_from_settings() -> None:
	ssl_cert_file = str(settings.SSL_CERT_FILE or "").strip()
	requests_ca_bundle = str(settings.REQUESTS_CA_BUNDLE or "").strip()
	curl_ca_bundle = str(settings.CURL_CA_BUNDLE or "").strip()

	if ssl_cert_file:
		os.environ["SSL_CERT_FILE"] = ssl_cert_file
		# Mirror SSL_CERT_FILE for stacks that consult requests/curl style variables.
		if not requests_ca_bundle:
			os.environ["REQUESTS_CA_BUNDLE"] = ssl_cert_file
		if not curl_ca_bundle:
			os.environ["CURL_CA_BUNDLE"] = ssl_cert_file

	if requests_ca_bundle:
		os.environ["REQUESTS_CA_BUNDLE"] = requests_ca_bundle
	if curl_ca_bundle:
		os.environ["CURL_CA_BUNDLE"] = curl_ca_bundle


_apply_ssl_env_from_settings()
