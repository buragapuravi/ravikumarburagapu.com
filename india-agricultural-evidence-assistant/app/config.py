from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


_load_env_file(ROOT / ".env")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else ROOT / value


@dataclass(frozen=True)
class Settings:
    framework: str = os.getenv("RAG_FRAMEWORK", "llamaindex").lower()
    vector_backend: str = os.getenv("VECTOR_BACKEND", "chroma").lower()
    chroma_path: Path = _path("CHROMA_PATH", "data/index/chroma")
    collection: str = os.getenv("CHROMA_COLLECTION", "agri_evidence")
    raw_dir: Path = _path("RAW_DATA_DIR", "data/raw")
    demo_dir: Path = _path("DEMO_DATA_DIR", "data/demo")
    manifest_path: Path = _path(
        "MANIFEST_PATH", "data/index/ingestion-manifest.json"
    )
    ui_settings_path: Path = _path(
        "UI_SETTINGS_PATH", "data/index/ui-settings.json"
    )
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    reranker_model: str = os.getenv("RERANKER_MODEL", "")
    dense_top_k: int = int(os.getenv("DENSE_TOP_K", "12"))
    lexical_top_k: int = int(os.getenv("LEXICAL_TOP_K", "12"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "8"))
    evidence_min_score: float = float(os.getenv("EVIDENCE_MIN_SCORE", "0.08"))
    evidence_min_coverage: float = float(os.getenv("EVIDENCE_MIN_COVERAGE", "0.15"))
    max_official_source_age_days: int = int(os.getenv("MAX_OFFICIAL_SOURCE_AGE_DAYS", "365"))
    llm_provider: str = os.getenv("LLM_PROVIDER", "none").lower()
    llm_model: str = os.getenv("LLM_MODEL", "llama3.1:8b")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
    postgres_dsn: str = os.getenv(
        "POSTGRES_DSN", "postgresql://localhost:5432/agri_evidence"
    )


def settings() -> Settings:
    config = Settings()
    if config.framework not in {"llamaindex", "langchain"}:
        raise ValueError("RAG_FRAMEWORK must be 'llamaindex' or 'langchain'.")
    if config.vector_backend not in {"chroma", "pgvector"}:
        raise ValueError("VECTOR_BACKEND must be 'chroma' or 'pgvector'.")
    if config.llm_provider not in {"none", "ollama", "openai-compatible"}:
        raise ValueError("LLM_PROVIDER must be none, ollama, or openai-compatible.")
    if config.vector_backend == "pgvector" and config.framework != "llamaindex":
        raise ValueError("The pgvector backend currently supports the LlamaIndex adapter only.")
    if min(config.dense_top_k, config.lexical_top_k, config.rerank_top_k) <= 0:
        raise ValueError("Retrieval top-k settings must be positive.")
    if not 0 <= config.evidence_min_score <= 1 or not 0 <= config.evidence_min_coverage <= 1:
        raise ValueError("Evidence score and coverage thresholds must be between 0 and 1.")
    if config.llm_timeout <= 0 or not 0 <= config.llm_temperature <= 2:
        raise ValueError("LLM timeout must be positive and temperature must be between 0 and 2.")
    if config.max_official_source_age_days < 0:
        raise ValueError("MAX_OFFICIAL_SOURCE_AGE_DAYS must be zero or greater.")
    return config
