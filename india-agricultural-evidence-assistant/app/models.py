from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceMetadata:
    source_id: str
    filename: str
    title: str
    organization: str = "Unknown"
    source_url: str = ""
    issued_date: str = ""
    state: str = ""
    district: str = ""
    crop: str = ""
    season: str = ""
    language: str = "English"
    document_type: str = "research note"
    status: str = "synthetic teaching document"
    extraction_warnings: tuple[str, ...] = ()
    source_fetch_date: str = ""
    validity_start: str = ""
    validity_end: str = ""
    license_access_note: str = ""
    file_sha256: str = ""
    extraction_method: str = ""


@dataclass(frozen=True)
class PageText:
    page_number: int | None
    text: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: SourceMetadata
    page_number: int | None
    chunk_index: int


@dataclass
class RetrievedChunk:
    chunk: Chunk
    dense_score: float = 0.0
    lexical_score: float = 0.0
    fusion_score: float = 0.0
    reranker_score: float | None = None


@dataclass(frozen=True)
class EvidenceDecision:
    answerable: bool
    reason: str
    coverage: float
    score: float
    missing_context: tuple[str, ...] = ()


@dataclass
class AnswerResult:
    answer: str
    abstained: bool
    abstention_reason: str
    citations: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    evidence: list[RetrievedChunk] = field(default_factory=list)
    gate: EvidenceDecision | None = None
    provider_state: str = "not configured"
    diagnostics: dict[str, Any] = field(default_factory=dict)
