from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.chunking import chunk_pages
from app.config import Settings, settings
from app.documents import extract_document
from app.storage import ChromaStore, VectorStore

logger = logging.getLogger(__name__)
SUPPORTED = {".pdf", ".md", ".markdown", ".txt"}


@lru_cache(maxsize=2)
def create_store(config: Settings) -> VectorStore:
    if config.vector_backend == "chroma":
        config.chroma_path.mkdir(parents=True, exist_ok=True)
        return ChromaStore(
            str(config.chroma_path),
            config.collection,
            config.embedding_model,
            config.embedding_device,
        )
    from app.pgvector_store import PgVectorStore
    return PgVectorStore(config.postgres_dsn, config.embedding_model)


def ingest_paths(
    inputs: list[Path],
    store: VectorStore,
    manifest_path: Path,
    rebuild: bool = False,
    ocr: bool = False,
    framework: str = "llamaindex",
) -> dict[str, Any]:
    missing_inputs = [path for path in inputs if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            f"Input paths do not exist: {', '.join(str(path) for path in missing_inputs)}"
        )
    paths = sorted(
        {
            path
            for root in inputs
            for path in (root.rglob("*") if root.is_dir() else [root])
            if path.is_file() and path.name.casefold() != "readme.md" and path.suffix.lower() in SUPPORTED
        }
    )
    if not paths:
        raise FileNotFoundError(
            f"No supported source files found in: {', '.join(str(path) for path in inputs)}"
        )

    documents: list[dict[str, Any]] = []
    chunks = []
    failed: list[dict[str, str]] = []
    seen_source_ids: set[str] = set()
    pages_seen = 0
    for path in paths:
        try:
            metadata, pages = extract_document(path, ocr=ocr)
            if metadata.source_id in seen_source_ids:
                raise ValueError(f"Duplicate source_id {metadata.source_id!r}; source IDs must be unique.")
            seen_source_ids.add(metadata.source_id)
            pages_seen += len(pages)
            document_chunks = chunk_pages(metadata, pages)
            chunks.extend(document_chunks)
            documents.append(
                {
                    "filename": path.name,
                    "source_id": metadata.source_id,
                    "title": metadata.title,
                    "organization": metadata.organization,
                    "source_url": metadata.source_url,
                    "issued_date": metadata.issued_date,
                    "source_fetch_date": metadata.source_fetch_date,
                    "validity_start": metadata.validity_start,
                    "validity_end": metadata.validity_end,
                    "license_access_note": metadata.license_access_note,
                    "file_sha256": metadata.file_sha256,
                    "extraction_method": metadata.extraction_method,
                    "pages": len(pages),
                    "chunks": len(document_chunks),
                    "warnings": [
                        {"page": page.page_number, "warnings": list(page.warnings)}
                        for page in pages
                        if page.warnings
                    ],
                    "status": metadata.status,
                }
            )
        except (OSError, ValueError, RuntimeError) as error:
            logger.exception("Could not ingest %s", path)
            failed.append({"filename": path.name, "error": str(error)})
    if rebuild and chunks:
        store.clear()
    if chunks:
        if isinstance(store, ChromaStore):
            if framework == "llamaindex":
                from app.frameworks.llamaindex_adapter import ingest
            elif framework == "langchain":
                from app.frameworks.langchain_adapter import ingest
            else:
                raise ValueError(f"Unsupported ingestion framework: {framework}")
            ingest(store, chunks)
        else:
            store.upsert(chunks)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
        "document_count": len(documents),
        "page_count": pages_seen,
        "chunk_count": len(chunks),
        "failed_documents": failed,
        "rebuild_skipped": bool(rebuild and not chunks),
        "failed_page_count": sum(
            1
            for document in documents
            for warning in document["warnings"]
            if any(
                "no extractable text" in item.lower() or "page remains empty" in item.lower()
                for item in warning["warnings"]
            )
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local agricultural evidence documents.")
    parser.add_argument("--input", action="append", type=Path, help="Input file or directory; repeat to add paths.")
    parser.add_argument("--rebuild", action="store_true", help="Clear the configured index before ingestion.")
    parser.add_argument("--ocr", action="store_true", help="Explicitly OCR empty PDF pages using local Tesseract.")
    args = parser.parse_args()
    config = settings()
    inputs = args.input or [config.raw_dir]
    report = ingest_paths(
        inputs,
        create_store(config),
        config.manifest_path,
        args.rebuild,
        args.ocr,
        config.framework,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
