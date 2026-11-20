from __future__ import annotations

import hashlib
import re

from app.models import Chunk, PageText, SourceMetadata


def _paragraphs(text: str) -> list[str]:
    # Keep headings and Markdown table rows in their original context.
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def chunk_pages(
    source: SourceMetadata,
    pages: list[PageText],
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    chunks: list[Chunk] = []
    for page in pages:
        if not page.text:
            continue
        text = "\n\n".join(_paragraphs(page.text))
        page_chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind("\n\n", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            content = text[start:end].strip()
            if content:
                page_chunks.append(content)
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
        for index, content in enumerate(page_chunks):
            identity = f"{source.source_id}:{page.page_number}:{index}:{content}"
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            metadata = SourceMetadata(
                **{
                    **source.__dict__,
                    "extraction_warnings": tuple(dict.fromkeys(source.extraction_warnings + page.warnings)),
                }
            )
            chunks.append(Chunk(chunk_id, content, metadata, page.page_number, index))
    return chunks
