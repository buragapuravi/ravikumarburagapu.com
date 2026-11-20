from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import replace
from pathlib import Path

from app.models import PageText, SourceMetadata

SUPPORTED_TEXT = {".md", ".markdown", ".txt"}


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(path: Path) -> SourceMetadata:
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    details: dict[str, str] = {}
    if sidecar.exists():
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read metadata sidecar {sidecar}: {error}") from error
        if not isinstance(loaded, dict):
            raise ValueError(f"Metadata sidecar must contain a JSON object: {sidecar}")
        details = {str(key): str(value) for key, value in loaded.items()}
    digest = _file_digest(path)
    default_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", path.stem).strip("-")
    source_id = details.get("source_id", f"{default_id}-{digest[:16]}")
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", source_id):
        raise ValueError(f"source_id must contain only letters, numbers, '.', '_' or '-': {source_id!r}")
    title = details.get("title", path.stem.replace("-", " ").replace("_", " ").title())
    allowed = {
        "organization", "source_url", "issued_date", "state", "district",
        "crop", "season", "language", "document_type", "status",
        "source_fetch_date", "validity_start", "validity_end", "license_access_note",
    }
    return SourceMetadata(
        source_id=source_id,
        filename=path.name,
        title=title,
        **{key: details[key] for key in allowed if key in details},
        file_sha256=digest,
        extraction_method=(
            "PyMuPDF page text extraction"
            if path.suffix.casefold() == ".pdf"
            else "UTF-8 text decoding"
        ),
    )


def extract_document(path: Path, ocr: bool = False) -> tuple[SourceMetadata, list[PageText]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in SUPPORTED_TEXT:
        text = _clean(path.read_text(encoding="utf-8"))
        warnings = ["Page numbers are unavailable for this text source."]
        if not text:
            warnings.append("Text source contains no extractable text.")
        return _metadata(path), [
            PageText(None, text, tuple(warnings))
        ]
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported document type: {path.suffix}")

    try:
        import fitz
    except ImportError as error:
        raise RuntimeError("PDF extraction requires PyMuPDF; install the project dependencies.") from error

    pages: list[PageText] = []
    with fitz.open(path) as document:
        metadata = _metadata(path)
        ocr_used = False
        for page_number, page in enumerate(document, start=1):
            text = _clean(page.get_text("text"))
            if len(text) < 20 and ocr:
                ocr_used = True
                try:
                    import pytesseract
                    from PIL import Image
                except ImportError as error:
                    raise RuntimeError(
                        "Explicit OCR requires the OCR extra and a system Tesseract installation."
                    ) from error
                image_bytes = page.get_pixmap(dpi=200).tobytes("png")
                text = _clean(pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes))))
            if len(text) < 20:
                warning = (
                    "Page remains empty after explicitly requested OCR."
                    if ocr
                    else "Page may be scanned or contain no extractable text; OCR was not run."
                )
                warnings = (warning,)
            else:
                warnings = ()
            table_warning = (
                "Possible table layout extracted as text; review row and column relationships."
                if "|" in text or any(line.count("\t") >= 2 for line in text.splitlines())
                else None
            )
            if table_warning:
                warnings += (table_warning,)
            pages.append(PageText(page_number, text, warnings))
    if ocr_used:
        metadata = replace(
            metadata,
            extraction_method="PyMuPDF with explicitly requested Tesseract OCR",
        )
    return metadata, pages
