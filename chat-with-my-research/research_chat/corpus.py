"""Local SQLite FTS corpus, Markdown-aware chunking, and source provenance."""

import hashlib
import re
import sqlite3
from pathlib import Path

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}
DEFAULT_DB = Path("data/research.sqlite3")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def chunk_markdown(text: str, max_chars: int = 1400, overlap_chars: int = 180) -> list[tuple[str, str]]:
    if max_chars < 1 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("max_chars must be positive and overlap_chars must be between zero and max_chars")
    sections: list[tuple[str, str]] = []
    heading = "Introduction"
    body: list[str] = []
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            content = "\n".join(body).strip()
            if content:
                sections.append((heading, content))
            heading = match.group(2)
            body = []
        else:
            body.append(line)
    content = "\n".join(body).strip()
    if content:
        sections.append((heading, content))

    chunks: list[tuple[str, str]] = []
    for section, section_text in sections:
        paragraphs = section_text.split("\n\n")
        current = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            while len(paragraph) > max_chars:
                if current:
                    chunks.append((section, current))
                    current = ""
                split_at = paragraph.rfind(" ", 0, max_chars + 1)
                if split_at <= 0:
                    split_at = max_chars
                chunks.append((section, paragraph[:split_at].strip()))
                paragraph = paragraph[max(0, split_at - overlap_chars):].strip()
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= max_chars:
                current = candidate
            else:
                chunks.append((section, current))
                current = paragraph
        if current:
            chunks.append((section, current))
    return chunks


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
            source_id UNINDEXED, title UNINDEXED, source_path UNINDEXED,
            section UNINDEXED, content, tokenize='unicode61 remove_diacritics 2'
        )"""
    )
    return connection


def ingest_directory(
    directory: str | Path, db_path: str | Path = DEFAULT_DB, max_chars: int = 1400
) -> dict[str, int]:
    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"corpus path is not a directory: {root}")
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )
    if not files:
        raise ValueError(f"no Markdown or text documents found in {root}")

    documents: list[tuple[str, str, str, str, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(root).as_posix()
        if not text.strip():
            continue
        chunks = chunk_markdown(text, max_chars=max_chars)
        for number, (section, content) in enumerate(chunks, start=1):
            digest = hashlib.sha256(f"{relative_path}:{number}:{content}".encode()).hexdigest()[:12]
            source_id = f"{relative_path}#chunk-{number:03d}-{digest}"
            documents.append((source_id, path.stem, relative_path, section, content))
    if not documents:
        raise ValueError(f"no non-empty documents found in {root}")

    with _connect(db_path) as connection:
        connection.execute("DELETE FROM chunks")
        connection.executemany(
            "INSERT INTO chunks (source_id, title, source_path, section, content) VALUES (?, ?, ?, ?, ?)",
            documents,
        )
        connection.commit()
    return {"document_count": len(files), "chunk_count": len(documents)}


def search_corpus(
    query: str, db_path: str | Path = DEFAULT_DB, limit: int = 8
) -> list[dict[str, str | float]]:
    if not query.strip():
        return []
    if limit < 1:
        raise ValueError("limit must be positive")
    terms = list(dict.fromkeys(_TOKEN.findall(query.casefold())))
    if not terms:
        return []
    match_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
    path = Path(db_path)
    if not path.exists():
        return []
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """SELECT source_id, title, source_path, section, content, bm25(chunks) AS rank
               FROM chunks WHERE chunks MATCH ? ORDER BY rank LIMIT ?""",
            (match_query, limit),
        ).fetchall()
    return [
        {
            "source_id": row["source_id"],
            "title": row["title"],
            "source_path": row["source_path"],
            "section": row["section"],
            "content": row["content"],
            "score": -float(row["rank"]),
        }
        for row in rows
    ]


def list_sources(db_path: str | Path = DEFAULT_DB) -> list[dict[str, str]]:
    path = Path(db_path)
    if not path.exists():
        return []
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT source_path, title, COUNT(*) AS chunk_count FROM chunks GROUP BY source_path, title ORDER BY source_path"
        ).fetchall()
    return [
        {"source_path": row[0], "title": row[1], "chunk_count": str(row[2])}
        for row in rows
    ]
