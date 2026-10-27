"""Validated readers for corpus and evaluation fixture files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_corpus(path: Path) -> list[str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"corpus file does not exist or is not a regular file: {path}")
    texts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not texts:
        raise ValueError(f"corpus file contains no non-empty lines: {path}")
    return texts


def load_jsonl(path: Path, required: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("each JSONL record must be an object")
            for key in required:
                if key not in row:
                    raise ValueError(f"missing field {key!r}")
            if not isinstance(row.get("id"), str) or not row["id"].strip():
                raise ValueError("id must be a non-empty string")
            if row["id"] in seen:
                raise ValueError(f"duplicate id {row['id']!r}")
            seen.add(row["id"])
            rows.append(row)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError(f"no records found in {path}")
    return rows


def load_analogies(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path, ("id", "category", "a", "b", "c", "expected"))
    for index, row in enumerate(rows, 1):
        for key in ("category", "a", "b", "c", "expected"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"analogy row {index}: {key} must be a non-empty string")
    return rows


def load_neighbors(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path, ("id", "category", "word", "relevant"))
    for index, row in enumerate(rows, 1):
        if (
            not isinstance(row["category"], str)
            or not row["category"].strip()
            or not isinstance(row["word"], str)
            or not row["word"].strip()
            or not isinstance(row["relevant"], list)
            or not row["relevant"]
            or any(not isinstance(word, str) or not word.strip() for word in row["relevant"])
        ):
            raise ValueError(f"neighbor row {index}: invalid category, word, or relevant terms")
    return rows
