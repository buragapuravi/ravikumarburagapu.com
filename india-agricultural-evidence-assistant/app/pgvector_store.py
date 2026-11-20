from __future__ import annotations

import json
from typing import Any

from app.models import Chunk
from app.storage import VectorStore


class PgVectorStore(VectorStore):
    """Optional pgvector backend; embeddings are generated locally with SentenceTransformers."""

    def __init__(self, dsn: str, model_name: str, table: str = "agri_chunks") -> None:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Install the optional backend with `pip install -e '.[postgres]'`.") from error
        self._psycopg = psycopg
        self._register_vector = register_vector
        self._model: Any = SentenceTransformer(model_name)
        self._table = table
        self._dsn = dsn
        with self._connect() as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                "chunk_id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata JSONB NOT NULL, "
                "embedding VECTOR NOT NULL)"
            )

    def _connect(self):
        connection = self._psycopg.connect(self._dsn)
        self._register_vector(connection)
        return connection

    def upsert(self, chunks: list[Chunk]) -> None:
        from app.storage import metadata_dict
        if not chunks:
            return
        vectors = self._model.encode([item.text for item in chunks], normalize_embeddings=True)
        with self._connect() as connection:
            for chunk, vector in zip(chunks, vectors):
                connection.execute(
                    f"INSERT INTO {self._table} (chunk_id, text, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s) ON CONFLICT (chunk_id) DO UPDATE SET "
                    "text = EXCLUDED.text, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding",
                    (chunk.chunk_id, chunk.text, json.dumps(metadata_dict(chunk)), vector),
                )

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute(f"DELETE FROM {self._table}")

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        vector = self._model.encode(query, normalize_embeddings=True)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT chunk_id, 1 - (embedding <=> %s) AS score FROM {self._table} "
                "ORDER BY embedding <=> %s LIMIT %s",
                (vector, vector, limit),
            ).fetchall()
        return [(str(row[0]), float(row[1])) for row in rows]

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT chunk_id, text, metadata FROM {self._table} WHERE chunk_id = ANY(%s)",
                (chunk_ids,),
            ).fetchall()
        by_id = {row[0]: (row[1], row[2]) for row in rows}
        from app.models import SourceMetadata
        result = []
        for chunk_id in chunk_ids:
            if chunk_id not in by_id:
                continue
            text, metadata = by_id[chunk_id]
            warnings = tuple(filter(None, metadata.get("extraction_warnings", "").split(" | ")))
            source = SourceMetadata(
                **{key: metadata.get(key, "") for key in SourceMetadata.__dataclass_fields__ if key != "extraction_warnings"},
                extraction_warnings=warnings,
            )
            page = int(metadata.get("page_number", -1))
            result.append(Chunk(chunk_id, text, source, None if page < 0 else page, int(metadata.get("chunk_index", 0))))
        return result

    def all_chunks(self) -> list[Chunk]:
        with self._connect() as connection:
            ids = [
                row[0]
                for row in connection.execute(
                    f"SELECT chunk_id FROM {self._table} ORDER BY chunk_id"
                ).fetchall()
            ]
        return self.get_chunks(ids)

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute(f"SELECT count(*) FROM {self._table}").fetchone()[0])
