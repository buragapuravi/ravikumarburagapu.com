from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from typing import Protocol

from app.models import Chunk, SourceMetadata


class VectorStore(Protocol):
    def clear(self) -> None: ...

    def upsert(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, limit: int) -> list[tuple[str, float]]: ...

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]: ...

    def all_chunks(self) -> list[Chunk]: ...

    def count(self) -> int: ...


def metadata_dict(chunk: Chunk) -> dict[str, str | int]:
    values = asdict(chunk.metadata)
    values["extraction_warnings"] = " | ".join(chunk.metadata.extraction_warnings)
    values["page_number"] = chunk.page_number if chunk.page_number is not None else -1
    values["chunk_index"] = chunk.chunk_index
    values["chunk_id"] = chunk.chunk_id
    return values


@lru_cache(maxsize=2)
def _sentence_encoder(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)


class ChromaStore:
    def __init__(
        self,
        path: str,
        collection: str,
        embedding_model: str,
        embedding_device: str = "cpu",
    ) -> None:
        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError("Chroma requires the project dependencies. Install with `pip install -e .`.") from error
        self.path = path
        self.collection_name = collection
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def clear(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = _sentence_encoder(
            self.embedding_model,
            self.embedding_device,
        ).encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
        )
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[metadata_dict(chunk) for chunk in chunks],
            embeddings=embeddings.tolist(),
        )

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        vector = _sentence_encoder(self.embedding_model, self.embedding_device).encode(
            query,
            normalize_embeddings=True,
        )
        result = self._collection.query(query_embeddings=[vector.tolist()], n_results=limit)
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [(chunk_id, max(0.0, 1.0 - float(distance))) for chunk_id, distance in zip(ids, distances)]

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        result = self._collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        by_id = {}
        for chunk_id, text, metadata in zip(result["ids"], result["documents"], result["metadatas"]):
            warnings = tuple(filter(None, metadata.get("extraction_warnings", "").split(" | ")))
            source = SourceMetadata(
                **{
                    key: metadata.get(key, "")
                    for key in SourceMetadata.__dataclass_fields__
                    if key != "extraction_warnings"
                },
                extraction_warnings=warnings,
            )
            page = int(metadata.get("page_number", -1))
            by_id[chunk_id] = Chunk(
                chunk_id=chunk_id,
                text=text,
                metadata=source,
                page_number=None if page < 0 else page,
                chunk_index=int(metadata.get("chunk_index", 0)),
            )
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def all_chunks(self) -> list[Chunk]:
        result = self._collection.get(include=["documents", "metadatas"])
        return self.get_chunks(result["ids"])

    def count(self) -> int:
        return self._collection.count()
