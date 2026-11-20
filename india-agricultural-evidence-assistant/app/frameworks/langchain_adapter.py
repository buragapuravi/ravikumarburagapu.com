from __future__ import annotations

from functools import lru_cache

from app.models import Chunk
from app.storage import ChromaStore, metadata_dict


@lru_cache(maxsize=2)
def _embedding_model(model_name: str, device: str):
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest(store: ChromaStore, chunks: list[Chunk]) -> None:
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
    except ImportError as error:
        raise RuntimeError("Install the optional LangChain adapter with `pip install -e '.[langchain]'`.") from error
    vector_store = Chroma(
        collection_name=store.collection_name,
        persist_directory=str(store.path),
        embedding_function=_embedding_model(store.embedding_model, store.embedding_device),
    )
    vector_store.add_documents(
        [
            Document(page_content=chunk.text, metadata=metadata_dict(chunk))
            for chunk in chunks
        ],
        ids=[chunk.chunk_id for chunk in chunks],
    )


def retrieve(store: ChromaStore, query: str, limit: int) -> list[tuple[Chunk, float]]:
    """Run the same Chroma query through LangChain's vector-store integration."""
    try:
        from langchain_chroma import Chroma
        from langchain_chroma import Chroma
    except ImportError as error:
        raise RuntimeError(
            "LangChain support is optional. Install with `pip install -e '.[langchain]'`."
        ) from error
    embedding = _embedding_model(store.embedding_model, store.embedding_device)
    vector_store = Chroma(
        collection_name=store.collection_name,
        persist_directory=str(store.path),
        embedding_function=embedding,
    )
    records = vector_store.similarity_search_with_relevance_scores(query, k=limit)
    chunks = store.get_chunks([doc.metadata["chunk_id"] for doc, _ in records])
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return [
        (by_id[doc.metadata["chunk_id"]], float(score))
        for doc, score in records
        if doc.metadata.get("chunk_id") in by_id
    ]
