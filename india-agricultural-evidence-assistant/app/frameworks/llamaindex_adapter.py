from __future__ import annotations

from functools import lru_cache

from app.models import Chunk
from app.storage import ChromaStore, metadata_dict


@lru_cache(maxsize=2)
def _embedding_model(model_name: str, device: str):
    from huggingface_hub.constants import HF_HUB_CACHE
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(
        model_name=model_name,
        device=device,
        cache_folder=HF_HUB_CACHE,
    )


def ingest(store: ChromaStore, chunks: list[Chunk]) -> None:
    try:
        import chromadb
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.core.schema import TextNode
        from llama_index.vector_stores.chroma import ChromaVectorStore
    except ImportError as error:
        raise RuntimeError("The default LlamaIndex ingestion adapter is unavailable.") from error
    client = chromadb.PersistentClient(path=str(store.path))
    collection = client.get_collection(name=store.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    nodes = [
        TextNode(
            id_=chunk.chunk_id,
            text=chunk.text,
            metadata=metadata_dict(chunk),
        )
        for chunk in chunks
    ]
    VectorStoreIndex(
        nodes,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=_embedding_model(store.embedding_model, store.embedding_device),
    )


def retrieve(store: ChromaStore, query: str, limit: int) -> list[tuple[Chunk, float]]:
    try:
        import chromadb
        from llama_index.core import VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore
    except ImportError as error:
        raise RuntimeError(
            "The default LlamaIndex adapter is unavailable. Install the project dependencies."
        ) from error

    client = chromadb.PersistentClient(path=str(store.path))
    collection = client.get_collection(name=store.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed_model = _embedding_model(store.embedding_model, store.embedding_device)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    nodes = index.as_retriever(similarity_top_k=limit).retrieve(query)
    chunks = store.get_chunks([node.node.node_id for node in nodes])
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return [
        (by_id[item.node.node_id], float(item.score or 0.0))
        for item in nodes
        if item.node.node_id in by_id
    ]
