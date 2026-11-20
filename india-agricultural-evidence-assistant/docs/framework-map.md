# Framework map

The app does not nest orchestration frameworks. Shared extraction, chunking, metadata,
lexical retrieval, rank fusion, reranking, safety gates, answer validation, and UI code
are framework-neutral.

## Default: LlamaIndex

`RAG_FRAMEWORK=llamaindex` uses a LlamaIndex `VectorStoreIndex` to embed and ingest
normalized chunks into the persistent Chroma collection, then uses that index for dense
retrieval. The app combines those candidates with an independent BM25 ranking, then
performs reciprocal-rank fusion, optional cross-encoder
reranking, deterministic evidence gating, and grounded generation. This makes the
LlamaIndex retrieval operation visible without hiding the rest of the pipeline in a
framework chain.

```sh
python -m app.ingest --input data/demo --input data/raw --rebuild
streamlit run streamlit_app.py
```

## Comparison: LangChain

The optional adapter performs equivalent document ingestion and dense-retrieval
operations over the same Chroma collection. It does not duplicate extraction, metadata,
lexical ranking, fusion, or answer generation.

```sh
pip install -e '.[langchain]'
RAG_FRAMEWORK=langchain streamlit run streamlit_app.py
```

| Prototype concept | LlamaIndex | LangChain |
|---|---|---|
| Ingestion | `VectorStoreIndex` with stable `TextNode` IDs | Chroma `add_documents()` with stable chunk IDs |
| Dense retrieval | `VectorStoreIndex.as_retriever()` | Chroma `similarity_search_with_relevance_scores()` |
| Persistent vectors | Chroma vector-store integration | `langchain-chroma` |
| Embeddings | `HuggingFaceEmbedding` | `HuggingFaceEmbeddings` |
| Shared ranking and evidence checks | Framework-neutral `app/retrieval.py` | Framework-neutral `app/retrieval.py` |

The lexical retriever and RRF logic are deliberately outside either framework so both
examples use the same normalized chunks, metadata, tests, and storage collection.
