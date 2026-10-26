# Passage Construction

A fixed-size chunk contains a configured number of tokens. Overlap repeats a boundary region between neighboring chunks. A paragraph boundary often preserves a coherent topic. A sentence boundary limits splits inside a grammatical statement. A recursive splitter tries preferred separators before falling back to smaller units.

A sentence-window retriever ranks one sentence and expands nearby context after retrieval. A window radius of two includes up to two preceding and two following sentences. A semantic chunker compares adjacent sentence embeddings. A cosine drop threshold starts a new chunk when adjacent sentence similarity is low. A maximum chunk size prevents a semantically coherent passage from growing without bound.

Chunk metadata can retain the source document and sentence offsets. Stable chunk identifiers make experiment results easier to audit. A citation should point to the passage that supports a claim. A chunking evaluation should keep the corpus and retriever constant. Changing the embedding model at the same time makes attribution difficult.

Small chunks can make evidence precise but may omit context. Large chunks can preserve explanations but increase prompt cost. Overlap can preserve facts at boundaries while duplicating content. Repeated context can crowd distinct evidence out of a fixed retrieval budget. A chunking strategy should be assessed against its actual downstream question set.

Token counts depend on the tokenizer used by the generation model. Whitespace word counts are not a reliable substitute for model token counts. Context cost includes all retrieved passages sent to a model. A prompt budget should reserve space for the question and generated response. The same retrieval depth does not guarantee the same context length.
