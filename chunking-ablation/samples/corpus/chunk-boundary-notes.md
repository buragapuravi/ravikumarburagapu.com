# Chunk Boundary Notes

A chunking evaluation should compare boundaries without changing the retriever or question set. Overlap can preserve a sentence near a boundary, but it also repeats tokens. Sentence-oriented retrieval may expand a small result at read time, changing the context size after ranking.

Semantic grouping uses a similarity threshold to decide whether adjacent content belongs together. The threshold depends on the embedding representation and document style. A maximum size remains useful when adjacent statements are similar for many consecutive sentences.
