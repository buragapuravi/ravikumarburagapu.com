# Vector Search Notes

Search systems rank candidate documents based on lexical terms, metadata, or vector similarity. Query intent can be difficult to represent with only exact keywords. Embedding models map passages into a space where related meanings may be nearby, although rare names and identifiers can still need exact matching.

Evaluation sets record queries, candidate passages, and relevance labels. Ranking quality can be summarized at several retrieval depths. Combining lexical retrieval with dense retrieval may improve coverage, but duplicate results and weak candidates still compete for limited context.
