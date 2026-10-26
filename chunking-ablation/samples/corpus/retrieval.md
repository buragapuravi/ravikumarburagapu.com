# Search and Retrieval

Recall at k measures the fraction of relevant items found among the first k retrieved items. Precision at k measures the fraction of the first k retrieved items that are relevant. Mean reciprocal rank rewards a relevant result appearing near the top of a ranking. A retrieval query should preserve the user's important entities and constraints. A reranker reorders a candidate set using a more detailed relevance function.

An inverted index maps terms to documents that contain them. Dense retrieval compares vector representations of queries and passages. Cosine similarity compares the angle between two vectors. A vector normalization step scales a nonzero vector to unit length. A lexical retriever can remain useful for exact names and rare identifiers.

A relevance label records whether a passage helps answer a particular query. A judged query set should include examples from the intended users and domain. A held-out evaluation set should not be used to tune retrieval thresholds. Annotation guidelines should define what counts as sufficient evidence. Disagreements between annotators should be measured and reviewed.

Hybrid retrieval combines lexical and semantic search results. Reciprocal rank fusion combines rankings without requiring their raw scores to share a scale. Candidate generation should return enough results for later filtering. A metadata filter narrows results using document attributes. Duplicate passages can waste a limited retrieval budget.

A query expansion step adds related terms to a search request. Query rewriting should not remove important constraints. A retrieval cache reuses results for repeated identical queries. Freshness requirements determine how frequently an index must be updated. A retrieval failure should be observable rather than represented as an empty successful answer.
