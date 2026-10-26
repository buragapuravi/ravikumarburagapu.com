# Chat with My Research

An agentic, citation-first RAG application for asking questions about a personal research corpus. It uses LangGraph to decompose questions, retrieve passages, optionally rerank with Cohere, and generate answers with source citations.

> **Corpus notice:** The bundled sample notes are synthetic material written to exercise the software. They are not presented as Ravikumar Buragapu's papers, conference talks, or personal views. Add only documents you own or are authorized to process. No private source corpus is included.

## Quick start

Requires Python 3.11+.

```sh
cd chat-with-my-research
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Use the explicitly synthetic demo corpus.
research-chat ingest data/demo-corpus
research-chat ask "What does the sample note say about agent autonomy?"
streamlit run dashboard.py
```

Without model credentials, answers are **extractive previews** of the retrieved passages with `[S1]` citations. This mode makes no model-provider calls. Add your own local Markdown or text files to a directory and ingest them to build your personal index.

### Live answer generation

Set `OPENAI_API_KEY` in your shell to enable LangGraph query planning and cited answer generation with the configured OpenAI chat model. `RESEARCH_CHAT_MODEL` defaults to `gpt-4o-mini`. Use your organization's approved model endpoint and avoid sending restricted material to an unapproved provider. Retrieved passages are included in model requests.

### Cohere reranking (optional)

```sh
pip install -e ".[rerank]"
export COHERE_API_KEY="..."
```

When configured, Cohere reranks retrieved passages with `rerank-v3.5`. Without the key, the app uses SQLite FTS relevance ranking. Do not use personal or confidential corpus content with a provider unless authorized.

## Add your corpus

1. Create a private local directory for authorized `.md`, `.markdown`, and `.txt` material.
2. Ingest it with `research-chat ingest /path/to/your/corpus`.
3. Ask questions with `research-chat ask "..."`, or point the Streamlit app to the database:

   ```sh
   RESEARCH_CHAT_DB=data/research.sqlite3 streamlit run dashboard.py
   ```

The index stays local in `data/research.sqlite3`, which is git-ignored. Re-ingesting replaces the selected index; use a separate `--db` for a separate corpus. Chunk IDs are deterministic for a given relative path and chunk content. Supported source files remain your responsibility to authorize, license, and curate.

## Architecture

```text
User ──▶ Streamlit / CLI
             │
             ▼
       LangGraph workflow
       1. Query planning / comparison decomposition
       2. SQLite FTS retrieval across planned queries
       3. Optional Cohere reranking
       4. Extractive preview or grounded, cited model answer
             │
             ▼
       [S1] ── local relative path + heading + passage
       [S2] ── local relative path + heading + passage
```

Retrieval uses a local SQLite FTS5 index to make the demo self-contained; it does not require a vector database, embeddings, or network access. A Cohere reranker is optional. This version does not claim Pinecone/Weaviate vector search. Comparison queries preserve the original question and add focused subqueries.

## Example questions

On the bundled synthetic sample corpus:

- “What does the sample note say about agent autonomy?”
- “Compare evaluation to observability.”
- “What sample notes mention identity or accountability?”

The included sample corpus supports software smoke tests only; it does not contain the owner's actual research or talks.

## Retrieval and citation evaluation

Evaluation examples are JSONL with gold source IDs. The expected IDs can be stable prefixes of chunk citations:

```sh
research-chat evaluate \
  --dataset samples/evaluation.jsonl \
  --output results/evaluation.json
```

Reported metrics include recall@k, precision@k, reciprocal rank@k, citation precision, unsupported-citation count, citation coverage, and optional required-term answer coverage. The evaluation fixtures are synthetic examples, not an independently validated benchmark.

## MCTS decomposition experiments

The `search` command explores multi-hop decompositions against the same local retriever:

```sh
research-chat search \
  "Compare identity boundaries with observability in the sample engineering notes." \
  --algorithm mcts --iterations 20 --max-hops 3 --alpha 0.05
```

Each search node is a partial list of subquestions. Expanding a node proposes candidate next questions, retrieves evidence for each candidate, estimates confidence, and scores the evidence for the original question. With `OPENAI_API_KEY`, LangGraph-adjacent components use a structured LLM proposer and value function; without it, deterministic lexical proposals and evidence-coverage scoring run locally. The planner's score is an estimate, not answer correctness.

- **MCTS** selects with UCT (`mean value + exploration × sqrt(log(parent visits) / child visits)`), expands alternatives, and performs a greedy bounded rollout.
- **A\*** prioritizes cumulative retrieval-call cost. Future calls are non-negative, so its zero remaining-cost heuristic is admissible; normalized retrieval confidence breaks equal-cost ties, and alpha prunes low-confidence branches.
- **Beam search** retains the highest value-minus-call-cost partial plans at each depth.
- **Self-Ask, least-to-most, linear, and ReAct-style** baselines use the same proposer, value function, and retriever.

Confidence is lexical query-term coverage in retrieved passages (0–1), not a calibrated probability. `--alpha` prunes a branch when its confidence is below the threshold. Each run caches repeated identical queries and reports unique retrieval calls and wall-clock planner latency. LLM planning/value calls are additional model latency/cost; retrieval-call count is not a token or dollar-cost estimate.

Compare all seven strategies on a labeled JSONL set:

```sh
research-chat ingest data/demo-corpus
research-chat compare \
  --dataset samples/evaluation.jsonl \
  --algorithms mcts astar beam self_ask least_to_most linear react \
  --output results/decomposition-comparison.json
```

Each row reports source-label recall@k, precision@k, reciprocal rank, **multi-hop source accuracy** (all expected source labels retrieved within k), retrieval calls, and latency. This measures retrieval evidence coverage, not end-answer factual accuracy. Convert authorized HotpotQA, MuSiQue, or 2WikiMultiHop examples to JSONL with `question`, `expected_sources` (stable source/chunk IDs), and optional `required_terms`. Document benchmark version, source-to-chunk mapping, split, and sampling. No third-party benchmark corpus is bundled.

The Streamlit app can compare strategies on its labeled synthetic fixture, plot multi-hop source accuracy versus retrieval calls, and display a bounded search tree; pruned branches are red. The chart is a small-fixture demonstration, not a published Pareto result.

## Limitations and privacy

- Citation references point to retrieved chunks, not proof that the answer is correct; verify passages.
- Extractive mode is intentionally not described as generative AI.
- FTS is lexical retrieval and may miss semantic matches. Reranking is optional and does not replace retrieval evaluation.
- There is no public hosted demo or remote multi-user service. Do not index confidential material on a public deployment.
- The Streamlit app is intended for local use. Live model and reranking calls send retrieved text to their respective providers.
- Uploaded documents are not persisted through the browser; index files are local and ignored by Git.

## Development

```sh
pip install -e ".[dev]"
pytest
```

Contributions should add tests and document retrieval behavior, source provenance, and evaluation assumptions.

## License

MIT. See [LICENSE](LICENSE).
