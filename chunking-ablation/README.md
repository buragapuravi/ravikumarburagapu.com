# Chunking Strategy Ablation

Compare fixed-size, recursive, sentence-window, and semantic chunking in one controlled retrieval-and-QA experiment. The same original corpus, TF-IDF embedder, `k=5`, question set, answerer, and judge configuration are held constant for all four strategies.

> **Corpus notice:** The included engineering notes and 50 Q/A pairs are original illustrative fixtures, not the author's private research, clinical guidance, or an externally validated domain benchmark. Use only corpus content you own or are authorized to process.

## Quick start

Requires Python 3.11+.

```sh
cd chunking-ablation
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
chunking-ablation
pytest
```

The default command writes:

- `results/benchmark.json`: aggregate metrics and all per-question evidence, contexts, and answers.
- `results/human-review.csv`: editable per-answer review sheet; human judgments start as `pending`.

The bundled fixture has twelve original illustrative documents, including distractor notes, and 50 hand-authored questions with expected answers and evidence sentences.

## Strategies

| Strategy | Retrieval unit | Context sent to answerer |
|---|---|---|
| `fixed` | 512-token blocks, 50-token overlap | Retrieved block |
| `recursive` | Paragraphs, then sentences, packed to a 512-token target | Retrieved block |
| `sentence-window` | One sentence | That sentence plus up to two sentences on each side |
| `semantic` | Adjacent sentences grouped while cosine similarity stays above a threshold (or until the 512-token cap) | Retrieved group |

All chunk and context token counts use `tiktoken`'s `cl100k_base` encoding. Semantic chunking uses adjacent-sentence cosine similarity from the same TF-IDF vectorizer used by retrieval; it is not a separate learned embedding model. The vectorizer is fit once, shared across strategies, and each strategy's chunk vectors are precomputed once before query timing.

## Answering, accuracy, cost, and review

By default, the benchmark uses a deterministic extractive answerer. It makes no network calls and incurs no provider charge. `exact_answer_accuracy` is strict normalized exact match against the reference. `mean_reference_phrase_coverage` is a token-overlap diagnostic. Neither is a substitute for semantic correctness; every answer is written to the review CSV for human validation.

For model-generated answers and optional LLM-as-judge scoring, configure an OpenAI-compatible endpoint:

```sh
export OPENAI_API_KEY="..."
# Optional for compatible gateways:
export OPENAI_BASE_URL="https://your-approved-endpoint/v1"
chunking-ablation --answerer openai-compatible --judge \
  --model gpt-4o-mini \
  --input-cost-per-million 0.15 \
  --output-cost-per-million 0.60
```

`--judge` sends the question, reference, generated answer, and evidence to the selected model. Judge scores are estimates, not ground truth; inspect the CSV and edit `human_review` before citing a conclusion. Use an approved endpoint for the corpus: retrieved documents and Q/A examples are sent to that provider when `--answerer openai-compatible` or `--judge` is enabled. The default local mode remains the safe, reproducible baseline.

Cost is estimated from provider-reported prompt/completion token usage in LLM mode and the configured per-million token prices. In local mode, only the retrieved context is counted and default prices are zero. Zero means unpriced/not configured, not that a deployed system is free.

## Metrics

- **Retrieval recall@5:** whether at least one of the top five retrieved units contains the exact gold evidence sentence.
- **Answer accuracy:** strict exact match for local extractive answers; optional LLM-judge accuracy is reported separately.
- **Context tokens:** mean `cl100k_base` tokens included across the five retrieved contexts, including repeated window context.
- **Latency:** mean and p95 retrieval and end-to-end timings; LLM mode includes provider round trips.
- **Cost/query:** estimated model prompt/completion token charge using supplied rates.
- **Diagnostics:** token phrase coverage, chunk count and mean chunk size, per-query contexts, retrieval IDs, and judge/human review status.

Chunk sizes are targets; sentence boundaries may make actual sizes vary. Corpus, embedder, fixed `k`, and Q/A set stay the same, but retrieved context length intentionally differs by strategy.

## Options

```sh
chunking-ablation --help
chunking-ablation \
  --target-tokens 512 --overlap-tokens 50 \
  --semantic-threshold 0.12 --window-radius 2 \
  --corpus samples/corpus --qa samples/qa.jsonl \
  --output-dir results
```

## License

MIT. See [LICENSE](LICENSE).
