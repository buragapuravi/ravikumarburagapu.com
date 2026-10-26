# BenchmarkKit

BenchmarkKit is an open-source starter toolkit for comparing LLM inference servers and RAG pipelines with repeatable workloads. It measures latency, streaming time-to-first-token (TTFT), request and token throughput, estimated token cost, retrieval recall@k, and end-to-end RAG latency.

## What it measures

| Area | Metrics |
| --- | --- |
| Inference | Request throughput, output-token throughput, total-token throughput, P50/P95/P99 latency and TTFT, estimated cost per 1K tokens |
| RAG | Recall@1/@3/@5 against relevant document IDs, P50/P95/P99 end-to-end latency, lexical answer-support proxy |

Inference currently targets the OpenAI-compatible chat-completions API exposed by vLLM, TGI, SGLang, and TensorRT-LLM. A framework name is recorded as a label so runs against different backends can be compared. TTFT is measured from request start to the first streaming content delta; non-streaming calls use full response time as a TTFT upper bound. Token metrics use the server's usage response; servers that omit usage report zero rather than estimated token counts.

The RAG faithfulness value is intentionally labeled a **proxy**: it measures answer-term overlap with supplied contexts. It is deterministic, inexpensive, and useful for smoke comparisons, but it is not a semantic faithfulness score or a substitute for a calibrated human/LLM judge. The RAG command evaluates saved pipeline outputs; it does not prescribe a retrieval framework.

## Quick start

Requires Python 3.11+ and an OpenAI-compatible inference endpoint.

```sh
cd benchmarkkit
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dashboard,dev]"

# Point at a local vLLM/TGI/SGLang/TensorRT-LLM endpoint.
benchmarkkit inference \
  --dataset samples/prompts.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model your-served-model \
  --framework vllm \
  --requests 100 \
  --concurrency 8 \
  --max-tokens 128 \
  --input-price 0.15 \
  --output-price 0.60

# Evaluate saved retrieval/generation outputs.
benchmarkkit rag --dataset samples/rag-evaluation.jsonl --output results/rag.json

# Normalize downloaded HotpotQA or Natural Questions samples for inference.
benchmarkkit prepare --dataset hotpotqa --input /data/hotpot_train_v1.1.json \
  --limit 100 --output samples/hotpot-prompts.jsonl

# View JSON results.
streamlit run dashboard.py
```

The inference command writes `results/inference.json` by default and starts a Prometheus exporter on port `9464` for the duration of the run. Pass `--metrics-port 0` to disable it. Set `OPENAI_API_KEY` for authenticated endpoints, or supply `--api-key`. Prices are USD per one million input/output tokens; zero is the default for local deployments. Do not put credentials in result files or commit them.

## Docker and observability

Start the dashboard and Prometheus:

```sh
cd benchmarkkit
mkdir -p results
docker compose up --build
```

Open Streamlit at `http://localhost:8501` and Prometheus at `http://localhost:9090`. In another terminal, run the benchmark in the project environment as above; Prometheus scrapes the CLI's exporter at `host.docker.internal:9464` while the run is active. To run the CLI in Docker instead, build it with `docker build -t benchmarkkit .`, then pass the endpoint and mount a result directory; on macOS, use `host.docker.internal` to reach a host-served inference endpoint.

Exported Prometheus histograms:

- `benchmarkkit_request_latency_seconds`
- `benchmarkkit_ttft_seconds`
- `benchmarkkit_output_tokens`

Each is labeled by `framework`. Results are JSON, making them straightforward to archive and compare in CI.

## Dataset formats

Use JSON Lines, one object per line. `samples/prompts.jsonl` and `samples/rag-evaluation.jsonl` are tiny synthetic smoke-test fixtures, not benchmark-quality datasets.

Inference records:

```json
{"prompt":"Explain retrieval recall at k in one sentence."}
```

RAG evaluation records are produced by your own retrieval/generation pipeline. Supply the gold relevant IDs, ordered retrieved IDs, answer, retrieved context text, and end-to-end seconds:

```json
{"query":"What does the system measure?","expected_doc_ids":["doc-1"],"retrieved_doc_ids":["doc-1","doc-2"],"answer":"It measures latency.","contexts":["The system measures request latency."],"latency_seconds":0.4}
```

`prepare` accepts the HotpotQA JSON-array format (`question`, optional `_id`) and Natural Questions JSONL rows (`question_text`, optional `example_id`). It writes one prompt JSONL record per question; those prompts can be passed directly to `inference`. Obtain source datasets from [HotpotQA](https://hotpotqa.github.io/) or [Natural Questions](https://ai.google.com/research/NaturalQuestions), and follow their licenses and terms.

For real comparisons, pin and document your public dataset version, sampling method, prompt template, server/model revision, hardware, warm-up, concurrency, and token pricing. The included tiny sample fixtures are synthetic smoke tests, not benchmark-quality public dataset samples.

## Reproducibility notes

- Use the same prompt set, model revision, decoding settings, hardware, endpoint settings, and pricing for every framework.
- Run warm-up requests separately; BenchmarkKit reports all requests in the selected input.
- `--requests` caps the number of loaded prompts; prompts repeat when the cap exceeds the dataset size.
- Run multiple trials and compare distributions; one run is not a statistically robust result.
- P99 is nearest-rank; tiny sample sizes cannot provide a meaningful tail estimate.
- Do not compare token throughput when a server does not return usage counts.

## Development

```sh
pip install -e ".[dev]"
pytest
```

Contributions are welcome. Please include a focused test, document metric assumptions and dataset licensing, and avoid publishing endpoint credentials or private prompts. Keep backend adapters OpenAI-compatible where possible; propose a separate adapter when wire protocols differ.

## License

MIT. See [LICENSE](LICENSE).
