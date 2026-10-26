# Validated RAG Eval Harness

Compare three LLM-as-judge prompt styles against **your independent human labels** for answer correctness and citation support. The harness reports nominal Cohen's kappa, raw agreement, confusion counts, per-question-type disagreements, model-token cost, and optional human-label cost.

> **Important:** This repository does not contain human annotations or claim a validated result. Its 50 bundled cases are an illustrative smoke fixture derived from the Chunking Strategy Ablation demo; they are single-hop, reference-copy examples, not a representative RAG evaluation set. Label them yourself or replace them with an authorized, representative corpus and real RAG answers before interpreting agreement. Unlabeled cases are excluded from agreement calculations.

## Quick start

Requires Python 3.11+.

```sh
cd rag-judge-eval
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest

# First, create the independent human annotation sheet.
rag-judge-eval prepare
# Read each answer and citation; fill in answer label and citation support.

# Only after human review, run the three LLM judges.
export OPENAI_API_KEY="..."
rag-judge-eval run \
  --model gpt-4o-mini \
  --input-cost-per-million 0.15 \
  --output-cost-per-million 0.60 \
  --human-hourly-rate 60 \
  --human-minutes-per-case 2
```

The judge sends **three requests per case**. With 50 cases this is 150 model calls. Use an approved provider endpoint and only data you are authorized to send. `OPENAI_BASE_URL` can select an OpenAI-compatible endpoint. The CLI fails explicitly on API, schema, and annotation errors; it does not silently turn failed judgments into scores.

## Evaluation protocol

1. Supply one JSONL record per real RAG answer, including the reference answer and the passages actually cited by the system.
2. Run `prepare` and annotate answers without seeing judge results. Use `correct`, `partially_correct`, or `wrong`; label citation support separately as `true` or `false`.
3. Keep the generated case content unchanged in the CSV. Add notes in `human_notes` where useful.
4. Run all three prompts, then inspect `results/report.json`. Agreement is computed only for rows with both human labels present.
5. Repeat on a representative sample, and review disagreements with a second annotator where possible. One annotator's labels are a reference, not objective ground truth.

The third prompt asks the model to assess claims, reference agreement, and citation support in sequence, but does **not** request or expose private chain-of-thought. Its short rationale is returned for audit.

## Case format

```json
{
  "id": "case-001",
  "question_type": "multi_hop",
  "question": "What follows when ...?",
  "reference_answer": "The grounded reference answer.",
  "candidate_answer": "The answer produced by the RAG system.",
  "cited_passages": ["The passage the RAG answer actually cited."]
}
```

`question_type` is required and user-defined. Use consistent categories relevant to the study, such as `multi_hop`, `numeric`, `negation`, and `single_hop`. The sample cases all use `single_hop`, so the included fixture cannot support comparisons between those categories.

## Metrics and cost

- **Answer agreement:** fraction of paired human/judge answer labels that match.
- **Cohen's κ:** nominal, unweighted 3-class chance-corrected agreement. It is `null` when undefined (for example, empty or constant marginals).
- **Citation agreement and κ:** reported separately for the binary support judgment.
- **Per-question-type errors:** paired count, agreement, kappa, and answer/citation disagreement counts for each provided category.
- **Judge cost:** reported per prompt from provider-reported token usage and configured per-million input/output rates. The report provides both cost per 1,000 cases and per 1,000 individual label decisions (answer and citation are two decisions per case). Missing provider usage is reported as zero tokens; configure prices for meaningful cost estimates.
- **Human cost:** optional estimate from completed human cases × minutes per case × hourly rate, with cost per 1,000 reviewed cases. This is a stated labor-cost model, not a market-rate claim.

Kappa can be low even with high raw agreement when prevalence is highly skewed. Always report sample size, class counts, raw agreement, and the confusion matrix alongside κ. This harness reports confusion counts for answer labels; retain and inspect the case-level judgments in the JSON report.

## Files

- `samples/cases.jsonl`: 50 illustrative, unlabeled smoke cases.
- `results/human-review.csv`: generated blank human annotation template.
- `results/report.json`: per-case judgments, token usage, prompt-level metrics, and costs.

## License

MIT. See [LICENSE](LICENSE).
