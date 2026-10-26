# RNN vs LSTM vs GRU vs Tiny Transformer

A reproducible small-scale comparison of character-level vanilla RNN, LSTM, GRU, and causal Transformer models. Architectures are automatically sized to approximately match a 1M-parameter budget, then trained on the same bundled text and evaluated on a synthetic long-range recall task.

> **Research prototype:** Results depend on the small corpus, compute backend, training budget, and initialization. This experiment does not establish a universal winner or compare optimized production inference stacks.

## Quick start

Requires Python 3.11+ and PyTorch 2.4+.

```sh
cd sequence-models
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
sequence-models
pytest
```

The default command writes `results/results.json`, `results/samples.txt`, and `results/validation-loss.svg`. The source corpus is a short, original sample bundled in `samples/corpus.txt`; it is repeated for stochastic training windows, while validation is held out by text position before repeating.

For a quick smoke run:

```sh
sequence-models \
  --target-parameters 80000 \
  --epochs 1 --steps-per-epoch 3 --batch-size 4 \
  --sequence-length 24 --corpus-repeats 4 \
  --recall-train 64 --recall-valid 32 --recall-epochs 1 \
  --output-dir /tmp/sequence-models-smoke
```

## Experiment details

- **Language modeling:** next-character prediction over the shared character vocabulary; AdamW, same batch size, sequence length, optimizer, and update count across models.
- **Parameter matching:** recurrent hidden widths and Transformer width are searched to minimize distance to the target count. The Transformer has two layers and four heads by default; recurrent models use one layer. Actual counts and deltas are in the JSON report.
- **Loss curve:** validation cross-entropy is recorded at each epoch with the global training step; per-step training losses are retained for diagnosing optimization.
- **Samples:** each model generates 200 characters from the same corpus prompt. Sampling uses temperature 0.8.
- **Long-range recall:** a binary token at position zero must be classified at the final query marker after 99 distractor tokens. Validation distractors are generated from a separate seed. Random baseline is 50%.
- **Timing:** reports elapsed training time per epoch and autoregressive single-token throughput/mean latency on the selected device. These timings are machine-specific and include Python/framework overhead.
- **Compute:** CPU is the default. Choose `--device mps` or `--device cuda` when supported by your PyTorch installation.

Models and synthetic task models use the same selected architecture widths. The small difference in output head size between character prediction and binary classification is expected.

## Metrics

The JSON results include parameter count, parameter delta from target, per-epoch validation loss, perplexity, generated samples, per-epoch training time, inference tokens/second, mean latency per generated token, and long-range recall accuracy. The SVG plots validation loss against training steps.

Treat the recall task as a controlled probe, not a natural-language benchmark: all distractors are random symbols, and the answer is the first symbol. Repeated seeds, longer contexts, varied distractors, and matched training FLOPs are useful follow-up experiments. The default comparison matches parameters and optimizer steps, not wall-clock budget or compute FLOPs.

## License

MIT. See [LICENSE](LICENSE).
