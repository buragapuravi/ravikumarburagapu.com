"""Streamlit viewer for BenchmarkKit's JSON result files."""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def _load_results(directory: Path) -> list[tuple[str, dict[str, Any]]]:
    if not directory.exists():
        return []
    results = []
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as result_file:
            result = json.load(result_file)
        results.append((path.name, result))
    return results


st.set_page_config(page_title="BenchmarkKit results", page_icon="📈", layout="wide")
st.title("BenchmarkKit")
st.caption("LLM inference and RAG benchmark results")
results_dir = Path(os.getenv("BENCHMARKKIT_RESULTS_DIR", "results"))
results = _load_results(results_dir)
if not results:
    st.info(f"No result files found in `{results_dir}`. Run a benchmark and save its JSON output here.")
    st.code("benchmarkkit inference --dataset samples/prompts.jsonl --model your-model")
    st.stop()

inference_runs = [
    (name, result)
    for name, result in results
    if result.get("kind") == "inference"
]
rag_runs = [
    (name, result)
    for name, result in results
    if result.get("kind") == "rag"
]
if inference_runs:
    st.subheader("Inference comparison")
    comparison = pd.DataFrame(
        [
            {
                "Run": name,
                "Framework": result.get("framework", "unknown"),
                "Requests / second": result["requests_per_second"],
                "Output tokens / second": result["output_tokens_per_second"],
                "P99 latency (ms)": result["latency_ms"]["p99"],
                "P99 TTFT (ms)": result["ttft_ms"]["p99"],
                "Cost / 1K tokens (USD)": result["cost_per_1k_tokens_usd"],
            }
            for name, result in inference_runs
        ]
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    chart_columns = st.columns(2)
    chart_columns[0].caption("P99 latency and time to first token")
    chart_columns[0].bar_chart(comparison.set_index("Run")[["P99 latency (ms)", "P99 TTFT (ms)"]])
    chart_columns[1].caption("Request and output-token throughput")
    chart_columns[1].bar_chart(comparison.set_index("Run")[["Requests / second", "Output tokens / second"]])
if rag_runs:
    st.subheader("RAG comparison")
    rag_comparison = pd.DataFrame(
        [
            {
                "Run": name,
                "Recall@1": result["recall"]["recall_at_1"],
                "Recall@3": result["recall"]["recall_at_3"],
                "Recall@5": result["recall"]["recall_at_5"],
                "Faithfulness proxy": result["faithfulness_proxy"],
                "P99 latency (ms)": result["latency_ms"]["p99"],
            }
            for name, result in rag_runs
        ]
    )
    st.dataframe(rag_comparison, hide_index=True, use_container_width=True)
    st.bar_chart(rag_comparison.set_index("Run")[["Recall@1", "Recall@3", "Recall@5"]])

selected_name = st.selectbox("Result file", [name for name, _ in results])
result = next(data for name, data in results if name == selected_name)
if result.get("kind") == "inference":
    columns = st.columns(4)
    columns[0].metric("Requests / second", f"{result['requests_per_second']:.2f}")
    columns[1].metric("Output tokens / second", f"{result['output_tokens_per_second']:.2f}")
    columns[2].metric("P99 latency", f"{result['latency_ms']['p99']:.0f} ms")
    columns[3].metric("P99 TTFT", f"{result['ttft_ms']['p99']:.0f} ms")
    st.subheader("Latency distribution")
    st.bar_chart(pd.DataFrame({
        "Latency (ms)": result["latency_ms"],
        "TTFT (ms)": result["ttft_ms"],
    }))
    st.metric("Estimated cost / 1K tokens", f"${result['cost_per_1k_tokens_usd']:.6f}")
    st.caption(f"Model: {result.get('model', 'unknown')} · Framework: {result.get('framework', 'unknown')}")
else:
    columns = st.columns(3)
    columns[0].metric("Recall@5", f"{result['recall']['recall_at_5']:.1%}")
    columns[1].metric("Faithfulness proxy", f"{result['faithfulness_proxy']:.1%}")
    columns[2].metric("P99 end-to-end latency", f"{result['latency_ms']['p99']:.0f} ms")
    st.subheader("Retrieval recall")
    st.bar_chart(pd.Series(result["recall"], name="Recall"))
    st.warning("Faithfulness is a lexical-support proxy, not a semantic or LLM-judge evaluation.")

with st.expander("Raw result JSON"):
    st.json(result)
