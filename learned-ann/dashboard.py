"""Interactive recall/QPS and build/memory comparison for saved ANN results."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from learned_ann.benchmark import pareto_frontier

st.set_page_config(page_title="Learned ANN Index", page_icon="🧭", layout="wide")
st.title("Learned ANN Index")
st.caption("Partition search quality, query speed, index build time, and memory estimates")

results_path = Path("results/ann-benchmark.json")
if not results_path.exists():
    st.info("No benchmark result yet. Run the benchmark command below, then refresh this page.")
    st.code(
        "learned-ann benchmark --vectors 4000 --queries 160 "
        "--indexes learned-rmi ivf-pq --output results/ann-benchmark.json"
    )
    st.stop()

with results_path.open(encoding="utf-8") as result_file:
    result = json.load(result_file)

if not result.get("results"):
    st.error("Result file has no completed index runs.")
    st.json(result)
    st.stop()

frame = pd.DataFrame(result["results"])
frame["memory_mib"] = frame["memory_bytes_estimate"] / (1024 * 1024)
frontier = pareto_frontier(result["results"])
frontier_indexes = {item["index"] for item in frontier}
frame["pareto"] = frame["index"].isin(frontier_indexes)
left, right = st.columns(2)
left.metric("Vectors", f"{result['dataset']['vector_count']:,}")
right.metric("Queries", f"{result['dataset']['query_count']:,}")
st.caption(
    f"Dataset: {result['dataset']['kind']} · dimension: {result['dataset']['dimension']} · "
    f"metric: {frame.iloc[0]['metric']}"
)

st.subheader("Recall@k vs queries per second")
st.scatter_chart(frame, x="qps", y="recall_at_k", color="pareto")
st.caption("Colored frontier points are non-dominated: no measured run is both faster and higher recall.")
if frontier:
    st.line_chart(pd.DataFrame(frontier).set_index("index")[["recall_at_k"]])
st.subheader("Measured run results")
st.dataframe(
    frame[
        ["index", "recall_at_k", "k", "qps", "query_latency_ms", "build_time_ms", "memory_mib"]
    ],
    hide_index=True,
    width="stretch",
)
st.caption(
    "Memory is an implementation-specific estimate, not process RSS. HNSW and ScaNN require "
    "their optional native packages. Compare only runs with matching dataset, metric, hardware, "
    "and index parameters."
)
for unavailable in result.get("unavailable", []):
    st.warning(f"{unavailable['index']} unavailable: {unavailable['error']}")
