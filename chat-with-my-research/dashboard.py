"""Streamlit UI for querying the private local research corpus."""

import json
import os
from pathlib import Path

import streamlit as st

from research_chat.corpus import DEFAULT_DB, ingest_directory, list_sources
from research_chat.decomposition import (
    ALGORITHMS,
    DecompositionSearch,
    evaluate_algorithms,
    make_live_components,
)
from research_chat.rag import run_query

st.set_page_config(page_title="Chat with My Research", page_icon="📚", layout="wide")
st.title("Chat with My Research")
st.caption("Citation-first retrieval over your local corpus")
db_path = Path(os.getenv("RESEARCH_CHAT_DB", str(DEFAULT_DB)))
st.info(
    "The repository includes a clearly labeled sample corpus, not the owner's papers or talks. "
    "Add documents you are authorized to use. Your indexed database stays local by default."
)

with st.sidebar:
    st.subheader("Indexed sources")
    sources = list_sources(db_path)
    if sources:
        for source in sources:
            st.caption(f"{source['title']} · {source['chunk_count']} chunks")
    else:
        st.caption("No documents indexed yet.")
    st.divider()
    st.subheader("Index local documents")
    corpus_path = st.text_input("Markdown/text directory", value=os.getenv("RESEARCH_CHAT_CORPUS", "docs"))
    if st.button("Index documents", width="stretch"):
        try:
            with st.spinner("Reading local documents…"):
                summary = ingest_directory(corpus_path, db_path)
            st.success(f"Indexed {summary['document_count']} documents into {summary['chunk_count']} chunks.")
            st.rerun()
        except (OSError, ValueError) as error:
            st.error(str(error))

examples = [
    "What are the key themes in the sample engineering notes?",
    "Compare the sample notes on evaluation and production readiness.",
    "What sources discuss observability?",
]
question = st.chat_input("Ask a question about your indexed research")
if not question:
    st.markdown("**Try a question**")
    selected = st.radio("Example questions", examples, label_visibility="collapsed")
    if st.button("Ask this question", type="primary"):
        question = selected

if question:
    try:
        with st.spinner("Planning searches, retrieving passages, and checking sources…"):
            result = run_query(question, str(db_path))
    except (OSError, ValueError, RuntimeError) as error:
        st.error(str(error))
        st.stop()

    st.caption(f"{result['mode'].title()} answer · {result['reranker']} retrieval/reranking")
    st.markdown(result["answer"])
    for warning in result["warnings"]:
        st.warning(warning)
    if result["sources"]:
        st.subheader("Retrieved sources")
        for source in result["sources"]:
            with st.expander(
                f"[{source['citation']}] {source['title']} · {source['section']}"
            ):
                st.caption(source["source_path"])
                st.write(source["content"])
                st.caption(f"Relevance: {source.get('rerank_score', source['score']):.4f}")
    st.caption("Answers may be incomplete. Check each cited passage before relying on a claim.")

st.divider()
st.header("Compare decomposition strategies")
st.write(
    "Search multi-hop query plans with MCTS, A*, and beam search, then compare with "
    "Self-Ask, least-to-most, linear decomposition, and ReAct-style baselines."
)
compare_question = st.text_input(
    "Multi-hop comparison question",
    value="Compare identity boundaries with observability in the sample engineering notes.",
)
selected_algorithms = st.multiselect(
    "Strategies",
    options=list(ALGORITHMS),
    default=["mcts", "astar", "beam", "self_ask", "least_to_most", "linear", "react"],
)
if st.button("Run planner comparison", type="primary", disabled=not selected_algorithms):
    try:
        proposer, value_function = make_live_components()
        with st.spinner("Exploring decomposition plans…"):
            evaluation_path = Path(__file__).parent / "samples" / "evaluation.jsonl"
            examples = [
                json.loads(line)
                for line in evaluation_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            evaluation = evaluate_algorithms(
                examples,
                algorithms=selected_algorithms,
                db_path=str(db_path),
                proposer=proposer,
                value_function=value_function,
                max_hops=3,
                iterations=16,
                beam_width=4,
            )
            query_runs = {
                "runs": [
                    DecompositionSearch(
                        str(db_path),
                        proposer=proposer,
                        value_function=value_function,
                    ).run(
                        compare_question,
                        algorithm=algorithm,
                        max_hops=3,
                        iterations=16,
                        beam_width=4,
                    )
                    for algorithm in selected_algorithms
                ]
            }
            st.session_state["decomposition_result"] = {
                "evaluation": evaluation,
                "query_runs": query_runs,
            }
    except (OSError, RuntimeError, ValueError) as error:
        st.error(str(error))

comparison_state = st.session_state.get("decomposition_result")
if comparison_state:
    evaluation = comparison_state["evaluation"]
    query_runs = comparison_state["query_runs"]
    run_metrics = [
        {
            "algorithm": run["algorithm"],
            **run["metrics"],
        }
        for run in evaluation["runs"]
    ]
    st.subheader("Labeled sample evaluation")
    st.caption(
        "These examples are synthetic software fixtures, not the owner's papers or talks. "
        "Accuracy means all gold source labels were retrieved within the selected k."
    )
    st.dataframe(run_metrics, hide_index=True, width="stretch")
    st.subheader("Accuracy and retrieval-call frontier")
    st.scatter_chart(
        run_metrics,
        x="retrieval_calls",
        y="multi_hop_accuracy",
        color="algorithm",
    )
    st.caption("Lower calls and higher labeled accuracy are preferable; latency is reported in the table.")

    run_lookup = {run["algorithm"]: run for run in query_runs["runs"]}
    for run in run_lookup.values():
        st.caption(
            f"{run['algorithm']}: {run['retrieval_calls']} unique retrieval calls · "
            f"{run['latency_ms']:.1f} ms on the comparison question"
        )

    chosen = st.selectbox(
        "Inspect search tree",
        options=[run["algorithm"] for run in query_runs["runs"]],
    )
    chosen_run = run_lookup[chosen]

    def dot_escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")[:90]

    def append_tree(node: dict, lines: list[str], count: list[int]) -> None:
        if count[0] >= 100:
            return
        count[0] += 1
        label = f"{node['query']}\\nconfidence={node['confidence']:.2f} value={node['value']:.2f}"
        style = ', color="red", fontcolor="red"' if node["pruned"] else ""
        lines.append(f'"{node["id"]}" [label="{dot_escape(label)}"{style}];')
        if node["parent_id"]:
            lines.append(f'"{node["parent_id"]}" -> "{node["id"]}";')
        for child in node["children"]:
            append_tree(child, lines, count)

    graph_lines = ["digraph decomposition {", 'rankdir="TB";', 'node [shape="box"];']
    append_tree(chosen_run["tree"], graph_lines, [0])
    graph_lines.append("}")
    st.graphviz_chart("\n".join(graph_lines), width="stretch")
    st.json(
        {
            "best_plan": chosen_run["best_plan"],
            "retrieval_calls": chosen_run["retrieval_calls"],
            "latency_ms": chosen_run["latency_ms"],
            "warnings": chosen_run.get("warnings", []),
        }
    )
