import json
import tempfile
import unittest
from pathlib import Path

from research_chat.corpus import chunk_markdown, ingest_directory, search_corpus
from research_chat.decomposition import ALGORITHMS, DecompositionSearch
from research_chat.evaluation import (
    evaluate_answer_coverage,
    evaluate_citations,
    evaluate_example,
    evaluate_retrieval,
)
from research_chat.rag import build_rag_graph, decompose_query


class CorpusTests(unittest.TestCase):
    def test_markdown_sections_are_split_with_stable_titles(self) -> None:
        chunks = chunk_markdown("# First\n\nalpha\n\n# Second\n\nbeta", max_chars=50, overlap_chars=5)
        self.assertEqual(chunks, [("First", "alpha"), ("Second", "beta")])

    def test_ingest_and_search_returns_stable_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "docs"
            root.mkdir()
            (root / "one.md").write_text("# Evidence\n\nCitations identify supporting passages.", encoding="utf-8")
            (root / ".hidden.md").write_text("# Hidden\n\nMust not be indexed.", encoding="utf-8")
            db = Path(temporary_directory) / "index.sqlite3"

            summary = ingest_directory(root, db)
            first = search_corpus("supporting passages", db)
            second = search_corpus("supporting passages", db)

        self.assertEqual(summary["document_count"], 1)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["source_id"], second[0]["source_id"])
        self.assertEqual(first[0]["section"], "Evidence")


class QueryTests(unittest.TestCase):
    def test_decomposes_comparison_without_losing_original_question(self) -> None:
        self.assertEqual(
            decompose_query("Compare evaluation to observability"),
            ["Compare evaluation to observability", "evaluation", "observability"],
        )

    def test_extractive_graph_cites_retrieved_passages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            docs = Path(temporary_directory) / "docs"
            docs.mkdir()
            (docs / "notes.md").write_text(
                "# Evaluation\n\nEvaluation evidence should support autonomy decisions.",
                encoding="utf-8",
            )
            db = Path(temporary_directory) / "index.sqlite3"
            ingest_directory(docs, db)
            result = build_rag_graph(str(db)).invoke(
                {"question": "What supports autonomy decisions?"}
            )

        self.assertIn("[S1]", result["answer"])
        self.assertEqual(result["sources"][0]["source_path"], "notes.md")
        self.assertEqual(result["warnings"], [])

    def test_unknown_answer_citations_are_reported(self) -> None:
        from research_chat.rag import _citation_check_node

        checked = _citation_check_node(
            {
                "answer": "Claim [S9].",
                "sources": [{"citation": "S1"}],
                "warnings": [],
            }
        )
        self.assertIn("S9", checked["warnings"][0])

    def test_all_decomposition_strategies_return_search_trees(self) -> None:
        def retrieve(query: str, _db_path: str, limit: int):
            if "identity" in query.casefold():
                content = "Identity boundaries should use least privilege."
                source_id = "identity.md#chunk-1"
            elif "observability" in query.casefold():
                content = "Observability enables tracing source identifiers."
                source_id = "observability.md#chunk-1"
            else:
                content = "Identity and observability provide engineering context."
                source_id = "overview.md#chunk-1"
            return [{"source_id": source_id, "content": content, "score": 0.5}][:limit]

        def propose(_question: str, plan: list[str], _limit: int) -> list[str]:
            if not plan:
                return ["identity boundaries", "observability"]
            if "identity boundaries" in plan and len(plan) == 1:
                return ["least privilege"]
            if "observability" in plan and len(plan) == 1:
                return ["source identifiers"]
            return []

        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                result = DecompositionSearch(
                    retrieval=retrieve,
                    proposer=propose,
                ).run(
                    "Compare identity boundaries with observability.",
                    algorithm=algorithm,
                    max_hops=2,
                    iterations=3,
                    beam_width=2,
                    alpha=0,
                )
                self.assertEqual(result["algorithm"], algorithm)
                self.assertGreaterEqual(result["retrieval_calls"], 1)
                self.assertTrue(result["tree"]["id"].startswith("n"))
                self.assertIn("best_plan", result)

    def test_low_confidence_branches_are_pruned(self) -> None:
        result = DecompositionSearch(
            retrieval=lambda _query, _db, _limit: [],
            proposer=lambda _question, _plan, _limit: ["missing evidence"],
        ).run(
            "Find unsupported claims.",
            algorithm="mcts",
            alpha=0.25,
            iterations=2,
        )
        self.assertTrue(result["tree"]["pruned"])
        self.assertEqual(result["retrieval_calls"], 1)


class EvaluationTests(unittest.TestCase):
    def test_recall_citation_and_answer_coverage_metrics(self) -> None:
        retrieval = evaluate_retrieval(["S1", "S2"], ["S2", "S3"], k=2)
        citations = evaluate_citations("Claim [S2] with bad [S9].", ["S1", "S2"])
        coverage = evaluate_answer_coverage("Evidence supports autonomy decisions.", ["evidence", "decisions"])

        self.assertEqual(retrieval["recall_at_2"], 0.5)
        self.assertEqual(retrieval["reciprocal_rank_at_2"], 1.0)
        self.assertEqual(citations["unsupported_citation_count"], 1)
        self.assertEqual(citations["citation_coverage"], 0.0)
        self.assertEqual(coverage, 1.0)

    def test_evaluation_fixture_is_valid_jsonl(self) -> None:
        path = Path(__file__).parents[1] / "samples" / "evaluation.jsonl"
        examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(examples), 2)
        self.assertTrue(all(example["expected_sources"] for example in examples))

    def test_demo_evaluation_retrieves_labeled_sources(self) -> None:
        root = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            db = Path(temporary_directory) / "index.sqlite3"
            ingest_directory(root / "data" / "demo-corpus", db)
            graph = build_rag_graph(str(db))
            examples = [
                json.loads(line)
                for line in (root / "samples" / "evaluation.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            results = [evaluate_example(graph, example) for example in examples]

        self.assertGreater(results[0]["metrics"]["recall_at_5"], 0)
        self.assertGreater(results[1]["metrics"]["recall_at_5"], 0)
        self.assertEqual(results[0]["metrics"]["citation_precision"], 1)


if __name__ == "__main__":
    unittest.main()
