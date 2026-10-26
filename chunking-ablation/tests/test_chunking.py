import unittest
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from chunking_ablation.benchmark import load_documents, load_questions, run_benchmark
from chunking_ablation.chunkers import (
    Chunk,
    chunk_documents,
    materialize_context,
)
from chunking_ablation.metrics import context_token_cost, exact_answer_accuracy, recall_at_k


class ChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = {
            "alpha": (
                "The service uses exponential backoff after a retry. "
                "Jitter spreads retry times across clients. "
                "A circuit breaker stops calls after repeated failures."
            ),
            "beta": (
                "Dense retrieval compares vector representations of queries and passages. "
                "Cosine similarity compares the angle between two vectors."
            ),
        }
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(self.documents.values())

    def test_chunkers_preserve_each_sentence_as_retrievable_evidence(self) -> None:
        chunks = chunk_documents(
            self.documents, self.vectorizer, target_tokens=10, overlap_tokens=2
        )
        self.assertEqual(set(chunks), {"fixed", "recursive", "sentence-window", "semantic"})
        for strategy_chunks in chunks.values():
            self.assertTrue(strategy_chunks)
        sentence_chunk = chunks["sentence-window"][0]
        context, _ = materialize_context(
            sentence_chunk,
            {"alpha": [s.strip() for s in self.documents["alpha"].split(".") if s.strip()]},
        )
        self.assertIn("Jitter", context)

    def test_metrics_validate_k_and_price(self) -> None:
        chunk = Chunk("a", "alpha", "Jitter spreads retry times across clients.", 0, 1, "fixed")
        self.assertEqual(recall_at_k([chunk], chunk.text), 1)
        self.assertEqual(exact_answer_accuracy("A right answer.", "A right answer."), 1)
        cost = context_token_cost([("question", 100)], input_cost_per_million=2)
        self.assertEqual(cost["avg_context_tokens"], 100)
        self.assertAlmostEqual(cost["avg_estimated_cost_per_query"], 0.0002)
        with self.assertRaises(ValueError):
            recall_at_k([chunk], chunk.text, k=0)

    def test_benchmark_has_four_runs_and_pending_human_validation(self) -> None:
        qa = {
            "id": "a1",
            "document_id": "alpha",
            "question": "What spreads retry times?",
            "expected_answer": "Jitter spreads retry times across clients.",
            "evidence": "Jitter spreads retry times across clients.",
        }
        result = run_benchmark(
            self.documents,
            [qa],
            target_tokens=12,
            overlap_tokens=2,
        )
        self.assertEqual(
            [run["strategy"] for run in result["runs"]],
            ["fixed", "recursive", "sentence-window", "semantic"],
        )
        for run in result["runs"]:
            self.assertEqual(len(run["queries"]), 1)
            self.assertEqual(run["queries"][0]["human_review"], "pending")
            self.assertIn("avg_context_tokens", run)

    def test_demo_question_fixture_has_fifty_examples(self) -> None:
        questions = load_questions(Path("samples/qa.jsonl"))
        documents = load_documents(Path("samples/corpus"))
        self.assertEqual(len(questions), 50)
        for question in questions:
            self.assertIn(question["document_id"], documents)
            self.assertIn(question["evidence"], documents[question["document_id"]])


if __name__ == "__main__":
    unittest.main()
