import json
import tempfile
import unittest
from pathlib import Path

import httpx

from benchmarkkit.datasets import load_jsonl, prepare_public_questions, validate_prompts
from benchmarkkit.inference import _request
from benchmarkkit.metrics import evaluate_rag_record, percentile, summarize_inference, summarize_rag


class MetricsTests(unittest.TestCase):
    def test_percentile_uses_nearest_rank(self) -> None:
        self.assertEqual(percentile([8, 1, 4, 2], 0.5), 2)
        self.assertEqual(percentile([], 0.99), 0)

    def test_inference_summary_includes_p99_and_cost(self) -> None:
        result = summarize_inference(
            [{"latency_ms": 20, "ttft_ms": 5, "input_tokens": 10, "output_tokens": 20}],
            elapsed_seconds=2,
            input_price=1,
            output_price=2,
        )
        self.assertEqual(result["latency_ms"]["p99"], 20)
        self.assertEqual(result["ttft_ms"]["p99"], 5)
        self.assertAlmostEqual(result["cost_usd"], 0.00005)
        self.assertAlmostEqual(result["cost_per_1k_tokens_usd"], 0.0016666667)

    def test_rag_recall_and_faithfulness_proxy(self) -> None:
        record = {
            "expected_doc_ids": ["a", "b"],
            "retrieved_doc_ids": ["b", "x", "a"],
            "answer": "The moon orbits Earth",
            "contexts": ["Earth has a natural satellite called the Moon."],
            "latency_seconds": 0.5,
        }
        metrics = evaluate_rag_record(record)
        self.assertEqual(metrics["recall"]["recall_at_1"], 0.5)
        self.assertEqual(metrics["recall"]["recall_at_3"], 1)
        self.assertAlmostEqual(metrics["faithfulness_proxy"], 2 / 3)
        self.assertEqual(summarize_rag([record])["latency_ms"]["p99"], 500)

    def test_rag_rejects_missing_relevance_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected_doc_ids"):
            evaluate_rag_record(
                {"expected_doc_ids": [], "retrieved_doc_ids": [], "answer": "", "contexts": []}
            )


class DatasetTests(unittest.TestCase):
    def test_sample_files_match_supported_formats(self) -> None:
        root = Path(__file__).parents[1]
        prompts = validate_prompts(load_jsonl(root / "samples/prompts.jsonl"))
        rag_records = load_jsonl(root / "samples/rag-evaluation.jsonl")
        self.assertTrue(prompts)
        self.assertEqual(summarize_rag(rag_records)["record_count"], 2)

    def test_prepares_hotpotqa_and_natural_questions_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            hotpot_file = directory / "hotpot.json"
            hotpot_file.write_text(
                json.dumps([{"_id": "hp-1", "question": "What is recall at k?"}]),
                encoding="utf-8",
            )
            nq_file = directory / "nq.jsonl"
            nq_file.write_text(
                '{"example_id":"nq-1","question_text":"What is TTFT?"}\n',
                encoding="utf-8",
            )

            hotpot = prepare_public_questions(hotpot_file, "hotpotqa", 10)
            natural_questions = prepare_public_questions(nq_file, "natural-questions", 10)

        self.assertEqual(hotpot[0]["prompt"], "What is recall at k?")
        self.assertEqual(hotpot[0]["source_id"], "hp-1")
        self.assertEqual(natural_questions[0]["prompt"], "What is TTFT?")
        self.assertEqual(natural_questions[0]["source_id"], "nq-1")


class InferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_request_measures_ttft_and_usage(self) -> None:
        events = [
            {"choices": [{"delta": {"content": "hello"}}]},
            {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2}},
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"

        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await _request(client, "https://example.test/chat", "prompt", "model", 12, True)

        self.assertGreaterEqual(result["ttft_ms"], 0)
        self.assertGreaterEqual(result["latency_ms"], result["ttft_ms"])
        self.assertEqual(result["input_tokens"], 4)
        self.assertEqual(result["output_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
