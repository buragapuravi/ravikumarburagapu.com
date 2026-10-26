"""Concurrent benchmark runner for OpenAI-compatible chat completion APIs."""

import asyncio
import json
import math
import time
from collections.abc import Callable
from typing import Any

import httpx

from benchmarkkit.metrics import summarize_inference

try:
    from prometheus_client import Histogram, start_http_server
except ImportError:  # pragma: no cover - required dependency in packaged installs
    Histogram = None
    start_http_server = None

_REQUEST_LATENCY = (
    Histogram("benchmarkkit_request_latency_seconds", "Inference request latency", ["framework"])
    if Histogram
    else None
)
_TTFT = (
    Histogram("benchmarkkit_ttft_seconds", "Time to first generated token", ["framework"])
    if Histogram
    else None
)
_OUTPUT_TOKENS = (
    Histogram("benchmarkkit_output_tokens", "Generated tokens per request", ["framework"])
    if Histogram
    else None
)


async def _request(
    client: httpx.AsyncClient,
    endpoint: str,
    prompt: str,
    model: str,
    max_tokens: int,
    streaming: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": streaming,
    }
    if streaming:
        payload["stream_options"] = {"include_usage": True}

    started = time.perf_counter()
    first_token_at: float | None = None
    input_tokens = 0
    output_tokens = 0
    if streaming:
        async with client.stream("POST", endpoint, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as error:
                    raise ValueError(f"inference endpoint returned invalid SSE JSON: {data[:120]}") from error
                if chunk.get("usage"):
                    input_tokens = int(chunk["usage"].get("prompt_tokens", 0))
                    output_tokens = int(chunk["usage"].get("completion_tokens", 0))
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    first_token_at = first_token_at or time.perf_counter()
        completed = time.perf_counter()
    else:
        response = await client.post(endpoint, json=payload)
        response.raise_for_status()
        body = response.json()
        completed = time.perf_counter()
        usage = body.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        first_token_at = completed

    return {
        "latency_ms": (completed - started) * 1000,
        "ttft_ms": ((first_token_at or completed) - started) * 1000,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


async def run_inference(
    prompts: list[str],
    endpoint: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    input_price: float,
    output_price: float,
    streaming: bool = True,
    api_key: str | None = None,
    timeout_seconds: float = 120,
    metrics_port: int = 9464,
    framework: str = "openai-compatible",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not prompts:
        raise ValueError("at least one prompt is required")
    if concurrency < 1 or max_tokens < 1:
        raise ValueError("concurrency and max_tokens must be positive")
    if (
        not math.isfinite(input_price)
        or not math.isfinite(output_price)
        or input_price < 0
        or output_price < 0
    ):
        raise ValueError("token prices must be finite and non-negative")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if metrics_port < 0:
        raise ValueError("metrics_port must be non-negative")

    if metrics_port and start_http_server:
        start_http_server(metrics_port)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(timeout_seconds)
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()

    async with httpx.AsyncClient(headers=headers, limits=limits, timeout=timeout) as client:
        async def run_one(index: int) -> None:
            async with semaphore:
                result = await _request(client, endpoint, prompts[index % len(prompts)], model, max_tokens, streaming)
                result["request_id"] = index
                results.append(result)
                if _REQUEST_LATENCY:
                    _REQUEST_LATENCY.labels(framework).observe(result["latency_ms"] / 1000)
                    _TTFT.labels(framework).observe(result["ttft_ms"] / 1000)
                    _OUTPUT_TOKENS.labels(framework).observe(result["output_tokens"])
                if progress:
                    progress({"completed": len(results), "total": len(prompts)})

        await asyncio.gather(*(run_one(index) for index in range(len(prompts))))

    elapsed = time.perf_counter() - benchmark_started
    results.sort(key=lambda item: item["request_id"])
    return summarize_inference(results, elapsed, input_price, output_price)
