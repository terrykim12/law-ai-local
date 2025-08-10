from __future__ import annotations

import argparse
import statistics
import time
from typing import List

import httpx

from src.llm import LLMClient


def bench_e2e(question: str, models: List[str], runs: int, url: str) -> None:
    print(f"[E2E] server={url} question='{question}' runs={runs}")
    with httpx.Client(timeout=120.0) as client:
        for model in models:
            latencies: List[float] = []
            # warmup 1
            client.post(url, json={"question": question, "top_k": 3, "model": model})
            for _ in range(runs):
                t0 = time.perf_counter()
                r = client.post(url, json={"question": question, "top_k": 3, "model": model})
                r.raise_for_status()
                _ = r.json()
                latencies.append(time.perf_counter() - t0)
            print_metrics(model, latencies)


def bench_llm(question: str, models: List[str], runs: int) -> None:
    print(f"[LLM-only] question='{question}' runs={runs}")
    client = LLMClient("config.yaml")
    for model in models:
        latencies: List[float] = []
        # warmup 1
        client.generate(prompt=question, model_override=model)
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = client.generate(prompt=question, model_override=model)
            latencies.append(time.perf_counter() - t0)
        print_metrics(model, latencies)


def print_metrics(model: str, latencies: List[float]) -> None:
    avg = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p95 = percentile(latencies, 95)
    print(f"- model={model} avg={avg:.2f}s p50={p50:.2f}s p95={p95:.2f}s runs={len(latencies)}")


def percentile(data: List[float], p: int) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(data_sorted) - 1)
    if f == c:
        return data_sorted[int(k)]
    d0 = data_sorted[f] * (c - k)
    d1 = data_sorted[c] * (k - f)
    return d0 + d1


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 속도 벤치마크")
    parser.add_argument("--mode", choices=["e2e", "llm"], default="e2e")
    parser.add_argument("--question", default="간단한 테스트 질문입니다. 답변을 1-2문장으로 해주세요.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen2.5:7b-instruct", "qwen3:8b"],
        help="비교할 모델들(공백 구분)",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--server", default="http://localhost:8000/query")
    args = parser.parse_args()

    if args.mode == "e2e":
        bench_e2e(args.question, args.models, args.runs, args.server)
    else:
        bench_llm(args.question, args.models, args.runs)


if __name__ == "__main__":
    main()


