"""vLLM 서버 부하 측정: 동시 요청 수를 고정하고 판정 건/s와 지연 분위수를 잰다.

요청 하나 = pointwise 응답 하나에 대한 점수만 판정(max_tokens 1). 처음 --warmup건은 집계에서 뺀다.
판정 프롬프트가 평균 2천 토큰대라 처리량은 대부분 prefill이 정한다.

  python -m ko_judge.loadtest --served qwen3-4b-awq-f0 --tag lora_qwen3-4b_awq --concurrency 32
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

import httpx

from ko_judge.data import load_pointwise
from ko_judge.generation import OpenAIBackend, default_base_url
from ko_judge.prompts import score_only_messages


def percentile(values: list[float], q: float) -> float:
    """최근접 순위(nearest-rank) 분위수. q는 0~100."""
    if not values:
        raise ValueError("빈 목록")
    if not 0 < q <= 100:
        raise ValueError("q는 (0, 100]")
    s = sorted(values)
    return s[max(0, math.ceil(q / 100 * len(s)) - 1)]


def summarize(latencies: list[float], wall_s: float, errors: int) -> dict:
    n = len(latencies)
    return {
        "completed": n,
        "errors": errors,
        "wall_s": wall_s,
        "items_per_s": n / wall_s if wall_s > 0 else 0.0,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
    }


async def run(backend: OpenAIBackend, messages: list[list[dict]], concurrency: int, warmup: int):
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors = 0

    async def one(client, m, record: bool):
        nonlocal errors
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await client.post(backend.url, json=backend.score_payload(m))
                r.raise_for_status()
            except httpx.HTTPError:
                if record:
                    errors += 1
                return
            if record:
                latencies.append(time.perf_counter() - t0)

    async with httpx.AsyncClient(timeout=backend.timeout) as client:
        await asyncio.gather(*(one(client, m, False) for m in messages[:warmup]))
        t0 = time.perf_counter()
        await asyncio.gather(*(one(client, m, True) for m in messages[warmup:]))
        wall = time.perf_counter() - t0
    return summarize(latencies, wall, errors)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default=default_base_url())
    p.add_argument("--served", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=1000, help="측정 건수 (warmup 제외)")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--gpu", default="", help="기록용 GPU 이름 (예: RTX 4090)")
    p.add_argument("--out-dir", default="reports/throughput")
    args = p.parse_args()

    df = load_pointwise().sample(frac=1.0, random_state=0)  # 행 순서 영향을 없앤다
    queries = df.judge_query.tolist()[: args.warmup + args.n]
    messages = [score_only_messages(q) for q in queries]
    backend = OpenAIBackend(args.base_url, args.served)
    result = asyncio.run(run(backend, messages, args.concurrency, args.warmup))
    result.update(
        tag=args.tag,
        served=args.served,
        concurrency=args.concurrency,
        gpu=args.gpu,
        measured_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / f"{args.tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    p95_ms = result["latency_p95_s"] * 1000
    print(f"{args.tag}: {result['items_per_s']:.2f} items/s, p95 {p95_ms:.0f} ms")


if __name__ == "__main__":
    main()
