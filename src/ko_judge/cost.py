"""결과·건당 비용 표 생성. GPU 시간당 요금($/h)은 입력으로 받는다.

  $/100만 건 = GPU $/h ÷ (건/s × 3600) × 1,000,000

측정 파일이 없는 칸은 '측정 전'으로 둔다. 숫자를 지어내지 않는다.

  python -m ko_judge.cost --gpu-usd-per-hour 0.69 --gpu-name "RTX 4090"
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

PENDING = "측정 전"
SECONDS_PER_HOUR = 3600


def usd_per_million(gpu_usd_per_hour: float, items_per_s: float) -> float:
    if gpu_usd_per_hour < 0:
        raise ValueError("요금은 0 이상")
    if items_per_s <= 0:
        raise ValueError("처리량은 양수")
    return gpu_usd_per_hour / (items_per_s * SECONDS_PER_HOUR) * 1_000_000


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fmt(value, spec: str) -> str:
    return PENDING if value is None else format(value, spec)


COLUMNS = (
    # (헤더, 지표 경로, 서식)
    ("off-by-0.5 정확도", ("agreement", "all", "paper_acc"), ".4f"),
    ("피어슨 r", ("agreement", "all", "pearson"), ".3f"),
    ("τ (31개 모델)", ("ranking", "tau_all"), ".3f"),
    ("τ (보류 8개)", ("ranking", "tau_holdout"), ".3f"),
    ("거짓 정보 감지율", ("falseinfo", "detected_rate"), ".3f"),
)


def metric_value(m: dict | None, path: tuple[str, ...], spec: str) -> str:
    """단일 실행 파일이면 값, 시드 묶음 파일이면 '평균 (최소~최대, n)'."""
    if m is None:
        return PENDING
    if "seeds" in m:
        s = m["seeds"].get(".".join(path))
        if not s:
            return PENDING
        return (
            f"{format(s['mean'], spec)} ({format(s['min'], spec)}~{format(s['max'], spec)}, "
            f"n={s['n_seeds']})"
        )
    v = m
    for k in path:
        v = v.get(k) if isinstance(v, dict) else None
    return _fmt(v, spec)


def build_rows(report_cfg: dict, reports_dir: Path, gpu_usd_per_hour: float | None) -> list[dict]:
    rows = []
    for r in report_cfg["rows"]:
        m = _load(reports_dir / "metrics" / f"{r['metrics']}.json") if r.get("metrics") else None
        t = (
            _load(reports_dir / "throughput" / f"{r['throughput']}.json")
            if r.get("throughput")
            else None
        )
        ips = t.get("items_per_s") if t else None
        rows.append(
            {
                "name": r["name"],
                "cells": [metric_value(m, path, spec) for _, path, spec in COLUMNS],
                "items_per_s": ips,
                "p95_ms": t["latency_p95_s"] * 1000 if t and t.get("latency_p95_s") else None,
                "usd_per_million": (
                    usd_per_million(gpu_usd_per_hour, ips)
                    if ips and gpu_usd_per_hour is not None
                    else None
                ),
                "has_throughput_row": bool(r.get("throughput")),
            }
        )
    return rows


def render_markdown(rows: list[dict], gpu_usd_per_hour: float | None, gpu_name: str) -> str:
    rate = f"${gpu_usd_per_hour}/h" if gpu_usd_per_hour is not None else PENDING
    head = ["구성", *(c[0] for c in COLUMNS), "건/s", "p95 (ms)", "$/100만 건"]
    lines = [
        f"GPU: {gpu_name or PENDING} · 요금 {rate}",
        "",
        "| " + " | ".join(head) + " |",
        "|---|" + "---:|" * (len(head) - 1),
    ]
    for r in rows:
        tp = r["has_throughput_row"]
        lines.append(
            "| "
            + " | ".join(
                [
                    r["name"],
                    *r["cells"],
                    _fmt(r["items_per_s"], ".2f") if tp else "—",
                    _fmt(r["p95_ms"], ".0f") if tp else "—",
                    _fmt(r["usd_per_million"], ".2f") if tp else "—",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-usd-per-hour", type=float, default=None)
    p.add_argument("--gpu-name", default="")
    p.add_argument("--config", default="configs/report.yaml")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--out", default="reports/RESULTS.md")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = build_rows(cfg, Path(args.reports_dir), args.gpu_usd_per_hour)
    table = render_markdown(rows, args.gpu_usd_per_hour, args.gpu_name)
    header = (
        "# 결과\n\n`make report`가 reports/metrics·reports/throughput에서 생성한다. "
        "손으로 고치지 않는다.\n\n"
    )
    Path(args.out).write_text(header + table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
