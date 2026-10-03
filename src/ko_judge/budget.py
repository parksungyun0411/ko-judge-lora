"""GPU 시간 추정. 토큰 수는 실제 데이터와 실제 토크나이저로 세고, 처리 속도는 가정값이다.

결과는 **추정**이며 측정이 아니다. 첫 학습 실행의 스텝 시간을 보고 --train-tps 등을 고쳐 다시 낸다.

  python -m ko_judge.budget   →  reports/gpu_budget.json, reports/gpu_budget.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ko_judge import compat
from ko_judge.config import load_models
from ko_judge.data import load_pointwise
from ko_judge.prompts import score_only_messages
from ko_judge.splits import make_plan

# 가정값 (RTX 4090 24GB 1장 기준으로 잡은 출발점. 측정 아님)
ASSUMED = {
    "qwen3-4b": {"train_tps": 2500, "prefill_tps": 15000},
    "kanana-1.5-2.1b": {"train_tps": 4500, "prefill_tps": 25000},
}
ASSUMED_COMMON = {
    "zero_shot_output_tokens": 300,  # 피드백 + [RESULT]
    "decode_tps_batched": 1500,
    "merge_awq_hours_per_fold": 0.25,
    "server_start_hours": 0.05,
    "loadtest_hours_per_variant": 0.1,
    "setup_hours": 1.0,
    "slack_ratio": 0.2,
}


def count_tokens(model_key: str, queries: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer

    compat.apply()
    spec = load_models()[model_key]
    tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision)
    texts = [
        tok.apply_chat_template(score_only_messages(q), tokenize=False, add_generation_prompt=True)
        for q in queries
    ]
    return np.array([len(tok(t, add_special_tokens=False).input_ids) for t in texts])


def estimate(
    tokens: np.ndarray,
    train_rows_per_fold: list[np.ndarray],
    a: dict,
    folds: int,
    seeds: int,
    epochs: float,
    examples_per_row: int,
) -> dict:
    c = ASSUMED_COMMON
    train_tok = np.mean([tokens[idx].sum() for idx in train_rows_per_fold]) * examples_per_row
    train_h_run = epochs * train_tok / a["train_tps"] / 3600
    full_prefill_h = tokens.sum() / a["prefill_tps"] / 3600
    zs_h = (
        full_prefill_h + len(tokens) * c["zero_shot_output_tokens"] / c["decode_tps_batched"] / 3600
    )
    parts = {
        "train": train_h_run * folds * seeds,
        "eval_lora_bf16": seeds * full_prefill_h + seeds * c["server_start_hours"],
        "zero_shot": zs_h + c["server_start_hours"],
        "merge_awq": folds * c["merge_awq_hours_per_fold"],
        "eval_merged_and_awq": 2 * (full_prefill_h + folds * c["server_start_hours"]),
        "loadtest": 2 * c["loadtest_hours_per_variant"],
    }
    subtotal = sum(parts.values())
    return {
        "train_tokens_per_run": float(train_tok * epochs),
        "train_hours_per_run": train_h_run,
        "parts_hours": parts,
        "subtotal_hours": subtotal,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--examples-per-row", type=int, default=1, help="target=sample이면 1, both면 2")
    args = p.parse_args()

    df = load_pointwise()
    plan = make_plan(df)
    folds = plan.n_folds
    idx = [np.flatnonzero(plan.train_mask(df, k).values) for k in range(folds)]
    out = {"assumed": {"per_model": ASSUMED, "common": ASSUMED_COMMON}, "models": {}}
    total = ASSUMED_COMMON["setup_hours"]
    for key in ("qwen3-4b", "kanana-1.5-2.1b"):
        tok = count_tokens(key, df.judge_query.tolist())
        est = estimate(
            tok, idx, ASSUMED[key], folds, args.seeds, args.epochs, args.examples_per_row
        )
        out["models"][key] = {
            "prompt_tokens": {
                "sum": int(tok.sum()),
                "mean": float(tok.mean()),
                "p95": float(np.percentile(tok, 95)),
                "max": int(tok.max()),
            },
            **est,
        }
        total += est["subtotal_hours"]
    out["total_hours_with_slack"] = total * (1 + ASSUMED_COMMON["slack_ratio"])
    q = out["models"]["qwen3-4b"]["subtotal_hours"]
    out["fallback_qwen3_only_hours_with_slack"] = (q + ASSUMED_COMMON["setup_hours"]) * (
        1 + ASSUMED_COMMON["slack_ratio"]
    )
    out["settings"] = vars(args) | {"folds": folds}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/gpu_budget.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    Path("reports/gpu_budget.md").write_text(render(out))
    print(render(out))


def render(o: dict) -> str:
    s = o["settings"]
    L = [
        "# GPU 시간 추정 (측정 아님)",
        "",
        "`make budget`로 생성. 프롬프트 토큰 수는 실제 데이터(2,506행)를 각 모델의 실제 토크나이저와 "
        "채팅 템플릿으로 센 값이다. 처리 속도는 가정값이라 시간은 추정이다.",
        "",
        f"설정: fold {s['folds']} × 시드 {s['seeds']}, epoch {s['epochs']}, 행당 학습 예제 {s['examples_per_row']}",
        "",
        "| 모델 | 프롬프트 토큰 합 | 평균 | p95 | 최대 | 학습 1회 토큰 | 가정 학습 속도 (tok/s) | 학습 1회 (h) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, m in o["models"].items():
        t = m["prompt_tokens"]
        L.append(
            f"| {k} | {t['sum']:,} | {t['mean']:.0f} | {t['p95']:.0f} | {t['max']:,} | "
            f"{m['train_tokens_per_run']:,.0f} | {o['assumed']['per_model'][k]['train_tps']:,} | "
            f"{m['train_hours_per_run']:.2f} |"
        )
    L += ["", "| 모델 | 단계 | 추정 시간 (h) |", "|---|---|---:|"]
    for k, m in o["models"].items():
        for part, h in m["parts_hours"].items():
            L.append(f"| {k} | {part} | {h:.2f} |")
        L.append(f"| {k} | **소계** | **{m['subtotal_hours']:.2f}** |")
    L += [
        "",
        f"- 두 모델 전체 (준비 {o['assumed']['common']['setup_hours']}h, 여유 "
        f"{int(o['assumed']['common']['slack_ratio'] * 100)}% 포함): **약 {o['total_hours_with_slack']:.1f}h**",
        f"- 시간이 모자랄 때 Qwen3-4B 하나만: **약 {o['fallback_qwen3_only_hours_with_slack']:.1f}h**",
        "",
        "가정값: " + json.dumps(o["assumed"], ensure_ascii=False),
    ]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
