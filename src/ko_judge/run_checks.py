"""저장소에 넣은 예측·실행 기록에서 결과 표 밖의 비교 수치를 다시 낸다.

  python -m ko_judge.run_checks   # → reports/run_checks.json

입력은 모두 저장소에 있다(KUDGE만 `make data`로 받는다).
- outputs/predictions/{tag}/fold*.jsonl: 판정 점수·점수 분포뿐이고 응답 본문은 없다
- outputs/lora/kanana-1.5-2.1b/fold*_seed*/train_summary.json: 학습 손실·시간
- docs/kaggle_t4/vllm_*_f0.log: 부하 측정한 fold 0 서버의 vLLM 로그 발췌
- reports/metrics/*.json: 모델별 판정 평균(보류 모델 쌍 비교)
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ko_judge import data as kd
from ko_judge.predict import PRED_DIR
from ko_judge.ranking import model_means
from ko_judge.splits import make_plan

MODEL = "kanana-1.5-2.1b"
ADAPTER = {s: f"lora_{MODEL}_fp16_s{s}" for s in (0, 1, 2)}
MERGED = f"lora_{MODEL}_merged_s0"
AWQ = f"lora_{MODEL}_awq_s0"
LORA_DIR = Path("outputs/lora") / MODEL
VLLM_LOGS = {
    "merged": Path("docs/kaggle_t4/vllm_merged_f0.log"),
    "awq": Path("docs/kaggle_t4/vllm_awq_f0.log"),
}
METRICS_DIR = Path("reports/metrics")
OUT = Path("reports/run_checks.json")


def load_fold(tag: str, fold: int, pred_dir: Path = PRED_DIR) -> pd.Series:
    """(kind, id) → 정수 점수. 거짓 정보 응답 행도 들어 있다."""
    with (pred_dir / tag / f"fold{fold}.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows).set_index(["kind", "id"])["score"]


def same_rate_by_fold(tag_a: str, tag_b: str, pred_dir: Path = PRED_DIR) -> dict[int, dict]:
    """fold마다 두 예측의 정수 점수가 같은 비율(거짓 정보 응답 포함)."""
    out = {}
    for k in range(5):
        a, b = load_fold(tag_a, k, pred_dir), load_fold(tag_b, k, pred_dir)
        if not a.index.equals(b.index):
            raise ValueError(f"fold{k}: {tag_a}와 {tag_b}의 행이 다르다")
        out[k] = {"n": len(a), "same": float((a == b).mean())}
    return out


def awq_vs_merged(pw: pd.DataFrame, pred_dir: Path = PRED_DIR) -> dict:
    """pointwise 2,506행에서 AWQ와 병합 모델의 정수 점수 차이."""

    def pointwise(tag):
        s = pd.concat([load_fold(tag, k, pred_dir) for k in range(5)])
        return s.loc["pointwise"].sort_index()

    m, a = pointwise(MERGED), pointwise(AWQ)
    df = pw.set_index("row_id").loc[m.index, ["model", "final_score"]].assign(merged=m, awq=a)
    drop = df.groupby("model")[["merged", "awq"]].mean().pipe(lambda g: g.merged - g.awq)
    return {
        "n": len(df),
        "differ": int((m != a).sum()),
        "awq_lower": int((a < m).sum()),
        "mean_merged": float(m.mean()),
        "mean_awq": float(a.mean()),
        "mean_human": float(df.final_score.mean()),
        "models_lower": int((drop > 0).sum()),
        "models_same": int((drop == 0).sum()),
        "models_higher": int((drop < 0).sum()),
        "model_drop_range": [float(drop.min()), float(drop.max())],
    }


def holdout_discordant_pairs(pw: pd.DataFrame, tag: str, metrics_dir: Path = METRICS_DIR) -> dict:
    """보류 모델 쌍 중 판정 평균의 순서가 사람 평균과 반대인 쌍."""
    judge = json.loads((metrics_dir / f"{tag}.json").read_text())["ranking"]["judge_means"]
    human = model_means(pw, "final_score")
    pairs = list(itertools.combinations(make_plan(pw).holdout_models, 2))
    flipped = [(x, y) for x, y in pairs if (judge[x] - judge[y]) * (human[x] - human[y]) < 0]
    return {"pairs": len(pairs), "discordant": len(flipped)}


def train_summaries(lora_dir: Path = LORA_DIR) -> dict:
    files = sorted(lora_dir.glob("fold*_seed*/train_summary.json"))
    m = [json.loads(f.read_text())["metrics"] for f in files]
    loss = [x["train_loss"] for x in m]
    hours = [x["train_runtime"] / 3600 for x in m]
    return {
        "runs": len(m),
        "train_loss_range": [min(loss), max(loss)],
        "train_hours_range": [min(hours), max(hours)],
    }


PROMPT_TPS = re.compile(r"Avg prompt throughput: ([\d.]+) tokens/s.*Running: (\d+) reqs")
KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
WEIGHTS_GIB = re.compile(r"Model loading took ([\d.]+) GiB")


def vllm_log(path: Path) -> dict:
    """프롬프트 처리량은 0이 아닌 10초 간격 기록 전부(점수 판정 + 부하 측정 구간)."""
    text = path.read_text(encoding="utf-8")
    hits = [(float(t), int(r)) for t, r in PROMPT_TPS.findall(text) if float(t) > 0]
    tps = np.array([t for t, _ in hits])
    running = pd.Series([r for _, r in hits]).value_counts().sort_index()
    return {
        "weights_gib": float(WEIGHTS_GIB.search(text).group(1)),
        "kv_cache_tokens": int(KV_TOKENS.search(text).group(1).replace(",", "")),
        "attention_backend": "TRITON_ATTN" if "Using TRITON_ATTN" in text else None,
        "marlin_kernel": "MarlinLinearKernel" in text,
        "prompt_tps_intervals": len(tps),
        "prompt_tps_median": float(np.median(tps)),
        "prompt_tps_p10_p90": [float(x) for x in np.percentile(tps, [10, 90])],
        "running_reqs_counts": {int(k): int(v) for k, v in running.items()},
    }


def collect() -> dict:
    pw = kd.load_pointwise()
    seed_pairs = {
        f"s{a}_vs_s{b}": same_rate_by_fold(ADAPTER[a], ADAPTER[b])
        for a, b in itertools.combinations(ADAPTER, 2)
    }
    return {
        "adapter_s0_vs_merged": same_rate_by_fold(ADAPTER[0], MERGED),
        "adapter_seed_pairs": seed_pairs,
        "awq_vs_merged_by_fold": same_rate_by_fold(AWQ, MERGED),
        "awq_vs_merged": awq_vs_merged(pw),
        "holdout_discordant_pairs": {t: holdout_discordant_pairs(pw, t) for t in (MERGED, AWQ)},
        "train": train_summaries(),
        "vllm_f0": {k: vllm_log(p) for k, p in VLLM_LOGS.items()},
    }


def main() -> None:
    out = collect()
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
