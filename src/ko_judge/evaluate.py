"""out-of-fold 예측 → 지표.

  python -m ko_judge.evaluate run --tag lora_qwen3-4b_bf16_s0
  python -m ko_judge.evaluate seeds --group lora_qwen3-4b_bf16 --tags lora_qwen3-4b_bf16_s0 ...

지표
- 사람 점수 일치: 논문 정의 off-by-0.5(형식 실패 제외 / 포함 두 가지), 피어슨,
  평가자 개인 점수와의 완전 일치·1점 이내·피어슨(사람 간 일치와 같은 척도).
- 순위 재현: 판정 점수 평균으로 매긴 모델 순위와 사람 순위의 Kendall τ.
  전체 31개, 보류 모델 8개(학습에서 한 번도 못 본 모델), 학습에 쓰인 모델 23개 따로.
- 순위 안정성: 같은 질문 n개 표본에서 판정 순위 대 사람 순위 τ (부트스트랩).
- 거짓 정보: 오염된 응답 점수가 원래 응답 점수보다 낮으면 '감지'. 동점은 감지가 아니다.
  논문 7절은 피드백 글을 사람이 읽어 판단했고 점수 비교는 하지 않았으므로 이 정의는 이
  프로젝트의 것이다.

판정 점수의 해상도
- 위 지표는 모두 정수 판정으로 낸다. 제로샷(생성형)은 [RESULT] 정수, LoRA(점수만 모드)는 점수
  분포의 argmax다. 제로샷 대 LoRA를 같은 해상도로 비교하기 위해서다. 결과 표는 이 값을 쓴다.
- LoRA처럼 점수 분포가 있으면 기댓값(연속 점수)으로 낸 피어슨·순위 τ·거짓 정보 감지율을
  `expected`에 따로 남긴다. 연속 점수는 동점이 거의 없어, 구분 능력이 없는 판정도 거짓 정보
  감지율이 약 50%가 나온다. 정수 판정 쪽에는 동점 비율(`tie_rate`)을 함께 남긴다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ko_judge import data as kd
from ko_judge.metrics import (
    judge_vs_annotators,
    off_by_half,
    pair_agreement,
    pearson,
    seed_summary,
    wilson_interval,
)
from ko_judge.predict import PRED_DIR, falseinfo_items
from ko_judge.ranking import bootstrap_ranking, model_means, tau
from ko_judge.splits import make_plan

METRICS_DIR = Path("reports/metrics")
SEED_KEYS = (
    ("agreement", "all", "paper_acc"),
    ("agreement", "all", "strict_acc"),
    ("agreement", "all", "pearson"),
    ("agreement", "all", "vs_annotators", "pearson"),
    ("agreement", "holdout_models", "paper_acc"),
    ("ranking", "tau_all"),
    ("ranking", "tau_holdout"),
    ("ranking", "tau_seen"),
    ("falseinfo", "detected_rate"),
    ("falseinfo", "tie_rate"),
    ("expected", "pearson"),
    ("expected", "ranking", "tau_all"),
    ("expected", "ranking", "tau_holdout"),
    ("expected", "falseinfo", "detected_rate"),
)


def load_predictions(tag: str, pred_dir: Path = PRED_DIR) -> pd.DataFrame:
    files = sorted((pred_dir / tag).glob("fold*.jsonl"))
    if not files:
        raise FileNotFoundError(f"{pred_dir / tag}에 예측이 없다")
    rows = [json.loads(line) for f in files for line in f.open(encoding="utf-8") if line.strip()]
    return pd.DataFrame(rows)


def _values(col: pd.Series) -> list[float | None]:
    return [None if pd.isna(v) else float(v) for v in col]


def _agreement(sub: pd.DataFrame) -> dict:
    """정수 판정(score)으로 잰 사람 점수 일치."""
    score = [None if pd.isna(v) else int(v) for v in sub["score"]]
    return {
        **off_by_half(score, sub["final_score"]),
        "pearson": pearson(score, sub["final_score"].tolist()),
        "vs_annotators": judge_vs_annotators(score, sub["score1"], sub["score2"]),
    }


def _ranking(
    df: pd.DataFrame, col: str, human: pd.Series, held: list[str], seen: list[str]
) -> dict:
    judge = model_means(df[df[col].notna()], col)
    return {
        "tau_all": tau(judge, human),
        "tau_holdout": tau(judge[judge.index.isin(held)], human[human.index.isin(held)]),
        "tau_seen": tau(judge[judge.index.isin(seen)], human[human.index.isin(seen)]),
        "judge_means": judge.to_dict(),
    }


def _falseinfo(fi: pd.DataFrame, f: pd.DataFrame, df: pd.DataFrame, col: str) -> dict:
    """오염된 응답과 원래 응답을 같은 판정 점수(col)로 비교한다."""
    fi = fi.copy()
    fi["corrupted"] = fi.false_id.map(f[col])
    fi["original"] = fi.original_row_id.map(df.set_index("row_id")[col])
    valid = fi.dropna(subset=["corrupted", "original"])
    detected = valid.corrupted < valid.original
    lo, hi = wilson_interval(int(detected.sum()), len(valid)) if len(valid) else (None, None)
    return {
        "n": len(fi),
        "scored": len(valid),
        "detected_rate": float(detected.mean()) if len(valid) else None,
        "detected_ci": [lo, hi],
        "tie_rate": float((valid.corrupted == valid.original).mean()) if len(valid) else None,
        "by_type": {t: float((g.corrupted < g.original).mean()) for t, g in valid.groupby("type")},
    }


def evaluate(
    preds: pd.DataFrame, pw: pd.DataFrame, plan, n_boot: int = 1000, only_fold: int | None = None
) -> dict:
    """only_fold를 주면 그 fold의 test 행과 그 fold 질문의 거짓 정보 응답만으로 잰다
    (축소안에서 fold 0의 시드 간 변동을 볼 때)."""
    full_pw = pw
    if only_fold is not None:
        pw = pw[plan.test_mask(pw, only_fold)]
    p = preds[preds.kind == "pointwise"].set_index("id")
    missing = set(pw.row_id) - set(p.index)
    if missing:
        raise ValueError(f"예측이 없는 pointwise 행 {len(missing)}개 — 모든 fold를 돌려야 한다")
    df = pw.copy()
    df["score"] = df.row_id.map(p["score"]).astype(float)
    # 점수만 모드(LoRA)는 점수 분포의 기댓값도 있다. 생성형(제로샷)은 None
    df["expected"] = df.row_id.map(p["expected"]).astype(float) if "expected" in p else np.nan
    has_expected = bool(df["expected"].notna().any())
    hold = df.model.isin(plan.holdout_models)

    human = model_means(df, "final_score")
    held = list(plan.holdout_models)
    seen = sorted(set(human.index) - set(held))

    fi = falseinfo_items(full_pw)
    if only_fold is not None:
        fi = fi[fi.instruction.map(plan.fold_of_instruction) == only_fold]
    f = preds[preds.kind == "falseinfo"].set_index("id")
    f = f.assign(
        score=f["score"].astype(float),
        expected=f["expected"].astype(float) if "expected" in f else np.nan,
    )

    out = {
        "only_fold": only_fold,
        "n_rows": len(df),
        "judge_score": "integer",
        "agreement": {
            "all": _agreement(df),
            "holdout_models": _agreement(df[hold]),
            "seen_models": _agreement(df[~hold]),
            "human_reference": pair_agreement(df.score1, df.score2),
        },
        "ranking": {
            **_ranking(df, "score", human, held, seen),
            "stability": bootstrap_ranking(
                df[df.score.notna()],
                "score",
                ref_col="final_score",
                n_boot=n_boot,
                same_sample_ref=True,
            ),
        },
        "falseinfo": _falseinfo(fi, f, df, "score"),
        "expected": None,
    }
    if has_expected:
        out["expected"] = {
            "pearson": pearson(_values(df["expected"]), df["final_score"].tolist()),
            "ranking": _ranking(df, "expected", human, held, seen),
            "falseinfo": _falseinfo(fi, f, df, "expected"),
        }
    return out


def _get(d: dict | None, path: tuple):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def summarize_seeds(results: list[dict]) -> dict:
    out = {}
    for path in SEED_KEYS:
        vals = [_get(r, path) for r in results]
        vals = [v for v in vals if v is not None]
        out[".".join(path)] = seed_summary(vals) if vals else None
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--only-fold", type=int, default=None, help="이 fold의 test만 평가")
    s = sub.add_parser("seeds")
    s.add_argument("--group", required=True)
    s.add_argument("--tags", nargs="+", required=True)
    args = p.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "run":
        pw = kd.load_pointwise()
        res = {
            "tag": args.tag,
            **evaluate(load_predictions(args.tag), pw, make_plan(pw), args.n_boot, args.only_fold),
        }
        name = args.tag if args.only_fold is None else f"{args.tag}_fold{args.only_fold}"
        (METRICS_DIR / f"{name}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2, default=float) + "\n"
        )
        a = res["agreement"]["all"]
        print(f"{args.tag}: acc={a['paper_acc']:.4f} τ={res['ranking']['tau_all']:.3f}")
    else:
        results = [json.loads((METRICS_DIR / f"{t}.json").read_text()) for t in args.tags]
        if len(results) < 3:
            print(f"주의: 시드 {len(results)}개 — 계획은 3개 이상")
        out = {"group": args.group, "tags": args.tags, "seeds": summarize_seeds(results)}
        (METRICS_DIR / f"{args.group}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(out["seeds"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
