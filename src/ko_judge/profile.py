"""KUDGE 프로파일: GPU 없이 실제 데이터로 낼 수 있는 수치만 계산한다.

python -m ko_judge.profile   →  reports/dataset_profile.json, reports/dataset_profile.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ko_judge import data as kd
from ko_judge.metrics import chance_off_by_half, pair_agreement
from ko_judge.ranking import bootstrap_ranking, min_n_for, model_means, tau
from ko_judge.splits import make_plan, summary

N_BOOT = 1000
NS = (10, 20, 30, 40, 50, 60, 70, 87)


def _counts(s: pd.Series) -> dict:
    return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}


def build(data_dir: Path = kd.DATA_DIR) -> dict:
    pw = kd.load_pointwise(data_dir)
    pair = kd.load("pairwise", data_dir)
    pf = kd.load("pointwise_false", data_dir)
    prf = kd.load("pairwise_false", data_dir)
    raw = kd.load("human_raw", data_dir)

    per_model = pw.groupby("model").size()
    per_instr = pw.groupby("instruction").size()
    common = set.intersection(*(set(g.instruction) for _, g in pw.groupby("model")))
    diff = (pw.score1 - pw.score2).abs()

    raw_both = raw[raw.score2 != -1]
    raw_diff = (raw_both.score1 - raw_both.score2).abs()

    human = model_means(pw, "final_score")
    human_common = model_means(pw[pw.instruction.isin(common)], "final_score")
    by_s1, by_s2 = model_means(pw, "score1"), model_means(pw, "score2")
    ranking = (
        pd.DataFrame({"mean_final_score": human, "rows": per_model})
        .sort_values("mean_final_score", ascending=False, kind="stable")
        .reset_index()
        .rename(columns={"index": "model"})
    )
    ranking.insert(0, "rank", range(1, len(ranking) + 1))

    stab_full = bootstrap_ranking(pw, "final_score", ns=NS, n_boot=N_BOOT, seed=0)
    stab_annot = bootstrap_ranking(
        pw, "score1", ref_col="score2", ns=NS, n_boot=N_BOOT, seed=0, same_sample_ref=True
    )
    plan = make_plan(pw)

    return {
        "source": {
            "github": f"guijinSON/kudge@{kd.GITHUB_COMMIT}",
            "huggingface": f"HAERAE-HUB/KUDGE@{kd.HF_REVISION}",
            "sha256": {k: kd.sha256_file(data_dir / s.name) for k, s in kd.SOURCES.items()},
            "command": "make data profile",
        },
        "files": {
            "pointwise": len(pw),
            "pairwise": len(pair),
            "pointwise_falseinfo": len(pf),
            "pairwise_falseinfo": len(prf),
            "human_annotation_raw": len(raw),
        },
        "pointwise": {
            "instructions": int(pw.instruction.nunique()),
            "models": int(pw.model.nunique()),
            "rows_per_model": {"min": int(per_model.min()), "max": int(per_model.max())},
            "rows_per_instruction": {"min": int(per_instr.min()), "max": int(per_instr.max())},
            "instructions_answered_by_all_models": len(common),
            "score1": _counts(pw.score1),
            "score2": _counts(pw.score2),
            "final_score": _counts(pw.final_score),
            "final_is_mean_of_two": bool(np.allclose((pw.score1 + pw.score2) / 2, pw.final_score)),
            "abs_diff_score1_score2": _counts(diff),
        },
        "human_human": pair_agreement(pw.score1, pw.score2),
        "human_raw": {
            "rows": len(raw),
            "instructions": int(raw.instruction.nunique()),
            "models": int(raw.model.nunique()),
            "rows_with_second_score": len(raw_both),
            "exact": float((raw_diff == 0).mean()),
            "within_1": float((raw_diff <= 1).mean()),
            "diff_ge_2": float((raw_diff >= 2).mean()),
            "diff_gt_2": float((raw_diff > 2).mean()),
            "unflagged_rows": int(raw["healthy?"].isna().sum()),
            "models_not_in_pointwise": sorted(set(raw.model) - set(pw.model)),
        },
        "chance": chance_off_by_half(pw.final_score),
        "human_ranking": ranking.to_dict(orient="records"),
        "ranking_checks": {
            "tau_all_vs_common_instructions": tau(human, human_common),
            "tau_score1_vs_score2": tau(by_s1, by_s2),
        },
        "ranking_stability_vs_full": stab_full,
        "ranking_stability_annotator1_vs_annotator2": stab_annot,
        "min_n_tau_lower_ge": {str(t): min_n_for(stab_full, t) for t in (0.7, 0.8, 0.9)},
        "pairwise": {
            "instructions": int(pair.instruction.nunique()),
            "chosen_score": _counts(pair.chosen_score),
            "rejected_score": _counts(pair.rejected_score),
            "winner": _counts(pair.winner),
            "models_not_in_pointwise": sorted(
                (set(pair.chosen_model) | set(pair.rejected_model)) - set(pw.model)
            ),
        },
        "falseinfo": {
            "instructions": int(pf.instruction.nunique()),
            "instructions_in_pointwise": int(pf.instruction.isin(set(pw.instruction)).sum()),
            "type": _counts(pf["type"]),
            "original_human_score": _counts(pf.original_human_score),
            "pairwise_winner": _counts(prf.winner),
        },
        "splits": {
            "n_folds": plan.n_folds,
            "holdout_models": list(plan.holdout_models),
            "folds": summary(pw, plan),
        },
    }


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def render(p: dict) -> str:
    pw, hh, raw, ch = p["pointwise"], p["human_human"], p["human_raw"], p["chance"]
    L = [
        "# KUDGE 데이터 프로파일",
        "",
        f"`{p['source']['command']}`로 생성. 원본: GitHub `{p['source']['github']}` "
        f"(HF `{p['source']['huggingface']}`와 sha256 동일). 모델 학습·추론 없이 데이터만으로 계산한 값이다.",
        "",
        "## 규모",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        f"| pointwise 행 | {p['files']['pointwise']:,} |",
        f"| pointwise 질문 / 평가 대상 모델 | {pw['instructions']} / {pw['models']} |",
        f"| 모델당 행 (최소~최대) | {pw['rows_per_model']['min']}~{pw['rows_per_model']['max']} |",
        f"| 모든 모델이 답한 질문 | {pw['instructions_answered_by_all_models']} |",
        f"| pairwise 행 | {p['files']['pairwise']:,} |",
        f"| 거짓 정보 주입 (pointwise / pairwise) | {p['files']['pointwise_falseinfo']} / "
        f"{p['files']['pairwise_falseinfo']} (질문 {p['falseinfo']['instructions']}개) |",
        f"| 원 주석 행 (질문 {raw['instructions']}, 모델 {raw['models']}) | {raw['rows']:,} |",
        "",
        "## 사람 점수",
        "",
        "| 점수 | 1 | 2 | 3 | 4 | 5 |",
        "|---|---:|---:|---:|---:|---:|",
        "| score1 | " + " | ".join(str(pw["score1"].get(str(k), 0)) for k in range(1, 6)) + " |",
        "| score2 | " + " | ".join(str(pw["score2"].get(str(k), 0)) for k in range(1, 6)) + " |",
        "",
        "final_score 분포: "
        + ", ".join(f"{k}: {v}" for k, v in pw["final_score"].items())
        + f" (final = (score1+score2)/2: {pw['final_is_mean_of_two']})",
        "",
        "| 사람 간 일치 (pointwise, score1 대 score2) | 값 |",
        "|---|---:|",
        f"| 완전 일치 | {_pct(hh['exact'])} |",
        f"| 1점 이내 | {_pct(hh['within_1'])} |",
        f"| 피어슨 r | {hh['pearson']:.3f} |",
        "| 평가자 간 점수 차이 분포 | "
        + ", ".join(f"{k}점: {v}" for k, v in pw["abs_diff_score1_score2"].items())
        + " |",
        "",
        f"원 주석 파일(두 번째 점수가 있는 {raw['rows_with_second_score']:,}행): 완전 일치 {_pct(raw['exact'])}, "
        f"1점 이내 {_pct(raw['within_1'])}, 2점 이상 차이 {_pct(raw['diff_ge_2'])}. "
        f"pointwise 2,506행은 원 주석에서 `healthy?` 표시가 없는 {raw['unflagged_rows']:,}행과 같다.",
        "",
        "## 우연 기준선 (off-by-0.5)",
        "",
        f"- 정수 1~5 균등 추측: {_pct(ch['uniform'])}",
        f"- 가장 나은 상수 점수({ch['best_constant']}): {_pct(ch['best_constant_acc'])}",
        "",
        "## 사람 점수로 매긴 31개 모델 순위",
        "",
        "| 순위 | 모델 | 평균 final_score | 행 |",
        "|---:|---|---:|---:|",
    ]
    for r in p["human_ranking"]:
        L.append(f"| {r['rank']} | {r['model']} | {r['mean_final_score']:.3f} | {r['rows']} |")
    rc = p["ranking_checks"]
    L += [
        "",
        f"- 전체 질문 순위 대 공통 질문 {pw['instructions_answered_by_all_models']}개 순위: "
        f"Kendall τ = {rc['tau_all_vs_common_instructions']:.3f}",
        f"- 평가자 1(score1) 순위 대 평가자 2(score2) 순위: Kendall τ = {rc['tau_score1_vs_score2']:.3f}",
        "",
        f"## 순위 안정성 (질문 복원추출 부트스트랩, {p['ranking_stability_vs_full']['n_boot']}회)",
        "",
        "질문 n개로 매긴 순위와 87개 전체 순위의 Kendall τ, 그리고 같은 n개 질문에서 "
        "평가자 1 순위와 평가자 2 순위의 τ(사람끼리의 재현 상한).",
        "",
        "| 질문 수 n | τ(전체 대비) 평균 | 95% 구간 | 1위 일치 | τ(평가자1 대 평가자2) 평균 | 95% 구간 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for a, b in zip(
        p["ranking_stability_vs_full"]["results"],
        p["ranking_stability_annotator1_vs_annotator2"]["results"],
        strict=True,
    ):
        L.append(
            f"| {a['n']} | {a['tau_mean']:.3f} | {a['tau_p2_5']:.3f}~{a['tau_p97_5']:.3f} | "
            f"{_pct(a['top1'])} | {b['tau_mean']:.3f} | {b['tau_p2_5']:.3f}~{b['tau_p97_5']:.3f} |"
        )
    mn = p["min_n_tau_lower_ge"]
    L += [
        "",
        "τ 2.5 백분위가 기준 이상이 되는 최소 질문 수: "
        + ", ".join(f"τ≥{t}: {v if v is not None else '87개로도 미달'}" for t, v in mn.items()),
        "",
        "## pairwise·거짓 정보 부분집합",
        "",
        f"- pairwise chosen_score: {p['pairwise']['chosen_score']}",
        f"- pairwise rejected_score: {p['pairwise']['rejected_score']}",
        f"- pointwise에 없는 모델: {p['pairwise']['models_not_in_pointwise']}",
        f"- 거짓 정보 유형: {p['falseinfo']['type']}, 원래 사람 점수: {p['falseinfo']['original_human_score']}",
        f"- 거짓 정보 질문 중 pointwise에도 있는 질문: {p['falseinfo']['instructions_in_pointwise']}/"
        f"{p['files']['pointwise_falseinfo']}행",
        "",
        "## 분할 (질문 5-fold + 보류 모델)",
        "",
        f"보류 모델 {len(p['splits']['holdout_models'])}개: "
        + ", ".join(p["splits"]["holdout_models"]),
        "",
        "| fold | test 질문 | train 행 | test 행 | test 중 보류 모델 행 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for f in p["splits"]["folds"]:
        L.append(
            f"| {f['fold']} | {f['test_instructions']} | {f['train_rows']} | {f['test_rows']} | "
            f"{f['test_rows_holdout_models']} |"
        )
    return "\n".join(L) + "\n"


def main() -> None:
    p = build()
    Path("reports").mkdir(exist_ok=True)
    Path("reports/dataset_profile.json").write_text(
        json.dumps(p, ensure_ascii=False, indent=2) + "\n"
    )
    Path("reports/dataset_profile.md").write_text(render(p))
    print(render(p))


if __name__ == "__main__":
    main()
