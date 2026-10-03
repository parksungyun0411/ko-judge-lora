"""평가 대상 모델 순위와 그 안정성.

순위 점수 = 모델별 점수 평균(그 모델이 답한 질문만). 모델마다 답한 질문 수가 72~87개로
다르므로(전 모델 공통 질문은 33개) 평균은 질문 구성의 영향을 받는다. 그래서 같은 계산을
사람 점수와 판정 점수에 똑같이 적용하고 둘을 비교한다.

부트스트랩은 rag-eval-harness의 stability.ranking_stability와 같은 발상(질문을 복원추출해
"평가 질문이 n개였다면")이다. 그 함수는 모든 설정이 모든 질문에 결과를 가진 완전 행렬을
전제하고, 31개 모델의 '순위 완전 일치'는 사실상 0이라 쓸모가 없어 Kendall τ 기준으로 새로 썼다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ko_judge.metrics import wilson_interval


def model_means(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("model")[col].mean().sort_index()


def tau(a: pd.Series, b: pd.Series) -> float:
    """공통 모델에 대한 Kendall τ-b."""
    idx = a.index.intersection(b.index)
    if len(idx) < 3:
        raise ValueError("공통 모델이 3개 미만")
    return float(kendalltau(a[idx].values, b[idx].values).statistic)


def _sample_means(groups: dict[str, pd.DataFrame], picks, col: str) -> pd.Series:
    sample = pd.concat([groups[q] for q in picks], ignore_index=True)
    return model_means(sample, col)


def bootstrap_ranking(
    df: pd.DataFrame,
    col: str,
    ref_col: str | None = None,
    ns: Sequence[int] = (10, 20, 30, 40, 50, 60, 70, 87),
    n_boot: int = 1000,
    seed: int = 0,
    same_sample_ref: bool = False,
) -> dict:
    """질문 n개를 복원추출해 `col` 기준 모델 순위를 만들고 기준 순위와 τ를 잰다.

    - same_sample_ref=False: 기준 = 전체 데이터의 `ref_col`(없으면 `col`) 순위.
      "질문 n개만으로 매긴 순위가 전체 87개 순위와 얼마나 같은가".
    - same_sample_ref=True: 기준 = 같은 표본의 `ref_col` 순위.
      판정 모델이 같은 질문들 위에서 사람 순위를 얼마나 재현하는가.
    """
    ref_col = ref_col or col
    groups = {q: g for q, g in df.groupby("instruction")}
    qs = sorted(groups)
    full_ref = model_means(df, ref_col)
    top_model = full_ref.idxmax()
    rng = np.random.default_rng(seed)
    results = []
    for n in ns:
        taus, top1, coverage = [], 0, []
        for _ in range(n_boot):
            picks = rng.choice(qs, size=n, replace=True)
            m = _sample_means(groups, picks, col)
            ref = _sample_means(groups, picks, ref_col) if same_sample_ref else full_ref
            taus.append(tau(m, ref))
            top1 += int(m.idxmax() == (ref.idxmax() if same_sample_ref else top_model))
            coverage.append(len(m) / len(full_ref))
        taus = np.array(taus)
        lo, hi = wilson_interval(top1, n_boot)
        results.append(
            {
                "n": int(n),
                "tau_mean": float(taus.mean()),
                "tau_p2_5": float(np.quantile(taus, 0.025)),
                "tau_p97_5": float(np.quantile(taus, 0.975)),
                "top1": top1 / n_boot,
                "top1_ci": [lo, hi],
                "models_covered_mean": float(np.mean(coverage)),
            }
        )
    return {
        "col": col,
        "ref_col": ref_col,
        "same_sample_ref": same_sample_ref,
        "n_boot": n_boot,
        "seed": seed,
        "results": results,
    }


def min_n_for(result: dict, target: float = 0.9) -> int | None:
    """τ 2.5 백분위가 target 이상이 되는 가장 작은 n. 없으면 None."""
    for r in sorted(result["results"], key=lambda r: r["n"]):
        if r["tau_p2_5"] >= target:
            return r["n"]
    return None
