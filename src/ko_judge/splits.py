"""분할: 질문(instruction) 단위 K-fold + 평가 대상 모델 보류(held-out).

- 질문 87개를 K개 묶음으로 나눈다. fold k의 test는 묶음 k의 질문에 대한 모든 응답이고,
  train에는 그 질문이 한 번도 나오지 않는다.
- 평가 대상 모델 31개 중 H개를 보류한다. 보류 모델의 응답은 어느 fold의 train에도 들어가지 않는다.
  사람 점수 순위를 고르게 덮도록 순위 구간마다 하나씩 고른다.
- 두 조건을 한 번의 학습으로 함께 만족시킨다: train = (묶음 k 밖의 질문) ∩ (보류 밖의 모델).
  그래서 K번 학습한 out-of-fold 예측이 2,506행 전체를 덮고,
  보류 모델의 예측은 질문도 모델도 처음 보는 상태에서 나온 것이다.
- 분할 시드는 학습 시드와 따로 둔다. 학습 시드를 바꿔도 분할은 같다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_FOLDS = 5
N_HOLDOUT_MODELS = 8
SPLIT_SEED = 0


@dataclass(frozen=True)
class SplitPlan:
    fold_of_instruction: dict[str, int]
    holdout_models: tuple[str, ...]
    n_folds: int

    def train_mask(self, df: pd.DataFrame, fold: int) -> pd.Series:
        f = df["instruction"].map(self.fold_of_instruction)
        return (f != fold) & ~df["model"].isin(self.holdout_models)

    def test_mask(self, df: pd.DataFrame, fold: int) -> pd.Series:
        return df["instruction"].map(self.fold_of_instruction) == fold

    def fold_of(self, instruction: str) -> int:
        return self.fold_of_instruction[instruction]


def assign_folds(instructions, n_folds: int = N_FOLDS, seed: int = SPLIT_SEED) -> dict[str, int]:
    uniq = sorted(set(instructions))
    if len(uniq) < n_folds:
        raise ValueError("질문 수가 fold 수보다 적다")
    order = np.random.default_rng(seed).permutation(len(uniq))
    return {uniq[i]: int(rank % n_folds) for rank, i in enumerate(order)}


def pick_holdout_models(
    model_means: pd.Series, n: int = N_HOLDOUT_MODELS, seed: int = SPLIT_SEED
) -> tuple[str, ...]:
    """사람 평균 점수 순위를 n개 구간으로 나눠 구간마다 한 모델씩 고른다."""
    ranked = model_means.sort_values(ascending=False, kind="stable").index.tolist()
    if not 0 < n < len(ranked):
        raise ValueError("보류 모델 수가 범위를 벗어났다")
    rng = np.random.default_rng(seed)
    bins = np.array_split(np.arange(len(ranked)), n)
    return tuple(sorted(ranked[int(rng.choice(b))] for b in bins))


def make_plan(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    n_holdout: int = N_HOLDOUT_MODELS,
    seed: int = SPLIT_SEED,
) -> SplitPlan:
    means = df.groupby("model")["final_score"].mean()
    return SplitPlan(
        fold_of_instruction=assign_folds(df["instruction"], n_folds, seed),
        holdout_models=pick_holdout_models(means, n_holdout, seed),
        n_folds=n_folds,
    )


def summary(df: pd.DataFrame, plan: SplitPlan) -> list[dict]:
    rows = []
    for k in range(plan.n_folds):
        tr, te = plan.train_mask(df, k), plan.test_mask(df, k)
        rows.append(
            {
                "fold": k,
                "test_instructions": int(df.loc[te, "instruction"].nunique()),
                "train_rows": int(tr.sum()),
                "test_rows": int(te.sum()),
                "test_rows_holdout_models": int((te & df["model"].isin(plan.holdout_models)).sum()),
            }
        )
    return rows
