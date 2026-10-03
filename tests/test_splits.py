import pytest
from conftest import needs_data

from ko_judge import data as kd
from ko_judge.splits import assign_folds, make_plan, pick_holdout_models, summary


def _check_leak_free(df, plan):
    seen_in_test = []
    for k in range(plan.n_folds):
        tr, te = df[plan.train_mask(df, k)], df[plan.test_mask(df, k)]
        assert set(tr.instruction).isdisjoint(te.instruction), "질문이 train·test에 함께 있다"
        assert set(tr.model).isdisjoint(plan.holdout_models), "보류 모델이 train에 있다"
        seen_in_test.extend(te.row_id)
    assert sorted(seen_in_test) == sorted(df.row_id), "모든 행이 정확히 한 번 test에 나와야 한다"


def test_leak_free_synthetic(pw):
    plan = make_plan(pw, n_folds=5, n_holdout=3)
    _check_leak_free(pw, plan)


def test_folds_balanced_and_deterministic(pw):
    a = assign_folds(pw.instruction, 5, seed=0)
    assert a == assign_folds(pw.instruction.sample(frac=1, random_state=1), 5, seed=0)
    sizes = [list(a.values()).count(k) for k in range(5)]
    assert max(sizes) - min(sizes) <= 1
    assert a != assign_folds(pw.instruction, 5, seed=1)
    with pytest.raises(ValueError):
        assign_folds(["a", "b"], 5)


def test_holdout_spans_ranking(pw):
    means = pw.groupby("model").final_score.mean()
    held = pick_holdout_models(means, 3, seed=0)
    ranked = means.sort_values(ascending=False).index.tolist()
    pos = sorted(ranked.index(m) for m in held)
    assert pos[0] < 4 and 4 <= pos[1] < 8 and pos[2] >= 8  # 12개를 3구간으로
    assert held == pick_holdout_models(means, 3, seed=0)
    with pytest.raises(ValueError):
        pick_holdout_models(means, 12)


@needs_data
def test_leak_free_real_kudge():
    df = kd.load_pointwise()
    plan = make_plan(df)
    _check_leak_free(df, plan)
    assert len(plan.holdout_models) == 8
    rows = summary(df, plan)
    assert sum(r["test_rows"] for r in rows) == 2506
    assert sum(r["test_instructions"] for r in rows) == 87
