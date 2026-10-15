import pytest

from ko_judge.metrics import (
    chance_off_by_half,
    judge_vs_annotators,
    off_by_half,
    pair_agreement,
    pearson,
    seed_summary,
    wilson_interval,
)


def test_off_by_half_matches_paper_definition():
    # 사람 평균 [1, 1.5, 3, 4.5, 2], 판정 [1, 2, 4, None, 5]
    # |차이| < 1: 0 ✓, 0.5 ✓, 1 ✗, (실패), 3 ✗
    m = off_by_half([1, 2, 4, None, 5], [1, 1.5, 3, 4.5, 2])
    assert m["failures"] == 1
    assert m["paper_acc"] == pytest.approx(2 / 4)  # 논문 코드: 실패는 분모에서 뺀다
    assert m["strict_acc"] == pytest.approx(2 / 5)


def test_off_by_half_errors():
    with pytest.raises(ValueError):
        off_by_half([1], [1, 2])


def test_pearson_skips_failures_and_degenerate():
    assert pearson([1, 2, None, 3], [1, 2, 5, 3]) == pytest.approx(1.0)
    assert pearson([2, 2, 2], [1, 2, 3]) is None


def test_pair_agreement_hand_computed():
    a = pair_agreement([1, 2, 3, 5], [1, 3, 5, 5])
    assert a["exact"] == 0.5 and a["within_1"] == 0.75


def test_judge_vs_annotators():
    # 판정 [2, 4, None], s1 [2, 3, 1], s2 [3, 5, 1]
    # s1: exact 1/3, within1 2/3 · s2: exact 0, within1 2/3
    r = judge_vs_annotators([2, 4, None], [2, 3, 1], [3, 5, 1])
    assert r["exact"] == pytest.approx((1 / 3 + 0) / 2)
    assert r["within_1"] == pytest.approx(2 / 3)
    assert r["pearson"] is None  # 형식 실패를 빼면 2행뿐이라 계산하지 않는다


def test_judge_vs_annotators_pearson_is_mean_of_per_annotator_r():
    pred, s1, s2 = [1, 2, 3, 4, 5], [1, 2, 4, 4, 5], [2, 2, 3, 5, 4]
    r = judge_vs_annotators(pred, s1, s2)
    assert r["pearson"] == pytest.approx((pearson(pred, s1) + pearson(pred, s2)) / 2)


def test_chance_hand_computed():
    # final [1, 1.5, 5]: k=1 → 2/3, k=2 → 1/3, k=3 → 0, k=4 → 0, k=5 → 1/3
    c = chance_off_by_half([1, 1.5, 5])
    assert c["uniform"] == pytest.approx((2 / 3 + 1 / 3 + 0 + 0 + 1 / 3) / 5)
    assert c["best_constant"] == 1


def test_seed_summary_hand_computed():
    s = seed_summary([0.5, 0.6, 0.7])
    assert s["mean"] == pytest.approx(0.6) and s["sd"] == pytest.approx(0.1)
    half = 4.302652729911275 * 0.1 / 3**0.5  # t(0.975, df=2)
    assert s["ci95"] == pytest.approx([0.6 - half, 0.6 + half])
    assert "ci95" not in seed_summary([0.5])


def test_wilson_known_value():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
