import math

import pytest

from ko_judge.parsing import argmax_score, expected_score, extract_result, score_distribution


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Feedback: 좋다 [RESULT] 4", 4),
        ("[RESULT]3", 3),
        ("[RESULT] 2 ... 다시 보니 [RESULT] 5", 5),  # 논문 코드처럼 마지막 것
        ("[RESULT] 6", None),
        ("[RESULT] 0", None),
        ("점수: 4", None),
        ("", None),
        (None, None),
    ],
)
def test_extract_result(text, expected):
    assert extract_result(text) == expected


def test_score_distribution_merges_variants_and_ignores_other_tokens():
    top = {"3": math.log(0.4), " 3": math.log(0.1), "4": math.log(0.3), "A": math.log(0.2)}
    d = score_distribution(top)
    assert d[3] == pytest.approx(0.5 / 0.8)
    assert d[4] == pytest.approx(0.3 / 0.8)
    assert d[1] == 0.0
    assert sum(d.values()) == pytest.approx(1.0)
    assert score_distribution({"A": 0.0}) is None


def test_score_distribution_ignores_non_ascii_digits():
    # '①'·'²'는 str.isdigit()가 참이지만 int()가 실패한다. '٣'(아랍 숫자)은 int()가 3으로 읽는다.
    top = {"①": math.log(0.3), "²": math.log(0.1), "٣": math.log(0.1), "2": math.log(0.5)}
    assert score_distribution(top) == {1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0}
    assert score_distribution({"①": 0.0, "10": 0.0}) is None


def test_argmax_and_expected():
    d = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.3, 5: 0.1}
    assert argmax_score(d) == 3  # 동률이면 낮은 점수
    assert expected_score(d) == pytest.approx(0.1 + 0.4 + 0.9 + 1.2 + 0.5)
