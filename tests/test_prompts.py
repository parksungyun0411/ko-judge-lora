import pytest

from ko_judge.prompts import SCORE_TAIL, score_only_prompt, sft_records, zero_shot_messages

ROWS = [
    {"judge_query": "평가하라\n\n###Feedback:", "score1": 2, "score2": 3},
    {"judge_query": "평가하라 2\n\n\n###Feedback:  ", "score1": 5, "score2": 5},
]


def test_score_only_prompt_replaces_feedback_tail():
    p = score_only_prompt(ROWS[0]["judge_query"])
    assert p.endswith(SCORE_TAIL) and "###Feedback" not in p
    assert score_only_prompt("꼬리 없음").endswith(SCORE_TAIL)


def test_zero_shot_uses_query_verbatim():
    assert zero_shot_messages("q") == [{"role": "user", "content": "q"}]


def test_sft_records_targets():
    both = sft_records(ROWS, "both")
    assert [r["completion"][0]["content"] for r in both] == ["2", "3", "5", "5"]
    sample = sft_records(ROWS, "sample", seed=0)
    assert len(sample) == 2 and sample[0]["completion"][0]["content"] in {"2", "3"}
    assert sample == sft_records(ROWS, "sample", seed=0)
    with pytest.raises(ValueError):
        sft_records(ROWS, "mean")
