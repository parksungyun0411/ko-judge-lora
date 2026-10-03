"""판정 프롬프트.

- 제로샷(생성형): KUDGE의 judge_query를 그대로 user 메시지로 쓴다. 논문 부록 B(17쪽)의
  pointwise 프롬프트이며, 질문·응답·참조 답안(5점)·채점 기준이 이미 들어 있다.
- 점수만(LoRA): 학습 데이터에는 사람의 피드백 글이 없고 점수만 있다. 그래서 judge_query 끝의
  "###Feedback:"을 점수 칸으로 바꾸고, 정답 completion은 점수 숫자 한 글자로 둔다.
"""

from __future__ import annotations

import random

FEEDBACK_TAIL = "###Feedback:"
SCORE_TAIL = "###Score (1~5 사이 정수 하나만 쓴다):"
TARGET_MODES = ("both", "sample")


def zero_shot_messages(judge_query: str) -> list[dict]:
    return [{"role": "user", "content": judge_query}]


def score_only_prompt(judge_query: str) -> str:
    q = judge_query.rstrip()
    if q.endswith(FEEDBACK_TAIL):
        q = q[: -len(FEEDBACK_TAIL)].rstrip()
    return f"{q}\n\n{SCORE_TAIL}"


def score_only_messages(judge_query: str) -> list[dict]:
    return [{"role": "user", "content": score_only_prompt(judge_query)}]


def sft_records(rows: list[dict], target: str = "sample", seed: int = 0) -> list[dict]:
    """행마다 사람 점수를 정답으로 만든다. 평균 점수(0.5 단위) 대신 개별 평가자 점수(정수)를 쓴다.

    - both: score1·score2를 각각 한 예제로 (토큰 수 2배)
    - sample: 행마다 둘 중 하나를 시드로 고른다
    """
    if target not in TARGET_MODES:
        raise ValueError(f"target은 {TARGET_MODES} 중 하나")
    rng = random.Random(seed)
    out = []
    for r in rows:
        scores = (
            [r["score1"], r["score2"]]
            if target == "both"
            else [rng.choice([r["score1"], r["score2"]])]
        )
        for s in scores:
            out.append(
                {
                    "prompt": score_only_messages(r["judge_query"]),
                    "completion": [{"role": "assistant", "content": str(int(s))}],
                }
            )
    return out
