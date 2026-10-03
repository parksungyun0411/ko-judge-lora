"""판정문 → 점수.

- 생성형(제로샷): 논문 저장소 src/data_processing.py의 extract_pointwise와 같은 규칙.
  `[RESULT]` 뒤 숫자 한 자리의 마지막 출현을 쓴다. 없으면 None.
- 점수만(LoRA): 첫 생성 토큰의 상위 logprob에서 '1'~'5'만 모아 정규화한다.
  argmax는 정수 점수, 기댓값은 연속 점수(상관·순위 계산용).
"""

from __future__ import annotations

import math
import re

_RESULT = re.compile(r"\[RESULT\]\s*(\d)")
SCORES = (1, 2, 3, 4, 5)


def extract_result(text: str | None) -> int | None:
    if not text:
        return None
    matches = _RESULT.findall(text)
    if not matches:
        return None
    v = int(matches[-1])
    return v if v in SCORES else None


def score_distribution(top_logprobs: dict[str, float]) -> dict[int, float] | None:
    """{토큰: logprob} → {점수: 확률}. 점수 토큰이 하나도 없으면 None."""
    probs: dict[int, float] = {}
    for tok, lp in top_logprobs.items():
        t = tok.strip()
        if t.isdigit() and int(t) in SCORES:
            probs[int(t)] = probs.get(int(t), 0.0) + math.exp(lp)
    total = sum(probs.values())
    if total <= 0:
        return None
    return {s: probs.get(s, 0.0) / total for s in SCORES}


def argmax_score(dist: dict[int, float]) -> int:
    return max(SCORES, key=lambda s: (dist[s], -s))  # 동률이면 낮은 점수


def expected_score(dist: dict[int, float]) -> float:
    return sum(s * p for s, p in dist.items())
