"""점수 일치도 지표.

논문 Table 3(5쪽)의 pointwise 정확도는 "off-by-0.5"다. 사람 점수는 두 평가자의 평균이라
0.5 단위이고 판정은 정수이므로, 논문 저장소 src/pointwise_eval.py는 |판정 − 사람| < 1을
맞음으로 센다(정수 판정에서는 ≤ 0.5와 같다). 형식 실패(None)는 분모에서 뺀다.
여기서는 같은 정의(`paper_acc`)와, 실패를 오답으로 센 값(`strict_acc`)을 함께 낸다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import pearsonr

from ko_judge.parsing import SCORES


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _pred(pred) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in pred])


def off_by_half(pred: Sequence[float | None], final: Sequence[float]) -> dict:
    final = _arr(final)
    if len(pred) != len(final) or len(final) == 0:
        raise ValueError("길이가 같고 비어 있지 않아야 한다")
    p = _pred(pred)
    parsed = ~np.isnan(p)
    hit = np.abs(p - final) < 1
    n_ok = int(parsed.sum())
    return {
        "n": len(final),
        "failures": len(final) - n_ok,
        "paper_acc": float(hit[parsed].mean()) if n_ok else 0.0,
        "strict_acc": float(np.where(parsed, hit, False).mean()),
    }


def pearson(pred: Sequence[float | None], final: Sequence[float]) -> float | None:
    pairs = [(p, f) for p, f in zip(pred, final, strict=True) if p is not None]
    if len(pairs) < 3 or len({p for p, _ in pairs}) < 2:
        return None
    a, b = zip(*pairs, strict=True)
    return float(pearsonr(a, b)[0])


def pair_agreement(a: Sequence[float], b: Sequence[float]) -> dict:
    """두 정수 평정 사이 일치: 완전 일치, 1점 이내, 피어슨."""
    a, b = _arr(a), _arr(b)
    d = np.abs(a - b)
    return {
        "n": len(a),
        "exact": float((d == 0).mean()),
        "within_1": float((d <= 1).mean()),
        "pearson": float(pearsonr(a, b)[0]),
    }


def judge_vs_annotators(pred: Sequence[int | None], s1, s2) -> dict:
    """판정을 각 평가자와 따로 비교해 평균한다. 사람 간 일치(score1 대 score2)와 같은 척도.
    형식 실패는 오답으로 센다."""
    p = _pred(pred)
    out = {}
    for key, ok_fn in (("exact", lambda d: d == 0), ("within_1", lambda d: d <= 1)):
        vals = [
            float(np.where(np.isnan(p), False, ok_fn(np.abs(p - _arr(s)))).mean()) for s in (s1, s2)
        ]
        out[key] = float(np.mean(vals))
    return out


def chance_off_by_half(final: Sequence[float]) -> dict:
    """정수 1~5를 고르게 찍을 때의 기대 정확도, 그리고 가장 나은 상수 점수."""
    final = _arr(final)
    per_k = {k: float((np.abs(k - final) < 1).mean()) for k in SCORES}
    best = max(per_k, key=per_k.get)
    return {
        "uniform": float(np.mean(list(per_k.values()))),
        "best_constant": best,
        "best_constant_acc": per_k[best],
        "per_constant": per_k,
    }


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """rag-eval-harness의 stats.wilson_interval과 같은 식.

    그 저장소는 비공개라 CI에서 설치할 수 없어 의존성으로 넣지 않고 복제했다.
    """
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("0 ≤ successes ≤ n, n ≥ 1")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def seed_summary(values: Sequence[float]) -> dict:
    """시드별 값 → 평균, 최소·최대, 표준편차, t 기반 95% 구간(시드 2개 이상)."""
    from scipy.stats import t

    v = _arr(values)
    n = len(v)
    out = {"n_seeds": n, "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    if n >= 2:
        sd = float(v.std(ddof=1))
        half = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        out.update(sd=sd, ci95=[out["mean"] - half, out["mean"] + half])
    return out
