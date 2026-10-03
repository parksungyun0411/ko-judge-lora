"""픽스처. 실제 KUDGE는 data/kudge에 내려받았을 때만 쓴다(`make data`)."""

import numpy as np
import pandas as pd
import pytest

from ko_judge import data as kd

HAS_DATA = all((kd.DATA_DIR / s.name).exists() for s in kd.SOURCES.values())
needs_data = pytest.mark.skipif(not HAS_DATA, reason="data/kudge 없음 — make data")


def synthetic_pointwise(n_instr: int = 20, n_models: int = 12, seed: int = 0) -> pd.DataFrame:
    """질문 × 모델 격자에서 일부 칸을 비운 KUDGE 모양 데이터. 모델 품질에 순서가 있다."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_instr):
        for m in range(n_models):
            if rng.random() < 0.1:
                continue
            base = 1 + 4 * m / (n_models - 1)
            s1 = int(np.clip(round(base + rng.normal(0, 0.7)), 1, 5))
            s2 = int(np.clip(s1 + rng.choice([-1, 0, 0, 1]), 1, 5))
            rows.append(
                {
                    "instruction": f"q{i:02d}",
                    "response": f"r{i}-{m}",
                    "model": f"m{m:02d}",
                    "score1": s1,
                    "score2": s2,
                    "judge_query": f"질문 q{i} 응답 m{m}\n\n###Feedback:",
                    "final_score": (s1 + s2) / 2,
                }
            )
    df = pd.DataFrame(rows)
    df.insert(0, "row_id", range(len(df)))
    return df


@pytest.fixture
def pw() -> pd.DataFrame:
    return synthetic_pointwise()
