"""실제 KUDGE에 가짜 판정 모델을 붙여 예측 → 지표 경로를 확인한다."""

import pandas as pd
import pytest
from conftest import needs_data

from ko_judge import data as kd
from ko_judge.evaluate import evaluate, summarize_seeds
from ko_judge.metrics import pearson
from ko_judge.predict import falseinfo_items, run_fold
from ko_judge.prompts import score_only_prompt
from ko_judge.splits import make_plan

pytestmark = needs_data


class OracleScorer:
    """pointwise는 평가자 1 점수에 확률 1, 거짓 정보 응답(학습 행에 없는 프롬프트)은 1점."""

    def __init__(self, pw):
        self.by_prompt = {
            score_only_prompt(q): int(s) for q, s in zip(pw.judge_query, pw.score1, strict=True)
        }

    def score(self, batch):
        hits = [self.by_prompt.get(m[0]["content"], 1) for m in batch]
        return [{"dist": {s: float(s == h) for s in range(1, 6)}} for h in hits]


@pytest.fixture(scope="module")
def setup():
    pw = kd.load_pointwise()
    plan = make_plan(pw)
    fi = falseinfo_items(pw)
    return pw, plan, fi


def test_falseinfo_originals_found(setup):
    pw, _, fi = setup
    assert len(fi) == 54 and fi.original_row_id.notna().all()
    assert set(pw.set_index("row_id").loc[fi.original_row_id, "final_score"]) <= {4.5, 5.0}


def test_oracle_end_to_end(setup):
    pw, plan, fi = setup
    backend = OracleScorer(pw)
    fold0 = run_fold(backend, "score", pw, fi, plan, 0, seed=0)
    n_test = int(plan.test_mask(pw, 0).sum())
    assert sum(r["kind"] == "pointwise" for r in fold0) == n_test

    # 전체 fold를 돌리는 대신 정답으로 바로 만든 예측(속도)
    preds = [
        {"kind": "pointwise", "id": r, "score": int(s), "expected": float(s)}
        for r, s in zip(pw.row_id, pw.score1, strict=True)
    ]
    preds += [{"kind": "falseinfo", "id": f, "score": 1, "expected": 1.0} for f in fi.false_id]
    res = evaluate(pd.DataFrame(preds), pw, plan, n_boot=20)
    a = res["agreement"]["all"]
    # score1 대 평균: |s1 − (s1+s2)/2| = |s1−s2|/2 < 1 ⇔ 차이 2점 미만 → 2,506 − 405행
    assert a["paper_acc"] == pytest.approx((2506 - 405) / 2506)
    assert a["failures"] == 0
    assert res["agreement"]["all"]["vs_annotators"]["exact"] == pytest.approx(
        (1 + float((pw.score1 == pw.score2).mean())) / 2
    )
    assert res["ranking"]["tau_seen"] > 0.8
    assert res["falseinfo"]["detected_rate"] == 1.0  # 원래 응답은 평가자 1 점수 4~5

    s = summarize_seeds([res, res, res])
    assert s["agreement.all.paper_acc"]["n_seeds"] == 3
    assert s["agreement.all.paper_acc"]["sd"] == 0


def test_missing_rows_rejected(setup):
    pw, plan, _ = setup
    with pytest.raises(ValueError, match="예측이 없는"):
        evaluate(
            pd.DataFrame([{"kind": "pointwise", "id": 0, "score": 1, "expected": 1.0}]), pw, plan
        )


def test_only_fold_uses_that_fold_only(setup):
    pw, plan, fi = setup
    test0 = pw[plan.test_mask(pw, 0)]
    preds = [
        {"kind": "pointwise", "id": r, "score": int(s), "expected": float(s)}
        for r, s in zip(test0.row_id, test0.score1, strict=True)
    ]
    f0 = fi[fi.instruction.map(plan.fold_of_instruction) == 0]
    preds += [{"kind": "falseinfo", "id": f, "score": 1, "expected": 1.0} for f in f0.false_id]
    res = evaluate(pd.DataFrame(preds), pw, plan, n_boot=10, only_fold=0)
    assert res["only_fold"] == 0 and res["n_rows"] == len(test0)
    assert res["falseinfo"]["n"] == len(f0)
    with pytest.raises(ValueError, match="예측이 없는"):
        evaluate(pd.DataFrame(preds), pw, plan, n_boot=10)  # 전체 평가는 모든 fold가 필요


def _lora_preds(pw, fi, shift_expected, falseinfo_tie=False):
    """정수 점수는 평가자 1 점수, 기댓값은 shift_expected(행)로 옮긴 값."""
    preds = [
        {
            "kind": "pointwise",
            "id": r.row_id,
            "score": int(r.score1),
            "expected": float(r.score1) + shift_expected(r),
        }
        for r in pw.itertuples()
    ]
    s1 = pw.set_index("row_id").score1
    for x in fi.itertuples():
        orig = int(s1[x.original_row_id])
        if falseinfo_tie:
            # 정수로는 원래 응답과 같고, 기댓값만 조금 낮다
            preds.append(
                {
                    "kind": "falseinfo",
                    "id": x.false_id,
                    "score": orig,
                    "expected": orig + shift_expected(None) - 0.01,
                }
            )
        else:
            preds.append({"kind": "falseinfo", "id": x.false_id, "score": 1, "expected": 1.0})
    return pd.DataFrame(preds)


def test_headline_metrics_use_integer_scores(setup):
    pw, plan, fi = setup
    # 기댓값을 행마다 다르게 흔들어 정수 점수와 다른 연속 점수를 만든다
    preds = _lora_preds(pw, fi, lambda r: 0.0 if r is None else 0.4 * ((r.row_id % 3) - 1))
    res = evaluate(preds, pw, plan, n_boot=5)
    a = res["agreement"]["all"]
    assert res["judge_score"] == "integer"
    assert a["pearson"] == pytest.approx(pearson(pw.score1.tolist(), pw.final_score.tolist()))
    exp = (pw.score1 + 0.4 * ((pw.row_id % 3) - 1)).tolist()
    assert res["expected"]["pearson"] == pytest.approx(pearson(exp, pw.final_score.tolist()))
    assert res["expected"]["pearson"] != pytest.approx(a["pearson"])
    # 순위 τ도 정수 점수 기준이고, 기댓값 기준은 따로 남는다
    assert set(res["ranking"]) >= {"tau_all", "tau_holdout", "tau_seen", "stability"}
    assert set(res["expected"]["ranking"]) >= {"tau_all", "tau_holdout", "tau_seen"}
    # 사람 간 피어슨과 같은 척도: 판정 대 평가자 1, 판정 대 평가자 2의 평균
    want = (1.0 + pearson(pw.score1.tolist(), pw.score2.tolist())) / 2
    assert a["vs_annotators"]["pearson"] == pytest.approx(want)


def test_falseinfo_ties_are_not_detected_with_integer_scores(setup):
    pw, plan, fi = setup
    preds = _lora_preds(pw, fi, lambda r: 0.0, falseinfo_tie=True)
    res = evaluate(preds, pw, plan, n_boot=5)
    # 정수로는 모두 동점이라 감지 0, 기댓값으로는 0.01 차이로 모두 감지된다
    assert res["falseinfo"]["detected_rate"] == 0.0
    assert res["falseinfo"]["tie_rate"] == 1.0
    assert res["expected"]["falseinfo"]["detected_rate"] == 1.0


def test_zero_shot_has_no_expected_block(setup):
    pw, plan, fi = setup
    preds = [
        {"kind": "pointwise", "id": r, "score": int(s), "expected": None}
        for r, s in zip(pw.row_id, pw.score1, strict=True)
    ]
    preds += [{"kind": "falseinfo", "id": f, "score": 1, "expected": None} for f in fi.false_id]
    res = evaluate(pd.DataFrame(preds), pw, plan, n_boot=5)
    assert res["expected"] is None
    assert res["falseinfo"]["detected_rate"] == 1.0
    s = summarize_seeds([res, res])
    assert s["expected.pearson"] is None
    assert s["falseinfo.detected_rate"]["n_seeds"] == 2
