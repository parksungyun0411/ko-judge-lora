"""저장소에 넣은 예측·실행 기록에서 reports/의 수치가 그대로 다시 나오는지 확인한다."""

import json
from pathlib import Path

import pytest
from conftest import needs_data

from ko_judge import data as kd
from ko_judge import run_checks
from ko_judge.evaluate import evaluate, load_predictions
from ko_judge.splits import make_plan

TAGS = sorted(p.name for p in Path("outputs/predictions").iterdir())


def _flat(d, prefix: str = "") -> dict:
    """중첩 dict·list를 '경로: 값'으로 편다(pytest.approx가 중첩을 받지 않는다)."""
    items = d.items() if isinstance(d, dict) else enumerate(d)
    out = {}
    for k, v in items:
        if isinstance(v, dict | list):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def test_vllm_log_parsing(tmp_path):
    log = tmp_path / "v.log"
    log.write_text(
        "INFO [cuda.py:538] Using TRITON_ATTN attention backend\n"
        "INFO [model_runner.py:428] Model loading took 1.34 GiB memory and 10.5 seconds\n"
        "INFO [kv_cache_utils.py:2395] GPU KV cache size: 87,296 tokens, Maximum concurrency\n"
        "Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, "
        "Running: 0 reqs, Waiting: 0 reqs\n"
        "Avg prompt throughput: 1000.0 tokens/s, Avg generation throughput: 0.7 tokens/s, "
        "Running: 2 reqs, Waiting: 28 reqs\n"
        "Avg prompt throughput: 1400.0 tokens/s, Avg generation throughput: 0.9 tokens/s, "
        "Running: 3 reqs, Waiting: 27 reqs\n"
    )
    r = run_checks.vllm_log(log)
    assert r["weights_gib"] == 1.34 and r["kv_cache_tokens"] == 87296
    assert r["attention_backend"] == "TRITON_ATTN" and not r["marlin_kernel"]
    assert r["prompt_tps_intervals"] == 2 and r["prompt_tps_median"] == 1200.0
    assert r["running_reqs_counts"] == {2: 1, 3: 1}


def test_same_rate_by_fold(tmp_path):
    for tag, scores in (("a", [1, 2, 3, 4]), ("b", [1, 2, 3, 5])):
        (tmp_path / tag).mkdir()
        for k in range(5):
            rows = [{"kind": "pointwise", "id": i, "score": s} for i, s in enumerate(scores)]
            text = "\n".join(json.dumps(r) for r in rows)
            (tmp_path / tag / f"fold{k}.jsonl").write_text(text + "\n")
    out = run_checks.same_rate_by_fold("a", "b", tmp_path)
    assert out[0] == {"n": 4, "same": 0.75}


def test_predictions_have_no_text():
    """예측 파일에는 점수만 있다. 응답·피드백 본문은 저장소에 넣지 않는다."""
    allowed = {"kind", "id", "score", "expected", "dist", "attempts"}
    for f in Path("outputs/predictions").glob("*/fold*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            assert set(json.loads(line)) <= allowed, f


@needs_data
@pytest.mark.parametrize("tag", TAGS)
def test_metrics_reproduce_from_committed_predictions(tag):
    pw = kd.load_pointwise()
    got = evaluate(load_predictions(tag), pw, make_plan(pw), n_boot=10)
    want = json.loads((Path("reports/metrics") / f"{tag}.json").read_text())
    assert _flat(got["agreement"]) == pytest.approx(_flat(want["agreement"]))
    for key in ("tau_all", "tau_holdout", "tau_seen"):
        assert got["ranking"][key] == pytest.approx(want["ranking"][key])
    assert got["falseinfo"]["detected_rate"] == pytest.approx(want["falseinfo"]["detected_rate"])


@needs_data
def test_run_checks_matches_committed_report():
    got = json.loads(json.dumps(run_checks.collect()))
    want = json.loads(run_checks.OUT.read_text())
    assert _flat(got) == pytest.approx(_flat(want))
