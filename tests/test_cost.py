import json

import pytest

from ko_judge.cost import PENDING, build_rows, metric_value, render_markdown, usd_per_million

CFG = {"rows": [{"name": "A", "metrics": "a", "throughput": "a"}, {"name": "B", "metrics": "b"}]}


def test_usd_per_million_hand_computed():
    # 1건/s, $3.6/h → 3,600건에 $3.6 → 100만 건에 $1,000
    assert usd_per_million(3.6, 1) == pytest.approx(1000.0)
    assert usd_per_million(0.72, 2) == pytest.approx(100.0)
    with pytest.raises(ValueError):
        usd_per_million(1.0, 0)


def test_all_pending(tmp_path):
    md = render_markdown(build_rows(CFG, tmp_path, 0.5), None, "")
    assert md.count(PENDING) >= 10
    assert next(x for x in md.splitlines() if x.startswith("| B")).endswith("| — | — | — |")


def test_single_and_seed_files(tmp_path):
    (tmp_path / "metrics").mkdir()
    (tmp_path / "throughput").mkdir()
    single = {
        "agreement": {"all": {"paper_acc": 0.5, "pearson": 0.25}},
        "ranking": {"tau_all": 0.8, "tau_holdout": None},
        "falseinfo": {"detected_rate": 0.1},
    }
    (tmp_path / "metrics" / "a.json").write_text(json.dumps(single))
    (tmp_path / "throughput" / "a.json").write_text(
        json.dumps({"items_per_s": 2.0, "latency_p95_s": 1.5})
    )
    seeds = {
        "seeds": {"agreement.all.paper_acc": {"mean": 0.5, "min": 0.4, "max": 0.6, "n_seeds": 3}}
    }
    (tmp_path / "metrics" / "b.json").write_text(json.dumps(seeds))
    rows = build_rows(CFG, tmp_path, 0.72)
    md = render_markdown(rows, 0.72, "g")
    assert "| A | 0.5000 | 0.250 | 0.800 | 측정 전 | 0.100 | 2.00 | 1500 | 100.00 |" in md
    assert "| B | 0.5000 (0.4000~0.6000, n=3) | 측정 전 |" in md
    assert metric_value(None, ("x",), ".2f") == PENDING


def test_not_run_rows_show_reason(tmp_path):
    cfg = {
        "rows": [
            {"name": "C LoRA", "metrics": "c", "not_run": "T4 할당 한도"},
            {"name": "C AWQ", "metrics": "d", "not_run": "T4 할당 한도"},
        ]
    }
    md = render_markdown(build_rows(cfg, tmp_path, 0.1), 0.1, "T4", price_note="출처 X")
    assert "| C LoRA |" + " 측정 안 함 |" * 8 in md
    assert md.count("T4 할당 한도") == 1 and "- C LoRA, C AWQ: T4 할당 한도" in md
    assert "요금 근거: 출처 X" in md
