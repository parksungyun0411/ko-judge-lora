import pandas as pd
import pytest

from ko_judge.ranking import bootstrap_ranking, min_n_for, model_means, tau


def test_tau_extremes():
    a = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
    assert tau(a, a) == pytest.approx(1.0)
    assert tau(a, -a) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        tau(a[:2], a[:2])


def test_bootstrap_identical_scores_give_tau_one(pw):
    r = bootstrap_ranking(
        pw, "final_score", ref_col="final_score", ns=(5, 20), n_boot=50, same_sample_ref=True
    )
    assert all(x["tau_mean"] == pytest.approx(1.0) for x in r["results"])
    assert all(x["top1"] == 1.0 for x in r["results"])


def test_bootstrap_vs_full_improves_with_n(pw):
    r = bootstrap_ranking(pw, "final_score", ns=(2, 20), n_boot=200, seed=1)
    small, large = r["results"]
    assert large["tau_mean"] > small["tau_mean"]
    assert small["tau_p2_5"] <= small["tau_mean"] <= small["tau_p97_5"]
    assert r == bootstrap_ranking(pw, "final_score", ns=(2, 20), n_boot=200, seed=1)


def test_min_n_for():
    r = {"results": [{"n": 10, "tau_p2_5": 0.7}, {"n": 20, "tau_p2_5": 0.91}]}
    assert min_n_for(r, 0.9) == 20
    assert min_n_for(r, 0.95) is None


def test_model_means(pw):
    m = model_means(pw, "final_score")
    assert list(m.index) == sorted(m.index)
    assert m["m11"] > m["m05"] > m["m00"]
