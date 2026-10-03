import pytest
from conftest import needs_data

from ko_judge import data as kd


def test_hash_check_rejects_mismatch(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a\n1\n")
    with pytest.raises(ValueError, match="sha256"):
        kd.check_hash(p, "0" * 64)
    kd.check_hash(p, kd.sha256_file(p))


def test_sources_are_pinned():
    for s in kd.SOURCES.values():
        assert len(s.sha256) == 64
        assert kd.GITHUB_COMMIT in s.url or kd.HF_REVISION in s.url


@needs_data
def test_reported_contents_hold():
    pw = kd.load_pointwise()
    assert len(pw) == 2506 and pw.instruction.nunique() == 87 and pw.model.nunique() == 31
    assert {"score1", "score2", "final_score"} <= set(pw.columns)
    assert ((pw.score1 + pw.score2) / 2 == pw.final_score).all()
    assert len(kd.load("pairwise")) == 818
    pf = kd.load("pointwise_false")
    assert len(pf) == 54 and len(kd.load("pairwise_false")) == 54
    assert set(pf["type"]) == {"word", "sentence", "general"}  # \x08 제거 확인


@needs_data
def test_two_point_disagreements_are_in_the_release():
    """논문 3.2절은 1점 넘게 갈린 사례를 뺐다고 적었지만 배포본에는 2점 차이 405행이 있다."""
    pw = kd.load_pointwise()
    assert int(((pw.score1 - pw.score2).abs() == 2).sum()) == 405
