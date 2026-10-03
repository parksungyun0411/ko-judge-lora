from pathlib import Path

import pytest
import yaml

from ko_judge.config import MODELS_YAML, ConfigError, load_models, load_train_config

CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("lora_*.yaml"))


def test_models_and_licences():
    models = load_models()
    assert set(models) == {"qwen3-4b", "kanana-1.5-2.1b"}
    assert {m.license for m in models.values()} == {"apache-2.0"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_validate(path):
    cfg = load_train_config(path)
    assert path.stem == f"lora_{cfg.model}"
    assert cfg.train.max_length >= 6874  # 가장 긴 프롬프트(Qwen3 토큰)도 자르지 않는다


@pytest.mark.parametrize(
    "doc,match",
    [
        ({"target": "sample"}, "model 키"),
        ({"model": "exaone"}, "model은"),
        ({"model": "qwen3-4b", "target": "mean"}, "target"),
        ({"model": "qwen3-4b", "lora": {"rank": 8}}, "알 수 없는 키"),
        ({"model": "qwen3-4b", "lora": {"r": 0}}, "양수"),
        ({"model": "qwen3-4b", "train": {"max_length": 1024}}, "max_length"),
        ({"model": "qwen3-4b", "train": {"learning_rate": 0.5}}, "learning_rate"),
    ],
)
def test_invalid_configs(tmp_path, doc, match):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ConfigError, match=match):
        load_train_config(p)


def test_rejects_unpinned_or_non_permissive(tmp_path):
    for field, value, match in (("revision", "main", "revision"), ("license", "other", "라이선스")):
        doc = yaml.safe_load(MODELS_YAML.read_text())
        doc["models"]["qwen3-4b"][field] = value
        p = tmp_path / "m.yaml"
        p.write_text(yaml.safe_dump(doc))
        with pytest.raises(ConfigError, match=match):
            load_models(p)
