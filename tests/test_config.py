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
        ({"model": "qwen3-4b", "train": {"fp16": True}}, "함께 켤 수 없다"),  # bf16 기본값과 충돌
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


def _cfg(tmp_path, model, max_length):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"model": model, "train": {"max_length": max_length}}))
    return p


def test_max_length_must_cover_longest_prompt_plus_answer(tmp_path):
    from ko_judge.config import ANSWER_TOKEN_MARGIN

    need = 6874 + ANSWER_TOKEN_MARGIN  # Qwen3-4B 최장 프롬프트(reports/gpu_budget.json)
    load_train_config(_cfg(tmp_path, "qwen3-4b", need))
    with pytest.raises(ConfigError, match="가장 긴 프롬프트 6874"):
        load_train_config(_cfg(tmp_path, "qwen3-4b", need - 1))
    with pytest.raises(ConfigError, match="max_length 4096"):
        load_train_config(_cfg(tmp_path, "qwen3-4b", 4096))
    load_train_config(_cfg(tmp_path, "kanana-1.5-2.1b", 6000))  # kanana 최장 5,721


def test_default_max_length_is_safe_for_all_models(tmp_path):
    from ko_judge.config import ANSWER_TOKEN_MARGIN, TrainSection

    for m in load_models().values():
        assert TrainSection().max_length >= m.max_prompt_tokens + ANSWER_TOKEN_MARGIN
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"model": "qwen3-4b"}))  # train 절 없이 기본값만
    load_train_config(p)


def test_max_prompt_tokens_match_budget_report():
    import json

    budget = json.loads((Path(__file__).parents[1] / "reports/gpu_budget.json").read_text())
    for key, m in load_models().items():
        assert m.max_prompt_tokens == budget["models"][key]["prompt_tokens"]["max"]


def test_fp16_path_dtypes():
    """T4용 fp16 경로: 학습 기반 가중치 dtype과 HF 생성 백엔드의 dtype 선택."""
    from types import SimpleNamespace

    from ko_judge.generation import auto_dtype
    from ko_judge.train_lora import model_dtype

    assert model_dtype(SimpleNamespace(bf16=True, fp16=False)) == "bfloat16"
    assert model_dtype(SimpleNamespace(bf16=False, fp16=True)) == "float16"
    assert model_dtype(SimpleNamespace(bf16=False, fp16=False)) == "float32"
    cuda = SimpleNamespace(is_bf16_supported=lambda: False)
    torch = SimpleNamespace(float32="f32", bfloat16="bf16", float16="f16", cuda=cuda)
    assert auto_dtype(torch, "cpu") == "f32"
    assert auto_dtype(torch, "cuda") == "f16"  # T4
    cuda.is_bf16_supported = lambda: True
    assert auto_dtype(torch, "cuda") == "bf16"
