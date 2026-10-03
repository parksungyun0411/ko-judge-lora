"""YAML 설정 로드와 검증. 학습을 GPU에 올리기 전에 잘못된 설정을 여기서 막는다."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from ko_judge.prompts import TARGET_MODES

MODELS_YAML = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str
    license: str
    architecture: str
    card: str


def load_models(path: str | Path = MODELS_YAML) -> dict[str, ModelSpec]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    allowed = set(doc["allowed_licenses"])
    out = {}
    for key, m in doc["models"].items():
        if m["license"] not in allowed:
            raise ConfigError(f"{key}: 허용하지 않는 라이선스 {m['license']}")
        if len(str(m["revision"])) != 40:
            raise ConfigError(f"{key}: revision은 40자 커밋 해시로 고정한다")
        out[key] = ModelSpec(key=key, **m)
    return out


@dataclass
class LoraSection:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"


@dataclass
class TrainSection:
    epochs: float = 2.0
    learning_rate: float = 1e-4
    per_device_batch_size: int = 2
    grad_accum: int = 8
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"
    max_length: int = 4096
    gradient_checkpointing: bool = True
    bf16: bool = True
    logging_steps: int = 10


@dataclass
class TrainConfig:
    model: str
    target: str = "sample"
    output_dir: str = "outputs/lora"
    lora: LoraSection = field(default_factory=LoraSection)
    train: TrainSection = field(default_factory=TrainSection)


def _section(cls, raw: dict | None, name: str):
    raw = raw or {}
    unknown = set(raw) - {f.name for f in fields(cls)}
    if unknown:
        raise ConfigError(f"{name}: 알 수 없는 키 {sorted(unknown)}")
    return cls(**raw)


def validate(cfg: TrainConfig, models: dict[str, ModelSpec]) -> None:
    if cfg.model not in models:
        raise ConfigError(f"model은 {sorted(models)} 중 하나: {cfg.model}")
    if cfg.target not in TARGET_MODES:
        raise ConfigError(f"target은 {TARGET_MODES} 중 하나: {cfg.target}")
    lo, tr = cfg.lora, cfg.train
    if lo.r <= 0 or lo.alpha <= 0:
        raise ConfigError("lora.r, lora.alpha는 양수")
    if not 0 <= lo.dropout < 1:
        raise ConfigError("lora.dropout은 [0, 1)")
    if not (0 < tr.learning_rate < 1e-2):
        raise ConfigError("train.learning_rate가 범위를 벗어났다")
    if tr.epochs <= 0 or tr.per_device_batch_size <= 0 or tr.grad_accum <= 0:
        raise ConfigError("epochs·batch·grad_accum은 양수")
    if not 0 <= tr.warmup_ratio < 1:
        raise ConfigError("train.warmup_ratio는 [0, 1)")
    if tr.max_length < 2048:
        # judge_query는 Qwen3 토크나이저로 평균 2,040토큰, 최대 6,853토큰(reports/gpu_budget.md)
        raise ConfigError("train.max_length가 너무 짧다 (프롬프트 중앙값이 2천 토큰대)")


def load_train_config(path: str | Path, models_path: str | Path = MODELS_YAML) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(raw) - {f.name for f in fields(TrainConfig)}
    if unknown:
        raise ConfigError(f"알 수 없는 키 {sorted(unknown)}")
    if "model" not in raw:
        raise ConfigError("model 키가 필요하다")
    cfg = TrainConfig(
        **{k: v for k, v in raw.items() if k not in {"lora", "train"}},
        lora=_section(LoraSection, raw.get("lora"), "lora"),
        train=_section(TrainSection, raw.get("train"), "train"),
    )
    validate(cfg, load_models(models_path))
    return cfg


def main() -> None:
    """셸에서 모델 정보를 꺼낼 때: python -m ko_judge.config qwen3-4b hf_id"""
    import sys

    key, attr = sys.argv[1], sys.argv[2]
    print(getattr(load_models()[key], attr))


if __name__ == "__main__":
    main()
