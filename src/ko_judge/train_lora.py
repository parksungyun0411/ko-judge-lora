"""PEFT LoRA + TRL SFTTrainer로 fold 하나를 학습한다.

train 행은 splits.SplitPlan.train_mask(fold)로 고른다: 그 fold의 질문과 보류 모델은 빠진다.
학습 중 평가 분할을 두지 않고 epoch 수를 고정한다. test fold를 보고 멈추는 일이 없게 하기 위해서다.

  python -m ko_judge.train_lora --config configs/lora_qwen3-4b.yaml --fold 0 --seed 0
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ko_judge import compat
from ko_judge.config import load_models, load_train_config
from ko_judge.data import load_pointwise
from ko_judge.prompts import sft_records
from ko_judge.splits import make_plan


def model_dtype(tr) -> str:
    """기반 가중치 dtype. fp16 학습에서도 LoRA 어댑터는 PEFT가 fp32로 올린다."""
    if tr.bf16:
        return "bfloat16"
    return "float16" if tr.fp16 else "float32"


def adapter_dir(cfg, fold: int, seed: int) -> Path:
    return Path(cfg.output_dir) / f"fold{fold}_seed{seed}"


def build_dataset(cfg, rows: list[dict], seed: int):
    from datasets import Dataset

    return Dataset.from_list(sft_records(rows, cfg.target, seed)).shuffle(seed=seed)


def make_trainer(cfg, model_id: str, revision: str | None, train_ds, out_dir, seed, **overrides):
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    tr, lo = cfg.train, cfg.lora
    args = SFTConfig(
        output_dir=str(out_dir),
        seed=seed,
        num_train_epochs=tr.epochs,
        learning_rate=tr.learning_rate,
        per_device_train_batch_size=tr.per_device_batch_size,
        gradient_accumulation_steps=tr.grad_accum,
        # transformers 5.17 TrainingArguments: "A float in range [0, 1): interpreted as ratio of
        # total training steps" (v5에서 warmup_ratio가 warmup_steps로 합쳐졌다)
        warmup_steps=tr.warmup_ratio,
        lr_scheduler_type=tr.lr_scheduler,
        max_length=tr.max_length,
        gradient_checkpointing=tr.gradient_checkpointing,
        bf16=tr.bf16,
        fp16=tr.fp16,
        logging_steps=tr.logging_steps,
        eval_strategy="no",
        save_strategy="no",
        completion_only_loss=True,
        report_to="none",
        model_init_kwargs={"revision": revision, "dtype": model_dtype(tr)},
        **overrides,
    )
    peft_config = LoraConfig(
        r=lo.r,
        lora_alpha=lo.alpha,
        lora_dropout=lo.dropout,
        target_modules=lo.target_modules,
        task_type="CAUSAL_LM",
    )
    return SFTTrainer(model=model_id, args=args, train_dataset=train_ds, peft_config=peft_config)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model-path", default=None, help="HF ID 대신 쓸 로컬 경로 (스모크용)")
    p.add_argument("--max-steps", type=int, default=None, help="스모크용 상한")
    args = p.parse_args()

    compat.apply()
    cfg = load_train_config(args.config)
    spec = load_models()[cfg.model]
    model_id = args.model_path or spec.hf_id
    revision = None if args.model_path else spec.revision

    df = load_pointwise()
    plan = make_plan(df)
    rows = df[plan.train_mask(df, args.fold)].to_dict(orient="records")
    train_ds = build_dataset(cfg, rows, args.seed)
    out = adapter_dir(cfg, args.fold, args.seed)
    overrides = {"max_steps": args.max_steps} if args.max_steps else {}
    trainer = make_trainer(cfg, model_id, revision, train_ds, out, args.seed, **overrides)
    result = trainer.train()
    trainer.save_model(str(out))
    trainer.processing_class.save_pretrained(str(out))
    (out / "train_summary.json").write_text(
        json.dumps(
            {
                "model": cfg.model,
                "hf_id": model_id,
                "revision": revision,
                "fold": args.fold,
                "seed": args.seed,
                "holdout_models": list(plan.holdout_models),
                "n_train_rows": len(rows),
                "n_train_examples": len(train_ds),
                "config": asdict(cfg),
                "metrics": result.metrics,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(f"adapter → {out}")


if __name__ == "__main__":
    main()
