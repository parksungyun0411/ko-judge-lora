"""병합 모델을 llm-compressor AWQ + W4A16_ASYM 프리셋으로 양자화한다
(4bit 가중치·16bit 활성값, 비대칭).

보정(calibration) 데이터는 그 fold의 train 행에서만 뽑는다. test fold 질문은 쓰지 않는다.
결과는 compressed-tensors 형식이라 vLLM이 그대로 읽는다.

  python -m ko_judge.quantize --model-dir outputs/merged/qwen3-4b/fold0_seed0 \
      --out outputs/awq/qwen3-4b/fold0_seed0 --fold 0
"""

from __future__ import annotations

import random

from ko_judge import compat
from ko_judge.data import load_pointwise
from ko_judge.prompts import score_only_messages
from ko_judge.splits import make_plan

SCHEME = "W4A16_ASYM"


def calibration_texts(tokenizer, rows: list[dict], n: int, seed: int) -> list[str]:
    picked = random.Random(seed).sample(rows, min(n, len(rows)))
    texts = []
    for r in picked:
        messages = score_only_messages(r["judge_query"]) + [
            {"role": "assistant", "content": str(int(r["score1"]))}
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return texts


def build_recipe():
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    return [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(ignore=["lm_head"], scheme=SCHEME, targets=["Linear"]),
    ]


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    compat.apply()
    from datasets import Dataset
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = load_pointwise()
    rows = df[make_plan(df).train_mask(df, args.fold)].to_dict(orient="records")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    texts = calibration_texts(tokenizer, rows, args.num_samples, args.seed)
    ds = Dataset.from_dict({"text": texts}).map(
        lambda s: tokenizer(
            s["text"],
            padding=False,
            max_length=args.max_seq_length,
            truncation=True,
            add_special_tokens=False,
        ),
        remove_columns=["text"],
    )
    oneshot(
        model=model,
        dataset=ds,
        recipe=build_recipe(),
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(texts),
    )
    model.save_pretrained(args.out, save_compressed=True)
    tokenizer.save_pretrained(args.out)
    print(f"AWQ {SCHEME} → {args.out}")


if __name__ == "__main__":
    main()
