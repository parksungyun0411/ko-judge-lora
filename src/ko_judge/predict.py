"""판정 실행: fold의 test 행(+ 그 fold 질문의 거짓 정보 응답)을 vLLM 서버에 보낸다.

- --mode judge: 생성형 제로샷. 학습이 없으므로 --fold all로 전체를 한 번에 돈다.
- --mode score: LoRA 판정(점수만). fold k 어댑터는 fold k의 test만 판정한다.

  python -m ko_judge.predict --mode score --served qwen3-4b-f0-s0 --fold 0 \
      --tag lora_qwen3-4b_bf16_s0
  python -m ko_judge.predict --mode judge --served qwen3-4b --fold all --tag zeroshot_qwen3-4b
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ko_judge import data as kd
from ko_judge.parsing import argmax_score, expected_score
from ko_judge.prompts import score_only_messages, zero_shot_messages
from ko_judge.splits import make_plan

PRED_DIR = Path("outputs/predictions")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def falseinfo_items(pw: pd.DataFrame, data_dir: Path = kd.DATA_DIR) -> pd.DataFrame:
    """거짓 정보 응답 54건 + 원래 응답의 pointwise row_id.

    원래 응답은 pointwise와 공백만 다르게 저장돼 있어 공백을 정규화한 뒤 맞춘다.
    """
    pf = kd.load("pointwise_false", data_dir)
    prf = kd.load("pairwise_false", data_dir)
    index = {
        (q, _norm(r)): rid for rid, q, r in zip(pw.row_id, pw.instruction, pw.response, strict=True)
    }
    orig = [
        index.get((q, _norm(o)))
        for q, o in zip(prf.instruction, prf.original_response, strict=True)
    ]
    if any(o is None for o in orig):
        raise ValueError("원래 응답을 pointwise에서 찾지 못한 행이 있다")
    return pd.DataFrame(
        {
            "false_id": range(len(pf)),
            "instruction": pf.instruction,
            "judge_query": pf.judge_query,
            "type": pf["type"],
            "original_row_id": orig,
        }
    )


def to_record(kind: str, key: int, res: dict) -> dict:
    if "dist" in res:
        dist = res["dist"]
        return {
            "kind": kind,
            "id": int(key),
            "score": argmax_score(dist) if dist else None,
            "expected": expected_score(dist) if dist else None,
            "dist": dist,
        }
    return {
        "kind": kind,
        "id": int(key),
        "score": res["score"],
        "expected": None,
        "attempts": res["attempts"],
    }


def run_fold(backend, mode: str, pw, fi, plan, fold: int, seed: int) -> list[dict]:
    test = pw[plan.test_mask(pw, fold)]
    ftest = fi[fi.instruction.map(plan.fold_of_instruction) == fold]
    to_msgs = score_only_messages if mode == "score" else zero_shot_messages
    batch = [to_msgs(q) for q in test.judge_query] + [to_msgs(q) for q in ftest.judge_query]
    results = backend.score(batch) if mode == "score" else backend.judge(batch, seed=seed)
    keys = [("pointwise", r) for r in test.row_id] + [("falseinfo", f) for f in ftest.false_id]
    return [to_record(k, i, r) for (k, i), r in zip(keys, results, strict=True)]


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["judge", "score"], required=True)
    p.add_argument("--served", required=True, help="vLLM 서빙 이름(어댑터 이름 포함)")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--fold", required=True, help="0~4 또는 all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", required=True)
    p.add_argument("--concurrency", type=int, default=32)
    args = p.parse_args()

    from ko_judge.generation import OpenAIBackend

    pw = kd.load_pointwise()
    plan = make_plan(pw)
    fi = falseinfo_items(pw)
    backend = OpenAIBackend(args.base_url, args.served, args.concurrency)
    folds = range(plan.n_folds) if args.fold == "all" else [int(args.fold)]
    out_dir = PRED_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in folds:
        recs = run_fold(backend, args.mode, pw, fi, plan, k, args.seed)
        with (out_dir / f"fold{k}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{args.tag} fold{k}: {len(recs)}건")


if __name__ == "__main__":
    main()
