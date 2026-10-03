"""작은 랜덤 Llama로 fold 학습 → 어댑터 저장 → 병합 → 점수 분포 경로를 CPU에서 2스텝 돌린다.

코드 경로만 확인한다. 이 모델의 출력·손실은 의미가 없고 어디에도 보고하지 않는다.
torch·transformers·peft·trl·datasets가 없으면 건너뛴다(CI 기본 환경).
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from ko_judge.config import LoraSection, TrainConfig, TrainSection  # noqa: E402
from ko_judge.prompts import SCORE_TAIL, score_only_messages  # noqa: E402
from ko_judge.splits import make_plan  # noqa: E402

pytestmark = pytest.mark.smoke

TEMPLATE = (
    "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
SPECIALS = ["<pad>", "<unk>", "<|end|>", "<|user|>", "<|assistant|>"]


def _tiny_model(path, corpus):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(corpus, trainers.WordLevelTrainer(special_tokens=SPECIALS))
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<|end|>"
    )
    hf_tok.chat_template = TEMPLATE
    hf_tok.save_pretrained(path)
    # kanana처럼 hidden_size가 head 수로 나눠떨어지지 않고 head_dim을 명시한 모양
    cfg = LlamaConfig(
        vocab_size=len(hf_tok),
        hidden_size=36,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        pad_token_id=hf_tok.pad_token_id,
        eos_token_id=hf_tok.eos_token_id,
    )
    LlamaForCausalLM(cfg).save_pretrained(path)


def test_train_merge_score(tmp_path, pw):
    from ko_judge import compat
    from ko_judge.generation import HFBackend
    from ko_judge.merge import merge
    from ko_judge.train_lora import build_dataset, make_trainer

    compat.apply()
    base = tmp_path / "tiny"
    corpus = [SCORE_TAIL, "1 2 3 4 5", *pw.judge_query]
    _tiny_model(base, corpus)

    plan = make_plan(pw, n_folds=5, n_holdout=3)
    rows = pw[plan.train_mask(pw, 0)].to_dict(orient="records")
    cfg = TrainConfig(
        model="qwen3-4b",
        output_dir=str(tmp_path / "out"),
        lora=LoraSection(r=4, alpha=8),
        train=TrainSection(
            per_device_batch_size=4,
            grad_accum=1,
            bf16=False,
            gradient_checkpointing=False,
            max_length=2048,
        ),
    )
    ds = build_dataset(cfg, rows, seed=0)
    assert len(ds) == len(rows)
    out = tmp_path / "out" / "fold0_seed0"
    trainer = make_trainer(cfg, str(base), None, ds, out, seed=0, max_steps=2, use_cpu=True)
    assert trainer.train().global_step == 2
    trainer.save_model(str(out))
    trainer.processing_class.save_pretrained(str(out))
    assert (out / "adapter_config.json").exists()

    merged = tmp_path / "merged"
    merge(str(base), None, str(out), str(merged))
    test = pw[plan.test_mask(pw, 0)].head(3)
    res = HFBackend(str(merged)).score([score_only_messages(q) for q in test.judge_query])
    assert len(res) == 3
    for r in res:
        assert set(r["dist"]) == {1, 2, 3, 4, 5}
        assert sum(r["dist"].values()) == pytest.approx(1.0)
