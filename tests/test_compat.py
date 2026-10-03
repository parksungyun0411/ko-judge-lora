import pytest

pytest.importorskip("transformers")

from transformers import LlamaConfig  # noqa: E402

from ko_judge import compat  # noqa: E402


def test_kanana_shape_loads_only_with_explicit_head_dim():
    assert compat.apply()
    # kanana-1.5-2.1b와 같은 모양: 1792 / 24 heads, head_dim 128 명시
    c = LlamaConfig(hidden_size=1792, num_attention_heads=24, num_key_value_heads=8, head_dim=128)
    assert c.head_dim == 128
    with pytest.raises(Exception, match="multiple"):
        LlamaConfig(hidden_size=30, num_attention_heads=4)


def test_warmup_float_is_ratio(tmp_path):
    """train_lora는 warmup_steps에 비율(0.05)을 넘긴다. 5.17 문서대로 비율로 해석되는지."""
    from transformers import TrainingArguments

    args = TrainingArguments(output_dir=str(tmp_path), warmup_steps=0.05, report_to="none")
    assert args.get_warmup_steps(200) == 10
    assert TrainingArguments(output_dir=str(tmp_path), warmup_steps=7).get_warmup_steps(200) == 7
