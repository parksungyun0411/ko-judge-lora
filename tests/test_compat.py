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
