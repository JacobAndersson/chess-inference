import math

import torch

from emsearch.config import ModelConfig
from emsearch.model import (
    MLP,
    CausalSelfAttention,
    ChessTransformer,
    SwiGLUMLP,
    TransformerBlock,
    estimate_params,
)


def test_causal_self_attention_shape():
    config = ModelConfig(d_model=64, n_heads=4, n_layers=2, dropout=0.0)
    attn = CausalSelfAttention(config)
    x = torch.randn(2, 10, 64)
    out = attn(x)
    assert out.shape == (2, 10, 64)


def test_attention_no_bias():
    config = ModelConfig(d_model=64, n_heads=4, n_layers=2, bias=False)
    attn = CausalSelfAttention(config)
    assert attn.qkv.bias is None
    assert attn.proj.bias is None


def test_attention_with_bias():
    config = ModelConfig(d_model=64, n_heads=4, n_layers=2, bias=True)
    attn = CausalSelfAttention(config)
    assert attn.qkv.bias is not None
    assert attn.proj.bias is not None


def test_mlp_gelu_shape():
    config = ModelConfig(d_model=64, n_heads=4, n_layers=2, use_swiglu=False)
    mlp = MLP(config)
    x = torch.randn(2, 10, 64)
    out = mlp(x)
    assert out.shape == (2, 10, 64)


def test_mlp_no_bias():
    config = ModelConfig(d_model=64, n_heads=4, n_layers=2, bias=False, use_swiglu=False)
    mlp = MLP(config)
    assert mlp.fc1.bias is None
    assert mlp.fc2.bias is None


def test_swiglu_mlp_shape():
    config = ModelConfig(d_model=64, n_heads=4, n_layers=2, use_swiglu=True)
    mlp = SwiGLUMLP(config)
    x = torch.randn(2, 10, 64)
    out = mlp(x)
    assert out.shape == (2, 10, 64)


def test_swiglu_no_bias():
    config = ModelConfig(d_model=64, n_heads=4, n_layers=2, bias=False, use_swiglu=True)
    mlp = SwiGLUMLP(config)
    assert mlp.gate.bias is None
    assert mlp.up.bias is None
    assert mlp.down.bias is None


def test_swiglu_hidden_dim_alignment():
    config = ModelConfig(d_model=256, n_heads=4, n_layers=2, use_swiglu=True)
    mlp = SwiGLUMLP(config)
    # 8/3 * 256 = 682.67, rounded up to nearest 256 = 768
    assert mlp.gate.out_features == 768


def test_transformer_block_uses_swiglu():
    config = ModelConfig(d_model=64, n_heads=4, n_layers=2, use_swiglu=True)
    block = TransformerBlock(config)
    assert isinstance(block.mlp, SwiGLUMLP)


def test_transformer_block_uses_gelu():
    config = ModelConfig(d_model=64, n_heads=4, n_layers=2, use_swiglu=False)
    block = TransformerBlock(config)
    assert isinstance(block.mlp, MLP)


def test_chess_transformer_forward():
    config = ModelConfig(vocab_size=32, d_model=64, n_heads=4, n_layers=2, max_seq_len=128)
    model = ChessTransformer(config)
    input_ids = torch.randint(0, 32, (2, 20))
    logits = model(input_ids)
    assert logits.shape == (2, 20, 32)


def test_chess_transformer_weight_tying():
    config = ModelConfig(vocab_size=32, d_model=64, n_heads=4, n_layers=2)
    model = ChessTransformer(config)
    assert model.tok_emb.weight is model.lm_head.weight


def test_residual_scaling():
    config = ModelConfig(vocab_size=32, d_model=64, n_heads=4, n_layers=6, use_swiglu=True)
    model = ChessTransformer(config)
    expected_std = 0.02 / math.sqrt(2 * 6)

    for block in model.blocks:
        proj_std = block.attn.proj.weight.std().item()
        assert abs(proj_std - expected_std) < 0.01, f"attn proj std {proj_std} != {expected_std}"

        down_std = block.mlp.down.weight.std().item()
        assert abs(down_std - expected_std) < 0.01, f"MLP down std {down_std} != {expected_std}"


def test_residual_scaling_gelu():
    config = ModelConfig(vocab_size=32, d_model=64, n_heads=4, n_layers=6, use_swiglu=False)
    model = ChessTransformer(config)
    expected_std = 0.02 / math.sqrt(2 * 6)

    for block in model.blocks:
        fc2_std = block.mlp.fc2.weight.std().item()
        assert abs(fc2_std - expected_std) < 0.01


def test_generate():
    config = ModelConfig(vocab_size=32, d_model=64, n_heads=4, n_layers=2, max_seq_len=64)
    model = ChessTransformer(config)
    model.eval()
    input_ids = torch.randint(0, 32, (1, 5))
    output = model.generate(input_ids, max_new_tokens=10, temperature=1.0)
    assert output.shape == (1, 15)


def test_count_parameters():
    config = ModelConfig(vocab_size=32, d_model=64, n_heads=4, n_layers=2)
    model = ChessTransformer(config)
    count = model.count_parameters()
    assert count > 0
    assert isinstance(count, int)


def test_estimate_params_swiglu():
    config = ModelConfig(vocab_size=32, d_model=256, n_heads=4, n_layers=6, use_swiglu=True)
    est = estimate_params(config)
    assert est > 0


def test_estimate_params_gelu():
    config = ModelConfig(vocab_size=32, d_model=256, n_heads=4, n_layers=6, use_swiglu=False)
    est = estimate_params(config)
    assert est > 0


def test_default_config_no_bias():
    config = ModelConfig()
    assert config.bias is False


def test_default_config_no_dropout():
    config = ModelConfig()
    assert config.dropout == 0.0


def test_default_config_swiglu():
    config = ModelConfig()
    assert config.use_swiglu is True
