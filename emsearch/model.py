"""GPT-2 style decoder-only transformer for chess move prediction."""

import math

import torch
from torch import nn
from torch.nn import functional as fn

from emsearch.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention."""
        b, t, c = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=2)

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        drop_p = self.dropout.p if self.training else 0.0
        out = fn.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop_p)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.dropout(self.proj(out))


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP with GELU activation."""
        x = fn.gelu(self.fc1(x))
        return self.dropout(self.fc2(x))


class SwiGLUMLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # 8/3 expansion, rounded to nearest multiple of 256 for efficiency
        hidden_dim = int(8 * config.d_model / 3)
        hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.gate = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.up = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.down = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU: down(silu(gate(x)) * up(x))."""
        return self.dropout(self.down(fn.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = SwiGLUMLP(config) if config.use_swiglu else MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block with residual connections."""
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ChessTransformer(nn.Module):
    """GPT-2 style decoder-only transformer for chess."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self) -> None:
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale residual projections (attn proj and MLP output)
        for block in self.blocks:
            torch.nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            if isinstance(block.mlp, SwiGLUMLP):
                torch.nn.init.normal_(block.mlp.down.weight, mean=0.0, std=residual_std)
            else:
                torch.nn.init.normal_(block.mlp.fc2.weight, mean=0.0, std=residual_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Optional mask (unused, kept for API compatibility)

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        _, t = input_ids.shape
        assert t <= self.config.max_seq_len

        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            idx_cond = (
                input_ids
                if input_ids.size(1) <= self.config.max_seq_len
                else input_ids[:, -self.config.max_seq_len :]
            )
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = fn.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def estimate_params(config: ModelConfig) -> int:
    """Estimate parameter count for a model config."""
    d = config.d_model
    v = config.vocab_size
    n_layers = config.n_layers
    s = config.max_seq_len

    embed_params = v * d + s * d
    attn_params = 4 * d * d
    if config.use_swiglu:
        hidden_dim = ((int(8 * d / 3) + 255) // 256) * 256
        mlp_params = 3 * d * hidden_dim
    else:
        mlp_params = 8 * d * d
    ln_params = 4 * d
    block_params = attn_params + mlp_params + ln_params

    return embed_params + n_layers * block_params + 2 * d
