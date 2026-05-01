"""
µP (Maximal Update Parameterization) GPT Model for SVG code generation.

Adapted from model.py (standard parameterization) with the following µP changes:
  1. Output head uses MuReadout / MuSharedReadout instead of nn.Linear
  2. Attention scaling changed from 1/sqrt(d) to 1/d
  3. Query weights initialized to zero
  4. Hidden weights use fan-in scaled initialization
  5. Optimizer uses MuAdamW for automatic per-layer LR scaling
"""

import math
import inspect
from dataclasses import dataclass
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F

from mup import MuReadout, MuSharedReadout, set_base_shapes, make_base_shapes
from mup.optim import MuAdamW

from model import GPTConfig, MODEL_CONFIGS, LayerNorm


class MuCausalSelfAttention(nn.Module):
    """Causal self-attention with µP-correct 1/d scaling (instead of 1/sqrt(d))."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Separate Q, K, V projections (needed for zero-init of Q)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # µP attention scaling: 8/d instead of 1/sqrt(d)
        # Use PyTorch's SDPA with custom scale parameter (enables Flash Attention)
        attn_scale = 8.0 / self.head_dim
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
            scale=attn_scale,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MuMLP(nn.Module):
    """MLP with explicit d_ff (same structure as SP, init handled by mup)."""

    def __init__(self, config):
        super().__init__()
        d_ff = config.d_ff if hasattr(config, 'd_ff') and config.d_ff is not None else 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, d_ff, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(d_ff, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class MuBlock(nn.Module):
    """Transformer block using µP attention."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MuCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MuMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MuGPT(nn.Module):
    """GPT model with µP parameterization."""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([MuBlock(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        # µP: Use MuSharedReadout for weight-tied output head
        self.lm_head = MuSharedReadout(self.transformer.wte.weight, bias=False)

    def _init_weights_mup(self):
        """µP-correct initialization after set_base_shapes is called."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Fan-in scaled init for hidden weights
                fan_in = module.weight.shape[1]
                std = 1.0 / math.sqrt(fan_in)
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Zero-init query weights (µP requirement for stable attention at init)
        for block in self.transformer.h:
            nn.init.zeros_(block.attn.q_proj.weight)
            if block.attn.q_proj.bias is not None:
                nn.init.zeros_(block.attn.q_proj.bias)

        # Zero-init output projection weights (µP readout recommendation)
        # MuSharedReadout uses the embedding weights, no separate init needed

        # Scale residual projections
        for block in self.transformer.h:
            std = 1.0 / math.sqrt(block.attn.c_proj.weight.shape[1])
            std = std / math.sqrt(2 * self.config.n_layer)
            nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=std)
            std_mlp = 1.0 / math.sqrt(block.mlp.c_proj.weight.shape[1])
            std_mlp = std_mlp / math.sqrt(2 * self.config.n_layer)
            nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=std_mlp)

    def get_num_params(self, non_embedding=True):
        """Return number of parameters (excluding position embeddings by default)."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def create_mup_model(model_name):
    """Create a µP model with proper base shape setup.

    Args:
        model_name: One of 'tiny', 'small', 'medium', 'large', 'xl'

    Returns:
        MuGPT model with base shapes set and µP-correct initialization
    """
    target_config = MODEL_CONFIGS[model_name]

    # Base width must be divisible by n_head. Use n_head as the
    # smallest valid width to minimize base model size.
    n_head = target_config.n_head
    base_width = n_head  # Guarantees n_embd % n_head == 0
    delta_width = n_head * 2

    # Base model: smallest width (proxy for HP tuning)
    base_config = GPTConfig(
        n_layer=target_config.n_layer,  # Same depth!
        n_head=target_config.n_head,
        n_embd=base_width,
        d_ff=base_width * 4,  # Scale d_ff proportionally
        block_size=target_config.block_size,
        vocab_size=target_config.vocab_size,
        dropout=target_config.dropout,
        bias=target_config.bias,
    )

    # Delta model: slightly wider (for mup to infer scaling dimensions)
    delta_config = GPTConfig(
        n_layer=target_config.n_layer,  # Same depth!
        n_head=target_config.n_head,
        n_embd=delta_width,
        d_ff=delta_width * 4,
        block_size=target_config.block_size,
        vocab_size=target_config.vocab_size,
        dropout=target_config.dropout,
        bias=target_config.bias,
    )

    # Create all three models
    base_model = MuGPT(base_config)
    delta_model = MuGPT(delta_config)
    model = MuGPT(target_config)

    # Set base shapes — this is the core µP operation
    # It tells mup which dimensions are "infinite" (will be scaled)
    set_base_shapes(model, base_model, delta=delta_model)

    # Apply µP-correct initialization AFTER set_base_shapes
    model._init_weights_mup()

    # Clean up base/delta (not needed for training)
    del base_model, delta_model

    return model
