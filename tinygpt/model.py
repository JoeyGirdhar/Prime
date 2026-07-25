"""
A tiny GPT, built from scratch in PyTorch.

Everything that makes a transformer a transformer is here and commented:
token + positional embeddings, multi-head *causal* self-attention (implemented by
hand, not via nn.MultiheadAttention), a LayerNorm + MLP block, and a language-model
head. It's a stripped-down, readable cousin of Karpathy's nanoGPT.

The one idea to take away: self-attention lets every position look back at every
earlier position and decide what to pull in. That's what lets the model learn
long-range structure -- like closing a ring bond it opened 20 characters ago.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64      # max context length (how far back attention can see)
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask (no peeking at the future)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        # One linear projects the input into query, key, and value at once.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # Lower-triangular mask: position t may attend to positions <= t only.
        self.register_buffer(
            "mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                        .view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape                       # batch, time (tokens), channels (embd)
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        # Reshape into heads: (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Attention scores: how much each token attends to each earlier token.
        att = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)            # weights sum to 1 over the past
        att = self.attn_drop(att)

        y = att @ v                             # weighted sum of value vectors
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # recombine heads
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network (expand 4x, GELU, project back)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """A transformer block: attention and MLP, each with a residual connection."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual: add the layer's output back on
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # what each token means
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # where it sits in the sequence
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))   # combine meaning + position
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))                       # next-token scores

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
        """Autoregressively sample tokens one at a time."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]           # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature            # last position's scores
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_token is not None and next_id.item() == stop_token:
                break
        return idx
