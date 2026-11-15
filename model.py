"""LM 1: a small GPT (decoder-only transformer) written from scratch.

Pre-norm blocks: x + attn(LN(x)), x + MLP(LN(x)); causal self-attention via PyTorch scaled_dot_product_attention;
learned positional embeddings; GELU MLP (4x); token embedding tied to the output head. ~20M parameters by default.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Config:
    vocab: int = 16384
    block: int = 512        # context length (tokens)
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__(); self.h = c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd); self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.drop = c.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, 2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.n_embd), nn.LayerNorm(c.n_embd)
        self.attn = Attention(c)
        self.mlp = nn.Sequential(nn.Linear(c.n_embd, 4 * c.n_embd), nn.GELU(), nn.Linear(4 * c.n_embd, c.n_embd))

    def forward(self, x):
        x = x + self.attn(self.ln1(x)); return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c=Config()):
        super().__init__(); self.c = c
        self.tok = nn.Embedding(c.vocab, c.n_embd); self.pos = nn.Embedding(c.block, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln = nn.LayerNorm(c.n_embd); self.head = nn.Linear(c.n_embd, c.vocab, bias=False)
        self.head.weight = self.tok.weight                                   # weight tying
        self.apply(self._init)
        for n, p in self.named_parameters():                                 # GPT-2 scaled init for residual projections
            if n.endswith("proj.weight") or n.endswith("mlp.2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks: x = b(x)
        logits = self.head(self.ln(x))
        loss = None if targets is None else F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8, top_k=50, eot=None):
        for _ in range(n):
            logits, _ = self(idx[:, -self.c.block:]); logits = logits[:, -1].float() / temperature
            v, _ = torch.topk(logits, top_k); logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1); idx = torch.cat([idx, nxt], 1)
            if eot is not None and nxt.item() == eot: break
        return idx


def count_params(m): return sum(p.numel() for p in m.parameters())
