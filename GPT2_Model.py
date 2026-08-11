import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------- Model (pure PyTorch, GPT-2 small config) ----------
class Config:
    n_layer, n_head, n_embd = 12, 12, 768
    vocab_size, block_size = 50257, 1024

class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_attn = nn.Linear(c.n_embd, 3 * c.n_embd)   # Q,K,V combined
        self.c_proj = nn.Linear(c.n_embd, c.n_embd)
        self.n_head, self.n_embd = c.n_head, c.n_embd
        self.register_buffer("mask",
            torch.tril(torch.ones(c.block_size, c.block_size)).view(1, 1, c.block_size, c.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_fc   = nn.Linear(c.n_embd, 4 * c.n_embd)
        self.c_proj = nn.Linear(4 * c.n_embd, c.n_embd)
    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln_1, self.attn = nn.LayerNorm(c.n_embd), CausalSelfAttention(c)
        self.ln_2, self.mlp  = nn.LayerNorm(c.n_embd), MLP(c)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # pre-LN + residual
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(c.vocab_size, c.n_embd),
            wpe=nn.Embedding(c.block_size, c.n_embd),
            h=nn.ModuleList([Block(c) for _ in range(c.n_layer)]),
            ln_f=nn.LayerNorm(c.n_embd),
        ))
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # weight tying

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))

# ---------- Load pretrained weights from HuggingFace ----------
def load_gpt2():
    from transformers import GPT2LMHeadModel
    model = GPT2(Config())
    sd = model.state_dict()
    hf = GPT2LMHeadModel.from_pretrained("gpt2").state_dict()
    # HF stores some Linear weights transposed (Conv1D); fix those
    transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
    for k in sd:
        hk = k
        if hk not in hf: continue
        if any(k.endswith(t) for t in transposed):
            sd[k].copy_(hf[hk].t())
        else:
            sd[k].copy_(hf[hk])
    return model

# ---------- Generate ----------
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40):
    for _ in range(max_new_tokens):
        logits = model(idx[:, -1024:])[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx

if __name__ == "__main__":
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    model = load_gpt2().eval()

    prompt = "The meaning of life is"
    ids = torch.tensor([enc.encode(prompt)])
    out = generate(model, ids, max_new_tokens=40)
    print(enc.decode(out[0].tolist()))
