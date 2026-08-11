import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Device selection: CUDA > MPS > CPU
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ============================================================
# Model configuration
# ============================================================

class Config:
    n_layer = 12
    n_head = 12
    n_embd = 768
    vocab_size = 50257
    block_size = 1024


# ============================================================
# Causal Self Attention
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, c):
        super().__init__()

        self.c_attn = nn.Linear(
            c.n_embd,
            3 * c.n_embd
        )

        self.c_proj = nn.Linear(
            c.n_embd,
            c.n_embd
        )

        self.n_head = c.n_head
        self.n_embd = c.n_embd

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    c.block_size,
                    c.block_size
                )
            ).view(
                1,
                1,
                c.block_size,
                c.block_size
            )
        )

    def forward(self, x):

        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(
            self.n_embd,
            dim=2
        )

        head_dim = C // self.n_head

        k = k.view(
            B, T, self.n_head, head_dim
        ).transpose(1, 2)

        q = q.view(
            B, T, self.n_head, head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T, self.n_head, head_dim
        ).transpose(1, 2)

        att = (
            q @ k.transpose(-2, -1)
        ) * (1.0 / math.sqrt(head_dim))

        att = att.masked_fill(
            self.mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        att = F.softmax(att, dim=-1)

        y = (
            att @ v
        ).transpose(1, 2).contiguous().view(
            B, T, C
        )

        return self.c_proj(y)


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):

    def __init__(self, c):
        super().__init__()

        self.c_fc = nn.Linear(
            c.n_embd,
            4 * c.n_embd
        )

        self.c_proj = nn.Linear(
            4 * c.n_embd,
            c.n_embd
        )

    def forward(self, x):

        return self.c_proj(
            F.gelu(
                self.c_fc(x)
            )
        )


# ============================================================
# Transformer Block
# ============================================================

class Block(nn.Module):

    def __init__(self, c):
        super().__init__()

        self.ln_1 = nn.LayerNorm(c.n_embd)
        self.attn = CausalSelfAttention(c)

        self.ln_2 = nn.LayerNorm(c.n_embd)
        self.mlp = MLP(c)

    def forward(self, x):

        x = x + self.attn(
            self.ln_1(x)
        )

        x = x + self.mlp(
            self.ln_2(x)
        )

        return x


# ============================================================
# GPT-2
# ============================================================

class GPT2(nn.Module):

    def __init__(self, c):
        super().__init__()

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(
                c.vocab_size,
                c.n_embd
            ),

            "wpe": nn.Embedding(
                c.block_size,
                c.n_embd
            ),

            "h": nn.ModuleList([
                Block(c)
                for _ in range(c.n_layer)
            ]),

            "ln_f": nn.LayerNorm(
                c.n_embd
            ),
        })

        self.lm_head = nn.Linear(
            c.n_embd,
            c.vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, idx):

        B, T = idx.size()

        pos = torch.arange(
            0,
            T,
            dtype=torch.long,
            device=idx.device
        )

        x = (
            self.transformer.wte(idx)
            +
            self.transformer.wpe(pos)
        )

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        return self.lm_head(x)


# ============================================================
# Load pretrained GPT-2 weights from Hugging Face
# ============================================================

def load_gpt2():

    from transformers import GPT2LMHeadModel

    print("Loading Hugging Face GPT-2...")

    model = GPT2(Config())

    sd = model.state_dict()

    hf = GPT2LMHeadModel.from_pretrained(
        "gpt2"
    ).state_dict()

    # HuggingFace GPT-2 uses Conv1D for these layers,
    # so their weights need to be transposed.
    transposed = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    for k in sd:

        hk = k

        if hk not in hf:
            continue

        if any(k.endswith(t) for t in transposed):
            sd[k].copy_(hf[hk].t())
        else:
            sd[k].copy_(hf[hk])

    model.load_state_dict(sd)

    return model


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    temperature=0.8,
    top_k=40
):

    for _ in range(max_new_tokens):

        # Keep context within GPT-2's maximum context length
        idx_cond = idx[:, -1024:]

        logits = model(idx_cond)

        logits = logits[:, -1, :]

        logits = logits / temperature

        if top_k is not None:

            v, _ = torch.topk(
                logits,
                min(top_k, logits.size(-1))
            )

            logits[
                logits < v[:, [-1]]
            ] = float("-inf")

        probs = F.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        idx = torch.cat(
            [idx, next_token],
            dim=1
        )

    return idx


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    # Load model
    model = load_gpt2().eval()

    # Move model to CUDA / MPS / CPU
    model = model.to(device)

    print(f"Model is running on: {device}")

    prompt = "The meaning of life is"

    for i in range(10):

        # Encode prompt
        ids = torch.tensor(
            [enc.encode(prompt)],
            dtype=torch.long,
            device=device
        )

        # Generate
        out = generate(
            model,
            ids,
            max_new_tokens=40
        )

        text = enc.decode(
            out[0].tolist()
        )

        print(f"\n{text}")

        # Use the last 10 characters as the next prompt
        prompt = text[-10:]