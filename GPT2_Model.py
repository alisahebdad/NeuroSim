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

    def __init__(self, **kwargs):
        # Allow overrides, e.g. Config(block_size=128) for CIM simulation.
        # Shorter sequences make the un-mappable attention matmuls a smaller
        # fraction of total compute, and keep simulate_array() tractable.
        for k, v in kwargs.items():
            if not hasattr(Config, k):
                raise AttributeError(f"unknown config field: {k}")
            setattr(self, k, v)


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

        # MAPPED TO CIM. self.c_attn is monkey-patched to CIMLinear by
        # quant_modules.initialize(), so calling it as a module runs the
        # crossbar simulation, emits a NetWork_gpt2.csv row, and dumps
        # input_*/weight_* traces on the first forward.
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

        # ------------------------------------------------------------------
        # TODO(neurosim-c++): QK^T is NOT mapped to CIM and never will be by
        # the current framework. Both operands are activations, so there is no
        # static weight to program into a crossbar. Because this is a bare
        # tensor op (not an nn.Module), CIMLinear's forward override never
        # sees it: no NetWork_gpt2.csv row, no trace file, no quantization.
        #
        # The C++ estimator therefore reports NOTHING for this operation. To
        # account for it, one of:
        #   (a) digital offload  - add an analytical area/latency/energy model
        #                          for a digital MAC unit of T*T*head_dim MACs
        #                          per head, and report it alongside NeuroSim's
        #                          numbers;
        #   (b) dynamic CIM      - write K into the array every token and model
        #                          the write cost (expensive for RRAM);
        #   (c) exclude          - state the exclusion explicitly in results.
        # See report.md 3.2. Cost share: ~2.7% of MACs at T=128, ~18% at T=1024.
        # ------------------------------------------------------------------
        att = (
            q @ k.transpose(-2, -1)
        ) * (1.0 / math.sqrt(head_dim))

        # TODO(neurosim-c++): causal masking has no analogue in NeuroSim. Swin
        # (the only transformer upstream supports) uses bidirectional windowed
        # attention. Masking is free here in software but would need a
        # peripheral model on real hardware.
        att = att.masked_fill(
            self.mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        # TODO(neurosim-c++): softmax is not modelled. NeuroSim's peripheral
        # circuit library covers ReLU and pooling only (see NeuroSIM/*.cpp:
        # no Softmax/Exp unit exists). Its exp() and division are non-trivial
        # in hardware and are currently unaccounted for in area and energy.
        att = F.softmax(att, dim=-1)

        # TODO(neurosim-c++): softmax(.)V - same situation as QK^T above.
        # Activation x activation, invisible to the CIM path.
        y = (
            att @ v
        ).transpose(1, 2).contiguous().view(
            B, T, C
        )

        # MAPPED TO CIM.
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

        # Both c_fc and c_proj are MAPPED TO CIM.
        #
        # TODO(neurosim-c++): F.gelu is not modelled. NeuroSim assumes ReLU in
        # its peripheral path. GELU is materially more expensive (erf/tanh
        # approximation) and its area/energy are unaccounted for.
        #
        # NOTE (quantization): GELU output feeds c_proj, and its distribution
        # has heavy outliers. This is the layer most likely to break the
        # per-tensor activation quantizer. If accuracy collapses, inspect the
        # amax of mlp.c_proj's input quantizer first. See report.md phase 6.
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

        # TODO(neurosim-c++): nn.LayerNorm is not intercepted (only Conv2d,
        # Linear and the pooling layers are in cim_quant_map) and has no C++
        # model. Two LayerNorms per block x 12 blocks + ln_f = 25 unaccounted
        # normalisation units, each requiring mean/variance over 768 elements.
        #
        # TODO(neurosim-c++): the residual adds are also unmodelled. They are
        # cheap, but they imply buffer traffic the C++ floorplanner does not
        # see because no NetWork_gpt2.csv row describes them.
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
        #
        # TODO(neurosim-c++): lm_head and wte share one weight tensor. A CIM
        # chip cannot share a physical crossbar between two layers, so on real
        # hardware this costs either a second 50257x768 array (~38.6M cells,
        # larger than all 12 blocks combined) or a dedicated digital path.
        # NeuroSim has no way to express sharing: if lm_head gets a
        # NetWork_gpt2.csv row, its area is counted as if it were separate.
        # Decide and document: counted separately, or excluded.
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, idx):

        B, T = idx.size()

        pos = torch.arange(
            0,
            T,
            dtype=torch.long,
            device=idx.device
        )

        # TODO(neurosim-c++): nn.Embedding is a table lookup, not a matmul, so
        # it is neither intercepted nor expressible as a NetWork_gpt2.csv row
        # (the format describes MAC layers). wte holds 38.6M parameters and
        # wpe 0.79M - together ~32% of GPT-2's 124M. Their storage cost is
        # real but invisible to the estimator. Model them as on-chip memory
        # separately, or state the exclusion.
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

    for i in range(100):

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