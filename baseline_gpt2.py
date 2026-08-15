"""
Pure-PyTorch GPT-2 baseline -- NO CIM, NO quantization, NO NeuroSim.

Ground truth for `inference_gpt2.py`. This script deliberately does NOT import
pytorch_quantization, so `quant_modules.initialize()` never runs and
`torch.nn.Linear` is never monkey-patched. What runs here is plain FP32
PyTorch.

Its job is to answer one question: are the weights loaded correctly and is the
evaluation protocol sound? Every number the CIM pipeline produces is only
interpretable relative to this one.

REFERENCE VALUES -- GPT-2 small (124M) on WikiText-2 raw, non-overlapping
windows:

    T=1024   perplexity ~30       top-1 ~35-40%
    T=512    perplexity ~35
    T=128    perplexity ~45-60    (shorter context => higher perplexity;
                                   each window restarts with no history)

If you see perplexity in the hundreds, the weights are NOT loaded correctly.
Do not proceed to CIM simulation until this script is in range -- run with
--sample to print generated text, which fails obviously and immediately when
weight loading is broken.

Usage:
    python baseline_gpt2.py --block_size 128 --num_batches 8      # match CIM run
    python baseline_gpt2.py --block_size 1024 --num_batches -1    # full, publishable
    python baseline_gpt2.py --sample                              # weight sanity check
"""

import time
import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset_gpt2 import get_wikitext2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--block_size', type=int, default=1024,
                   help='sequence length T (<= 1024)')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_batches', type=int, default=-1,
                   help='-1 for the full test split')
    p.add_argument('--split', type=str, default='test',
                   help='test | validation | train')
    p.add_argument('--device', type=str, default=None,
                   help='cuda | mps | cpu (auto-detected if unset)')
    p.add_argument('--sample', action='store_true',
                   help='generate text as a weight-loading sanity check')
    return p.parse_args()


def pick_device(requested):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@torch.no_grad()
def evaluate(model, loader, device, num_batches):
    """Perplexity and next-token top-1 over non-overlapping windows.

    Identical protocol to evaluate_lm() in inference_gpt2.py, so the two are
    directly comparable. Loss is summed over tokens and divided once at the
    end -- averaging per-batch means would weight a short final batch equally
    with a full one.
    """
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    total_correct = 0
    start = time.time()

    for i, (x, y) in tqdm(enumerate(loader), total=num_batches, desc="eval"):
        if num_batches > 0 and i >= num_batches:
            break

        x = x.to(device)
        y = y.to(device)

        logits = model(x)                       # (B, T, vocab)

        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y.reshape(-1),
            reduction='sum')

        total_nll += nll.item()
        total_tokens += y.numel()
        total_correct += (logits.argmax(dim=-1) == y).sum().item()

    mean_nll = total_nll / total_tokens
    ppl = float(torch.exp(torch.tensor(mean_nll)))
    top1 = 100.0 * total_correct / total_tokens
    return ppl, top1, mean_nll, total_tokens, time.time() - start


@torch.no_grad()
def sanity_sample(model, device):
    """Generate text. Broken weights produce obvious gibberish."""
    import tiktoken
    from GPT2_Model import generate

    enc = tiktoken.get_encoding("gpt2")
    prompt = "The capital city of France is"
    ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)

    out = generate(model, ids, max_new_tokens=30, temperature=0.7, top_k=40)
    print("\n--- generated sample ---")
    print(enc.decode(out[0].tolist()))
    print("------------------------")
    print("Coherent English => weights loaded correctly.")
    print("Word salad       => load_gpt2() is broken; fix before anything else.\n")


def main():
    args = parse_args()
    device = pick_device(args.device)

    print("=" * 62)
    print("GPT-2 BASELINE  (pure FP32 PyTorch -- no CIM, no quantization)")
    print("=" * 62)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"  device: {device}\n")

    # Imported here, AFTER we have confirmed nothing patched torch.nn.
    from GPT2_Model import load_gpt2

    model = load_gpt2().eval().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_unique = sum(p.numel() for p in
                   {id(p): p for p in model.parameters()}.values())
    print(f"\nParameters: {n_params/1e6:.1f}M "
          f"({n_unique/1e6:.1f}M unique -- lm_head is tied to wte)")

    # A sanity check that costs nothing: an untrained or badly loaded model
    # has near-zero weight magnitude variation across blocks.
    w = model.transformer.h[0].mlp.c_fc.weight
    print(f"h.0.mlp.c_fc.weight: mean={w.mean():.5f} std={w.std():.5f} "
          f"(expect std ~0.1-0.15 for pretrained GPT-2)")

    if args.sample:
        sanity_sample(model, device)

    loader = get_wikitext2(args.batch_size, args.block_size,
                           split=args.split)
    num_batches = len(loader) if args.num_batches == -1 else args.num_batches

    ppl, top1, loss, tokens, elapsed = evaluate(model, loader, device,
                                                num_batches)

    print("\n" + "=" * 62)
    print(f"{'WikiText-2 [' + args.split + ']':<24}{'value':>18}")
    print("-" * 62)
    print(f"{'tokens evaluated':<24}{tokens:>18,}")
    print(f"{'sequence length T':<24}{args.block_size:>18}")
    print(f"{'cross-entropy loss':<24}{loss:>18.4f}")
    print(f"{'perplexity':<24}{ppl:>18.3f}")
    print(f"{'next-token top-1 %':<24}{top1:>18.2f}")
    print(f"{'elapsed (s)':<24}{elapsed:>18.1f}")
    print("=" * 62)

    # Interpretation, so a wrong number cannot be mistaken for a right one.
    if ppl > 150:
        print("\n*** PERPLEXITY IS FAR TOO HIGH ***")
        print("GPT-2 small should score well under 100 at any T. Something is")
        print("wrong with weight loading or the evaluation protocol -- not")
        print("with quantization, which is not active in this script.")
        print("Run with --sample to check weight loading directly.")
    elif ppl > 100:
        print("\nHigher than expected. Plausible at very short T, but verify")
        print("with --block_size 1024 before trusting it.")
    else:
        print("\nIn the expected range. This is a valid reference point for")
        print("the CIM results in inference_gpt2.py.")

    print("\nNOTE: compare against inference_gpt2.py using the SAME "
          "--block_size and --num_batches. Perplexity is strongly dependent "
          "on T, so runs at different T are not comparable.")


if __name__ == '__main__':
    main()
