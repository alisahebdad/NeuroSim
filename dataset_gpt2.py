"""
WikiText-2 loader for the GPT-2 CIM port.

`dataset.py` upstream only provides image datasets (CIFAR-10/100, ImageNet).
This module supplies the text equivalent, shaped so that the existing
calibration helpers in `quantize.py` work unmodified: every batch is a
`(inputs, targets)` pair of LongTensors, matching the `(images, labels)`
unpacking in `collect_stats()` and `evaluate()`.

Perplexity is evaluated over non-overlapping windows of `block_size` tokens.
That is the standard, cheapest protocol; a strided/sliding-window evaluation
gives slightly lower (better) perplexity but costs `block_size/stride` times
more forward passes, which is prohibitive under CIM simulation.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset


def _encode_split(split, block_size, limit_tokens=None):
    """Download a WikiText-2 split and tokenise it into (x, y) windows."""
    import tiktoken
    from datasets import load_dataset

    print(f"Loading WikiText-2 [{split}] ...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)

    # The raw dataset is one row per line; join into a single stream.
    text = "\n\n".join(ds["text"])

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(text)

    if limit_tokens is not None:
        ids = ids[:limit_tokens]

    # Need block_size+1 tokens per window: inputs are [0:T], targets [1:T+1].
    n_windows = (len(ids) - 1) // block_size
    if n_windows < 1:
        raise ValueError(
            f"split '{split}' yielded {len(ids)} tokens, too few for "
            f"block_size={block_size}"
        )

    usable = n_windows * block_size + 1
    buf = torch.tensor(ids[:usable], dtype=torch.long)

    x = buf[:-1].view(n_windows, block_size).contiguous()
    y = buf[1:].view(n_windows, block_size).contiguous()

    print(f"  {len(ids)} tokens -> {n_windows} windows of {block_size}")
    return TensorDataset(x, y)


def get_wikitext2(batch_size, block_size, split="test", limit_tokens=None,
                  shuffle=False, num_workers=0):
    """Return a DataLoader yielding (input_ids, target_ids).

    Args:
        batch_size:   sequences per batch. Keep small (1-4) under CIM
                      simulation -- simulate_array() runs
                      cycles_per_input * cells_per_weight iterations per layer.
        block_size:   sequence length T. Must be <= Config.block_size (1024).
        split:        'train' for calibration, 'test' for evaluation.
        limit_tokens: truncate the token stream; useful to keep calibration
                      cheap.
    """
    dataset = _encode_split(split, block_size, limit_tokens)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
