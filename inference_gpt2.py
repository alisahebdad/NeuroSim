"""
GPT-2 inference with DNN+NeuroSim CIM functional simulation -- PYTHON SIDE ONLY.

Mirrors `inference.py`, but for a decoder-only language model. It produces the
ACCURACY half of the framework's output (perplexity and next-token top-1) and
prepares -- but deliberately does not run -- the C++ hardware estimator.

Every place where data crosses to the C++ side is marked `TODO(neurosim-c++)`.

WHAT IS SIMULATED ON CIM
    Per block: attn.c_attn (768->2304), attn.c_proj (768->768),
               mlp.c_fc (768->3072), mlp.c_proj (3072->768)
    x12 blocks = 48 layers, 85M of GPT-2's 124M parameters.

WHAT IS NOT (see TODOs in GPT2_Model.py)
    QK^T and softmax(.)V   - activation x activation, no static weight
    LayerNorm, GELU, Softmax - no NeuroSim peripheral model
    wte / wpe embeddings   - table lookups, not MACs
    lm_head                - excluded by default (weight-tied to wte)

Usage:
    python inference_gpt2.py --block_size 128 --batch_size 1 \
        --hardware 1 --bitcell 1 --sub_array "[128,128]" \
        --mem_type resistive --mem_states_file mem_states.csv --adc_precision 7
"""

import os
import ast
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pytorch_quantization.utils import misc, hook
import pytorch_quantization.cim.modules.macro as macro
import pytorch_quantization.nn as quant_nn
import pytorch_quantization.quant_modules as quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import cim

from quantize import cim_quant_map, collect_stats, compute_amax
from dataset_gpt2 import get_wikitext2

MODEL_NAME = "gpt2"


def parse_args():
    p = argparse.ArgumentParser()

    # ── model / data ─────────────────────────────────────────────────────
    p.add_argument('--model', default=MODEL_NAME)
    p.add_argument('--block_size', type=int, default=128,
                   help='sequence length T. Shorter T keeps the un-mappable '
                        'attention matmuls a small share of total MACs '
                        '(~2.7%% at 128, ~18%% at 1024) and keeps '
                        'simulate_array() tractable.')
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--calib_batch_size', type=int, default=1)
    p.add_argument('--calib_tokens', type=int, default=100_000,
                   help='truncate the calibration stream')
    p.add_argument('--num_batches', type=int, default=8,
                   help='evaluation batches; -1 for the full test split')
    p.add_argument('--gpu', default=0, type=int)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--logdir', default='log/gpt2')
    p.add_argument('--test_name', default='test')

    # ── which layers go on CIM ───────────────────────────────────────────
    p.add_argument('--quant_lm_head', type=int, default=0,
                   help='0 = keep lm_head in FP. It is 768->50257, which is '
                        'both weight-tied to wte (unmappable, see '
                        'GPT2_Model.py) and enormous: at 8-bit weights it '
                        'needs ceil(50257*8/128) = 3141 subarrays. Enabling '
                        'it will likely OOM.')
    p.add_argument('--skip_first_layer', type=int, default=0,
                   help='upstream quantize.py hardcodes layers[1:]. Off by '
                        'default here so the mapping matches the report.')

    # ── hardware (identical semantics to inference.py) ───────────────────
    p.add_argument('--hardware', type=int, default=1)
    p.add_argument('--ppa', type=int, default=0,
                   help='left at 0: the C++ estimator is out of scope for '
                        'this Python-side port. See TODO block in main().')
    p.add_argument('--sub_array', type=str, default="[128, 128]")
    p.add_argument('--parallel_read', type=int, default=128)
    p.add_argument('--weight_precision', type=int, default=8)
    p.add_argument('--input_precision', type=int, default=8)
    p.add_argument('--dac_precision', type=int, default=1)
    p.add_argument('--adc_precision', type=int, default=7)
    p.add_argument('--bitcell', type=int, default=1)
    p.add_argument('--mem_type', type=str, default='resistive')
    p.add_argument('--off_state', type=float, default=6e-3)
    p.add_argument('--on_state', type=float, default=6e-13 * 17)
    p.add_argument('--mem_states_file', type=str, default="mem_states.csv")
    p.add_argument('--vdd', type=float, default=1)
    p.add_argument('--read_noise', type=float, default=0.0)
    p.add_argument('--output_noise', type=float, default=0.0)
    p.add_argument('--output_noise_file', type=str, default="")

    # ── reliability ──────────────────────────────────────────────────────
    p.add_argument('--t', type=float, default=1)
    p.add_argument('--v', type=float, default=0.00)
    p.add_argument('--v_list', type=str, default="[0.005]*8")
    p.add_argument('--detect', type=int, default=0)
    p.add_argument('--target', type=float, default=0.0)
    p.add_argument('--rate_stuck_0', type=float, default=0.00)
    p.add_argument('--rate_stuck_1', type=float, default=0.00)

    # ── TensorRT quantization ────────────────────────────────────────────
    p.add_argument('--mode', default='TensorRT')
    p.add_argument('--input_calib_method', type=str, default='max')
    p.add_argument('--weight_calib_method', type=str, default='max')
    p.add_argument('--adc_calib_method', type=str, default='max')
    p.add_argument('--input_axis', type=int, default=None)
    p.add_argument('--weight_axis', type=int, default=0)
    p.add_argument('--adc_axis', type=int, default=None)
    p.add_argument('--adc_quant_method', type=str, default='scale')
    p.add_argument('--optimize_adc', type=int, default=0)
    p.add_argument('--adc_enable', type=int, default=1)

    args = p.parse_args()
    args.sub_array = ast.literal_eval(args.sub_array)
    args.v_list = [0.005] * 8 if args.v_list.startswith("[0.005]*") \
        else ast.literal_eval(args.v_list)
    return args


# =========================================================================
# TODO(neurosim-c++): NETWORK DESCRIPTION FILE
#
# This is handoff #1 of 3 to the C++ estimator. `./NeuroSIM/main` reads
# NetWork_gpt2.csv to learn each layer's geometry and build the tile
# floorplan.
#
# WHY THIS IS WRITTEN BY HAND instead of letting CIMLinear.forward emit it:
# cim_linear.py:125 has a rank-3 branch added for Swin --
#
#     if len(input.shape) > 2:
#         f.write(f'{input.shape[1]},{input.shape[2]},{self.in_features},...')
#
# Swin's Linear input is rank-4 (B, H, W, C), so shape[1],shape[2] = H,W,
# correctly giving the token grid. GPT-2's input is rank-3 (B, T, C), so
# shape[1],shape[2] = T,C -- and the row claims IFM_row=T, IFM_col=768,
# i.e. T*768 tokens instead of T. That inflates the mapped workload by 768x.
#
# Verify this before trusting any C++ output, then either fix cim_linear.py
# upstream-style or keep generating the file here. See report.md 3.2.
# =========================================================================
def write_network_csv(args, cfg):
    """Emit NeuroSIM/NetWork_gpt2.csv in forward-execution order.

    Row format (8 fields), per report.md 2:
        IFM_row, IFM_col, IFM_channel, K_row, K_col, Out_channel, pool, speedup

    A fully-connected layer is a 1x1 kernel. Token count goes in
    (IFM_row, IFM_col) = (T, 1), so IFM_row*IFM_col = T tokens.
    """
    path = f'./NeuroSIM/NetWork_{MODEL_NAME}.csv'
    os.makedirs('./NeuroSIM', exist_ok=True)

    T = args.block_size
    d = cfg.n_embd
    rows = []

    for _ in range(cfg.n_layer):
        rows.append((T, 1, d,     1, 1, 3 * d, 0, 1))   # attn.c_attn
        rows.append((T, 1, d,     1, 1, d,     0, 1))   # attn.c_proj
        rows.append((T, 1, d,     1, 1, 4 * d, 0, 1))   # mlp.c_fc
        rows.append((T, 1, 4 * d, 1, 1, d,     0, 1))   # mlp.c_proj

    if args.quant_lm_head:
        rows.append((T, 1, d, 1, 1, cfg.vocab_size, 0, 1))

    with open(path, 'w') as f:
        for r in rows:
            f.write(','.join(str(v) for v in r) + '\n')

    print(f"Wrote {len(rows)} layer rows to {path}")

    # TODO(neurosim-c++): the C++ side must also be configured by hand.
    # NeuroSIM/Param.cpp:99 has:
    #     novelMapping = true;   // false: conventional mapping (for swin_t)
    # Transformers require conventional mapping. There is no CLI flag for
    # this -- it is a source edit and a silent source of wrong numbers.
    print("REMINDER: set novelMapping = false in NeuroSIM/Param.cpp:99 "
          "before running the C++ estimator.")
    return path


def build_model(args):
    """Construct GPT-2 with nn.Linear monkey-patched to CIMLinear."""

    # ORDER IS CRITICAL. quant_modules.initialize() rebinds the attribute
    # torch.nn.Linear to cim.CIMLinear. Only models constructed AFTER this
    # call get CIM layers -- importing GPT2_Model earlier is fine, but
    # instantiating GPT2() earlier is not.
    quant_modules.initialize(
        float_module_list=['Conv2d', 'Linear', 'AvgPool2d',
                           'MaxPool2d', 'AdaptiveAvgPool2d'],
        custom_quant_modules=cim_quant_map)

    if args.input_calib_method == 'histogram':
        args.input_axis = None
    if args.weight_calib_method == 'histogram':
        args.weight_axis = None
    if args.adc_calib_method == 'histogram':
        args.adc_axis = None

    args.fake_quant = True

    input_desc = QuantDescriptor(calib_method=args.input_calib_method,
                                 num_bits=args.input_precision,
                                 fake_quant=True, axis=args.input_axis,
                                 unsigned=False)
    weight_desc = QuantDescriptor(calib_method=args.weight_calib_method,
                                  num_bits=args.weight_precision,
                                  fake_quant=True, axis=args.weight_axis,
                                  unsigned=False)
    adc_desc = QuantDescriptor(calib_method=args.adc_calib_method,
                               num_bits=args.adc_precision,
                               fake_quant=True, axis=args.adc_axis,
                               unsigned=True)

    # Class-level defaults: the model constructor passes no kwargs, so this
    # is the only channel by which config reaches each layer.
    cim.CIMLinear.set_default_quant_desc_input(input_desc)
    cim.CIMLinear.set_default_quant_desc_weight(weight_desc)
    cim.CIMLinear.set_default_quant_desc_adc(adc_desc)
    cim.CIMLinear.set_default_cim_args(args)

    # GPT-2 has no Conv2d or pooling, but the classes are patched in, so give
    # them valid defaults in case a future variant introduces them.
    cim.CIMConv2d.set_default_quant_desc_input(input_desc)
    cim.CIMConv2d.set_default_quant_desc_weight(weight_desc)
    cim.CIMConv2d.set_default_quant_desc_adc(adc_desc)
    cim.CIMConv2d.set_default_cim_args(args)

    from GPT2_Model import Config, load_gpt2

    cfg = Config(block_size=args.block_size)
    print(f"Building GPT-2 (n_layer={cfg.n_layer}, n_embd={cfg.n_embd}, "
          f"T={cfg.block_size}) with CIMLinear layers...")

    # load_gpt2() constructs GPT2(Config()) internally, which now yields
    # CIMLinear submodules, then copies HuggingFace weights into them.
    # This works because CIMLinear subclasses nn.Linear, so `weight` and
    # `bias` are genuine nn.Parameters and load_state_dict is unaffected.
    model = load_gpt2()
    model.config = cfg
    return model, cfg


@torch.no_grad()
def evaluate_lm(model, args, data_loader, num_batches, tag=""):
    """Perplexity and next-token top-1 accuracy.

    The classification `evaluate()` in quantize.py does not apply: logits are
    rank-3 (B, T, vocab) and there is one prediction per position, not one
    per sample.
    """
    device = torch.device(f"cuda:{args.gpu}")
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    total_correct = 0
    start = time.time()

    for i, (x, y) in tqdm(enumerate(data_loader), total=num_batches,
                          desc=f"eval {tag}"):
        if i >= num_batches:
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

    print(f"\n[{tag}] tokens={total_tokens}  loss={mean_nll:.4f}  "
          f"ppl={ppl:.3f}  top1={top1:.2f}%  ({time.time()-start:.1f}s)")
    return ppl, top1


def main():
    args = parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(args.seed)

    args.logdir = os.path.join(args.logdir, args.test_name)
    misc.ensure_dir(args.logdir)
    misc.logger.init(args.logdir,
                     'test_log_' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    args.logger = misc.logger.info

    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print(f'{k}: {v}')
    print("========================================\n")

    # ── data ─────────────────────────────────────────────────────────────
    loader_calib = get_wikitext2(args.calib_batch_size, args.block_size,
                                 split='train',
                                 limit_tokens=args.calib_tokens, shuffle=False)
    loader_test = get_wikitext2(args.batch_size, args.block_size,
                                split='test')
    if args.num_batches == -1:
        args.num_batches = len(loader_test)

    # =====================================================================
    # TODO(neurosim-c++): TRACE DIRECTORY + COMMAND SCRIPT
    #
    # Handoff #2 of 3. make_records() creates layer_record_gpt2/ and writes
    # the first line of trace_command.sh:
    #
    #     ./NeuroSIM/main ./NeuroSIM/NetWork_gpt2.csv <wp> <ip> <sub> <par>
    #
    # Each CIMLinear then appends its own "weight_<name>.csv input_<name>.csv"
    # pair on its FIRST forward (macro.py:25 -> write_layer, one-shot per
    # module because _cim_args is deepcopy'd per layer).
    #
    # CONSEQUENCE FOR GPT-2: only the 48 projection layers append traces.
    # QK^T and softmax(.)V produce no files, so the resulting command line
    # describes a network with no attention. If NetWork_gpt2.csv rows and
    # trace file pairs ever disagree in count or order, main() will silently
    # mis-associate them -- check both before running the estimator.
    # =====================================================================
    hook.make_records(args)
    args.hook = True

    # We generate the network description ourselves (see write_network_csv),
    # so suppress the buggy automatic emission in cim_linear.py.
    args.write_network = False

    # ── build ────────────────────────────────────────────────────────────
    hardware = args.hardware
    args.hardware = False           # non-idealities off during calibration

    model, cfg = build_model(args)
    write_network_csv(args, cfg)

    model = model.to(device)

    # ── select which layers run on CIM ───────────────────────────────────
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, macro.CIM):
            module._cim_args.quant_mode = 'iw'
            module._cim_args.name = name
            module._cim_args.batch_size = args.batch_size
            layers.append(name)

    layer_quant = list(layers)
    if not args.quant_lm_head:
        layer_quant = [n for n in layer_quant if 'lm_head' not in n]
    if args.skip_first_layer:
        layer_quant = layer_quant[1:]

    print(f"\n{len(layers)} CIM-capable layers found, "
          f"{len(layer_quant)} selected for quantization.")
    excluded = [n for n in layers if n not in layer_quant]
    if excluded:
        print(f"Excluded (kept in FP): {excluded}")

    # Suppress trace dumping for excluded layers. Two reasons:
    #
    #  1. Correctness. NetWork_gpt2.csv has no row for them, and
    #     ./NeuroSIM/main pairs rows to trace files POSITIONALLY. A stray
    #     trace would shift every subsequent layer's association.
    #
    #  2. Size. cim_linear.py:141 still calls self.linear() on disabled
    #     layers purely to dump traces. For lm_head that means np.savetxt on
    #     a 769 x 50258 integer matrix -- roughly 39M values, hundreds of MB
    #     of CSV text, per run.
    for name, module in model.named_modules():
        if isinstance(module, macro.CIM) and name not in layer_quant:
            module._cim_args.hook = False

    # ── baseline (FP) ────────────────────────────────────────────────────
    print("\nEvaluating FP baseline...")
    fp_ppl, fp_top1 = evaluate_lm(model, args, loader_test,
                                  num_batches=args.num_batches, tag="fp32")

    # ── calibrate input/weight quantizers ────────────────────────────────
    print("\nCollecting activation statistics...")
    collect_stats(model, layers, loader_calib, args.gpu,
                  quant_mode='iw', num_batches=2)

    print("\nComputing amax...")
    # strict=False: some quantizers never see data (e.g. an excluded
    # lm_head), and a strict load would raise on their empty calibrators.
    compute_amax(model, args, quant_mode='iw', layers=layers,
                 method="percentile", percentile=99.9999, strict=False)

    print("\nEvaluating input/weight-quantized model...")
    iw_ppl, iw_top1 = evaluate_lm(model, args, loader_test,
                                  num_batches=args.num_batches, tag="int8-iw")

    # ── arm the full CIM path ────────────────────────────────────────────
    for name, module in model.named_modules():
        if isinstance(module, macro.CIM):
            module._cim_args.quant_mode = 'adc'
            module._cim_args.hardware = hardware
        layer_name = name.split('._')[0]
        if layer_name not in layer_quant:
            if isinstance(module, quant_nn.TensorQuantizer):
                module.disable()
    args.hardware = hardware

    print("\nRunning CIM hardware simulation "
          f"(hardware={hardware}, adc={args.adc_precision}b, "
          f"cell={args.bitcell}b, array={args.sub_array})...")
    cim_ppl, cim_top1 = evaluate_lm(model, args, loader_test,
                                    num_batches=args.num_batches, tag="cim")

    # ── report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"{'stage':<24}{'perplexity':>16}{'top-1 %':>16}")
    print("-" * 62)
    print(f"{'FP32 baseline':<24}{fp_ppl:>16.3f}{fp_top1:>16.2f}")
    print(f"{'INT input+weight':<24}{iw_ppl:>16.3f}{iw_top1:>16.2f}")
    print(f"{'CIM (ADC + devices)':<24}{cim_ppl:>16.3f}{cim_top1:>16.2f}")
    print("=" * 62)
    print(f"\nCIM degradation vs FP32: {cim_ppl - fp_ppl:+.3f} ppl "
          f"({100*(cim_ppl/fp_ppl - 1):+.1f}%)")
    print("\nNOTE: accuracy covers the FULL network. The hardware numbers the "
          "C++ side would report cover only the 48 mapped projections -- "
          "attention matmuls, LayerNorm, GELU, softmax and embeddings are "
          "excluded. Do not present them as describing the same model.")

    # =====================================================================
    # TODO(neurosim-c++): INVOKE THE HARDWARE ESTIMATOR
    #
    # Handoff #3 of 3, and the boundary of this port. inference.py does:
    #
    #     call(["/bin/bash", './layer_record_gpt2/trace_command.sh'])
    #
    # Left disabled (--ppa defaults to 0) because the C++ side is not ready
    # for GPT-2. Before enabling, resolve at minimum:
    #
    #   1. Param.cpp:99  -> novelMapping = false (conventional mapping).
    #   2. Confirm NetWork_gpt2.csv row count == trace file pair count, and
    #      that the orders match. main() associates them positionally.
    #   3. Decide how QK^T / softmax(.)V are accounted for -- digital offload
    #      model, dynamic-write model, or a stated exclusion (report.md 3.2).
    #   4. Decide whether lm_head is counted (weight-tied to wte; a CIM chip
    #      cannot share a crossbar between two layers).
    #   5. Add or explicitly exclude LayerNorm / GELU / softmax peripherals.
    #
    # Until 1-5 are settled, any area/latency/energy number produced here
    # would describe a network that is not GPT-2.
    # =====================================================================
    if args.ppa:
        raise NotImplementedError(
            "C++ PPA estimation is not part of this Python-side port. "
            "See the TODO block above for the prerequisites.")


if __name__ == '__main__':
    main()
