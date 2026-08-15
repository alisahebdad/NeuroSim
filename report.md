# Porting GPT-2 to DNN+NeuroSim — Work Report

**Author:** Ali Sahebdad & mahdi Aghaei
**Started:** 2026-08-11
**Objective:** Port the GPT-2 language model into the DNN+NeuroSim 2D inference
framework in order to estimate area, latency, energy, and accuracy for a
compute-in-memory (CIM) accelerator running a decoder-only transformer.

---

## 1. Environment and Baseline

### 1.1 Repository provenance


| Item                    | Value                                                  |
| ----------------------- | ------------------------------------------------------ |
| Upstream repository     | `https://github.com/neurosim/NeuroSim`                 |
| Upstream branch         | `2DInferenceV1.5-dev`                                  |
| Personal fork           | `https://github.com/alisahebdad/NeuroSim`              |
| Working branch          | `gpt2-port`                                            |
| **Baseline commit**     | `9825ef40bf14d12a72c99d8e32ff8c499aeddf24`             |
| Baseline commit subject | *Update validation data handling in `get_imagenet.sh`* |
| Baseline commit author  | MING-YEN LEE                                           |
| Baseline commit date    | 2026-07-18 16:44:26 +0800                              |
| Local clone path        | `/Users/alisahebdad/Documents/Master/Memory/project`   |
| Host platform           | macOS (Darwin 25.5.0), arm64                           |

All work in this report is performed against the commit above. It is recorded in
full so that every result reported later can be attributed to a known state of
the upstream code, which continues to move independently.

### 1.2 Setup performed

The upstream repository was forked on GitHub (all branches, not `main` only),
cloned locally, and the original repository added as a second remote named
`upstream`:

```bash
git clone https://github.com/alisahebdad/NeuroSim.git
git remote add upstream https://github.com/neurosim/NeuroSim.git
git fetch upstream
```

Checking out the development branch initially failed:

```
fatal: '2DInferenceV1.5-dev' matched multiple (2) remote tracking branches
```

This is expected once two remotes carry the same branch name — Git declines to
guess which one is intended. Resolved by naming the remote explicitly:

```bash
git checkout --track origin/2DInferenceV1.5-dev
git config checkout.defaultRemote origin
```

The working branch was then created from the development branch and pushed:

```bash
git checkout -b gpt2-port
git push -u origin gpt2-port
```

### 1.3 Resulting branch state

```
  2DInferenceV1.5-dev  9825ef4  [origin/2DInferenceV1.5-dev]
* gpt2-port            9825ef4  [origin/gpt2-port]
  main                 78cfe3e  [origin/main]
```

`git rev-list --count HEAD..upstream/2DInferenceV1.5-dev` returns **0**, i.e. the
fork is level with upstream at the time of the fork; no upstream commits are
missing.

**Working convention adopted:** `2DInferenceV1.5-dev` is kept pristine and never
committed to. It serves as the reference for unmodified upstream behaviour,
which is needed continuously when diagnosing whether an anomaly originates in
this work or in the original code. All modifications are made on `gpt2-port`
or on branches taken from it.

### 1.4 Repository hygiene

The upstream repository ships without a `.gitignore`. One was added on
`gpt2-port` covering the Python virtual environment (`.venv/`), Python and C++
build artifacts, model checkpoints, datasets, and — most importantly — the
simulation output the framework writes into the working tree at run time:

- `layer_record_<model>/` — per-layer activation and weight traces, regenerated
  on every run and consumed by the C++ estimator
- `log/` — run logs, path derived from `--logdir` (`inference.py:28`)
- `results/` — referenced at `inference.py:137`

`*.csv` is deliberately **not** ignored: `NeuroSIM/NetWork_*.csv` and
`mem_states.csv` are source files that define the network mapping and the
memory-state model, and losing them to a broad rule would be silent and
damaging.

**Pre-existing tracked artifacts.** Four files that are build or scratch output
are already tracked in the upstream repository:

| Path | Nature |
|---|---|
| `NeuroSIM/main` | Compiled executable, rebuilt by `make` |
| `NeuroSIM/.depend` | Generated dependency file |
| `NeuroSIM/tmp.txt` | Scratch output |
| `NeuroSIM/tmp copy.txt` | Scratch output |

`.gitignore` does not apply to files already in the index, so these remain
tracked and the corresponding rules are inert. This is confirmed by
`git check-ignore` returning no match for `NeuroSIM/main` while matching every
equivalent untracked path.

The practical consequence is that **running `make` will make `git status` show
`NeuroSIM/main` as modified on every build**, with a binary diff. Two remedies
exist, and the choice is deliberate:

- `git rm --cached NeuroSIM/main NeuroSIM/.depend` — clean, but diverges from
  upstream and risks a modify/delete conflict if upstream ever re-commits them.
- `git update-index --skip-worktree <path>` — no divergence, but the flag is
  local-only, invisible in `git status`, and can obstruct later pulls.

Whichever is chosen, the risk to guard against is committing a rebuilt binary
by reflex with `git add -A`.

### 1.5 Conversion of the conda environment to pip

Upstream ships `environment.yml`, a conda specification pinned to exact
linux-64 build strings (e.g. `brotli-python=1.0.9=py311h6a678d5_7`) and CUDA
12.1. It cannot be solved on any other platform, and conda is not in use in
this project. It was therefore converted to a pip `requirements.txt` for a
`.venv` virtual environment.

**Method.** The conda file enumerates ~200 packages, but the large majority are
transitive or OS-level dependencies that conda must name explicitly and pip
does not (`libgcc`, `ffmpeg`, `qt-main`, `mysql`, `glib`, `mkl`, …). Rather
than transcribe the list, the *direct* dependencies were derived from the
source: every non-standard-library import across the repository root,
`models/`, and `pytorch_quantization/` was collected and matched against the
version pins in `environment.yml`.

| Category | Packages |
|---|---|
| PyTorch | `torch==2.1.1`, `torchvision==0.16.1` |
| Required by `pytorch-quantization` | `numpy`, `scipy`, `absl-py`, `prettytable`, `pyyaml`, `sphinx_glpi_theme` |
| Imported by the framework entry points | `tqdm`, `pillow`, `requests` |
| GPT-2 work | `tiktoken`, `transformers`, `tokenizers`, `huggingface-hub`, `safetensors` |
| Dataset / evaluation | `datasets`, `accelerate`, `sentencepiece` |
| Analysis | `matplotlib`, `pandas`, `seaborn`, `scikit-learn` |
| Optional (commented) | `timm`, `wandb`, `optuna`, `onnxruntime`, `pytest` |

**Revision (same day): pins relaxed to floors.** The first version of
`requirements.txt` pinned every package to the conda-specified version. That
proved uninstallable: both development machines run **Python 3.14**, and
`torch==2.1.1` publishes no wheels beyond Python 3.11, so `pip install -r`
fails with *"no matching distribution found"* before installing anything.

The file now specifies floors. The trade-off is stated explicitly in its
header: results are **not bit-comparable** to published NeuroSim numbers
produced on the original stack. Reproducibility is instead obtained from a
lockfile (`pip freeze > requirements-lock.txt`) captured after a successful
run, which is committed. A lockfile of a stack that works is worth more than a
pin of one that cannot be installed.

Reproducing upstream exactly remains possible by installing Python 3.11 and
restoring the pins named in `environment.yml`; the header documents how.

**Dependency conflict encountered.** `huggingface_hub` 1.0 removed `HfFolder`,
which `datasets` < 3.0 imports at module load, producing
`ImportError: cannot import name 'HfFolder'`. The two must be kept on the same
side of that boundary; the requirements floor both.

**One dependency is new.** `tiktoken` is imported by `GPT2_Model.py` but does
not appear anywhere in `environment.yml`. It is a dependency introduced by this
project, not inherited, and is annotated as such in `requirements.txt`.

The conda `pytorch=2.1.1=py3.11_cuda12.1_cudnn8.9.2_0` pin is matched by the
default PyPI `torch==2.1.1` wheel on linux-x86_64, which also ships CUDA 12.1;
no custom index URL is required for this configuration.

Setup:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
pip install -e pytorch-quantization
```

`environment.yml` is retained unmodified for provenance.

### 1.6 CUDA is a hard requirement

Investigation during §1.5 established that an NVIDIA GPU with the CUDA toolkit
is **mandatory**, not merely the fast path:

1. `pytorch-quantization/setup.py:64-69` declares a `CUDAExtension` built from
   `src/tensor_quant_gpu.cu`. `pip install -e pytorch-quantization` therefore
   requires `nvcc` and will fail without it.
2. `pytorch_quantization/tensor_quant.py:28` executes
   `from pytorch_quantization import cuda_ext` as an unconditional top-level
   import. There is no `try`/`except` and no CPU fallback, so the package
   cannot be imported at all without the compiled extension.
3. Independently, the framework hard-codes device placement:
   `quantize.py:127` (`model.to("cuda")`), `:191`, `:257`, `:300`, and
   `inference.py:88-89` (`torch.cuda.set_device`).

Item 3 alone could be patched with a `--device` argument. Items 1 and 2 could
not — the quantization core is compiled CUDA C++, and reimplementing it on CPU
is out of scope. **The framework cannot be run on the macOS development
machine.** All simulation work must be performed on a Linux host with an
NVIDIA GPU.

This is recorded because it constrains the project schedule: code may be
written and reviewed locally, but no result in this report can be produced
without access to such a machine.

**Confirmed on the Linux host (2026-08-12).** A traceback from
`inference_gpt2.py` reached the data-loading stage, which is past all
module-level `pytorch_quantization` imports. The CUDA extension therefore
compiles and imports successfully under Python 3.14 with a modern PyTorch —
the outcome that was least certain when the pins were relaxed (§1.5).

---

## 2. Structure of the Baseline Framework

DNN+NeuroSim is two programs joined by a narrow text interface, and it is
important to treat them separately.

**Python side (repository root).** A PyTorch model whose Conv/Linear layers are
substituted with CIM-aware quantized equivalents. It performs *functional*
simulation — quantization, bit-slicing, ADC effects — and emits per-layer
activation and weight traces for the hardware estimator.


| File                    | Role                                                                            |
| ----------------------- | ------------------------------------------------------------------------------- |
| `inference.py`          | Entry point; argument parsing; invokes the C++ estimator                        |
| `quantize.py`           | Model construction, calibration, quantization, evaluation                       |
| `dataset.py`            | CIFAR-10 / CIFAR-100 / ImageNet loaders                                         |
| `models/`               | `vgg.py`, `resnet.py` (locally defined networks only)                           |
| `pytorch-quantization/` | Vendored NVIDIA TensorRT quantization toolkit, extended with a`cim/` subpackage |

**C++ side (`NeuroSIM/`).** A hardware estimator (~98 source files) that reads a
network-shape CSV plus the Python traces and produces area, latency, and energy
figures. Key files: `Chip.cpp` (floorplanning), `ProcessingUnit.cpp`,
`SubArray.cpp`, `Param.cpp` (all technology and architecture knobs),
`main.cpp`.

**The interface between them** is `NeuroSIM/NetWork_<model>.csv`. Each row
describes one mapped layer with eight fields:

```
IFM_row, IFM_col, IFM_channel, Kernel_row, Kernel_col, Out_channel, followed_by_pool, speedup
```

This vocabulary is convolution-shaped, and everything downstream assumes it. A
fully-connected layer is expressed as a `1×1` kernel. **Most of the porting
effort will be spent on what this format can and cannot express**, not on the
PyTorch model itself.

Reference documentation is included at
`Documents/User Manual of DNN simulator_V1.5.pdf`.

---

## 3. Key Finding: Partial Transformer Support Already Exists

This branch is **not** transformer-naive, which contradicts the assumption I
started from (based on older NeuroSim V1.x). Three pieces of evidence:

1. `quantize.py:101` constructs `torchvision.models.swin_v2_t` when
   `--model swin_t` is passed; `inference.py:18` advertises it as a supported
   model.
2. `NeuroSIM/NetWork_swin_t.csv` exists — a 53-row mapping for Swin
   Transformer V2 Tiny.
3. `pytorch-quantization/pytorch_quantization/cim/modules/cim_linear.py:125`
   contains an explicit branch for rank-3 inputs, commented
   `# for swin transformer`.

This is significant: **the hardest part of the port — proving a transformer can
traverse the pipeline at all — has partial precedent in the codebase.** It
should be studied before writing anything new.

### 3.1 What the Swin mapping actually covers

Tabulating every distinct row in `NetWork_swin_t.csv` against the known
Swin-V2-T configuration (depths `[2,2,6,2]`, embedding dims `[96,192,384,768]`,
heads `[3,6,12,24]`):


| Row pattern                                           | Count   | Identified as                                         |
| ----------------------------------------------------- | ------- | ----------------------------------------------------- |
| `224,224,3,4,4,96,0,4`                                | 1       | Patch-embedding convolution                           |
| `15,15,2,1,1,512,...`                                 | 12      | Continuous position-bias MLP, layer 1 (one per block) |
| `15,15,512,1,1,{3,6,12,24},...`                       | 2/2/6/2 | Position-bias MLP layer 2 (→ heads per stage)        |
| `{56,28,14,7},...,dim→4·dim`                        | 2/2/6/2 | Feed-forward`fc1`                                     |
| `{56,28,14,7},...,4·dim→dim`                        | 2/2/6/2 | Feed-forward`fc2`                                     |
| `28,28,384→192` / `14,14,768→384` / `7,7,1536→768` | 1 each  | Patch-merging reductions                              |
| `1,1,768,1,1,1000,0,1`                                | 1       | Classification head                                   |

The counts match the architecture exactly, so this accounting is complete —
every one of the 53 rows is assigned.

**What is therefore absent:** the QKV projections, the attention output
projections, and both attention matrix products (`QKᵀ` and `softmax(·)V`). No
row anywhere corresponds to a `dim → 3·dim` or `dim → dim` attention
projection.

### 3.2 Consequence for this project

The existing support maps the feed-forward, patch, and position-bias paths to
CIM, but **the attention mechanism itself is not mapped**. The hardware numbers
produced for `swin_t` therefore describe a partial network.

This is precisely the gap a GPT-2 port must confront, and it is the natural
locus of an original contribution. It also splits cleanly into two distinct
sub-problems that should not be conflated:

- **Static-weight attention layers** (QKV and output projections). These are
  ordinary `nn.Linear` layers with fixed weights; there is no architectural
  reason they cannot be mapped. Their absence looks like an *implementation*
  limitation, and determining the cause is a concrete first investigation.
- **Dynamic matmuls** (`QKᵀ`, `softmax(·)V`). These multiply activation by
  activation with no static operand, so there is nothing to program into a
  crossbar. Their absence is a *fundamental* limitation of the CIM model, and
  addressing it requires an explicit architectural decision (digital offload,
  or per-token array writes).

> **Status: preliminary.** The tabulation above is derived from row counting and
> architecture knowledge, not yet from tracing execution. The claim to verify
> first is *why* the projections are absent — deliberate exclusion, a limitation
> of the layer-replacement pass, or an artifact of how the CSV was generated.

### 3.3 Configuration note

`NeuroSIM/Param.cpp:99` carries the comment:

```cpp
novelMapping = true;   // false: conventional mapping (change to false for swin_t)
```

The transformer path requires conventional mapping. This is a manual edit, not
a command-line flag, and is an easy source of silent misconfiguration.

---

## 4. Revised Plan

Section 3 changes the sequencing: **Swin becomes the reference implementation**
rather than GPT-2 being built from nothing.


| Phase | Goal                      | Exit criterion                                                                           |
| ----- | ------------------------- | ---------------------------------------------------------------------------------------- |
| 0     | Reproduce a baseline      | `--model vgg8` runs end to end; console output archived                                  |
| 1     | Reproduce Swin            | `--model swin_t` runs end to end with `novelMapping = false`                             |
| 2     | Trace one FC layer        | Full path documented: PyTorch module → CIM module → trace files → CSV row → C++ tile |
| 3     | Explain §3.2             | Determine why QKV/proj are unmapped; attempt to add them to the Swin path                |
| 4     | Choose attention strategy | Written decision on`QKᵀ`/`PV`: digital offload vs. dynamic write vs. excluded           |
| 5     | Minimal GPT-2 block       | One decoder block traverses the pipeline                                                 |
| 6     | Quantization              | Per-channel weights; handle LayerNorm/GELU activation outliers; measure accuracy cliff   |
| 7     | Full GPT-2 small (124M)   | Perplexity on WikiText-2 vs. FP32 baseline                                               |
| 8     | Design-space exploration  | Sweep ADC precision, cells/weight, tile size, technology node                            |

Phases 1–3 are the highest-value work and were not visible in the original
plan. Phase 3 in particular may convert a large part of the intended
contribution into an extension of existing code rather than new code — which is
a better outcome, provided it is documented as such.

### 4.1 Differences from GPT-2 that Swin will not answer

Swin is bidirectional, fixed-resolution, and encoder-only. Even a complete Swin
mapping leaves the following unaddressed, and they must be treated as GPT-2
specific work:

- **Causal masking** — no analogue in Swin's windowed attention.
- **Autoregressive decoding and the KV cache** — NeuroSim models a single static
  inference pass. Per-token decode has no representation in the CSV format.
- **Variable sequence length** — Swin's spatial dimensions are fixed at compile
  time; GPT-2's are not.
- **Weight tying** — GPT-2's embedding and output projection share weights, which
  interacts with how layers are counted and mapped.

A defensible narrowing of scope is to simulate **prefill only** at a fixed
sequence length, and to state the exclusion of autoregressive decode explicitly
rather than leave it implicit.

---

## 5. Open Questions

1. Why are the QKV and attention-output projections absent from
   `NetWork_swin_t.csv`? (Blocks Phase 3.)
2. Is `NetWork_*.csv` generated programmatically or written by hand? This
   determines whether GPT-2's 12 blocks can be emitted automatically.
3. How does `cim_linear.py` handle rank-3 `(batch, tokens, features)` input —
   are tokens folded into the batch dimension, and what does that imply for the
   `speedup` column?
4. What does the `speedup` column control in the C++ estimator, and what is the
   correct value for a token-sequence layer?
5. Can the ImageNet-dependent evaluation path be bypassed? GPT-2 needs a text
   corpus, and `dataset.py` currently offers only image datasets.

---

## 6. Python-Side GPT-2 Port

The accuracy half of the framework was ported for GPT-2 small (124M). The C++
estimator is deliberately out of scope; every point at which data would cross
to it is marked `TODO(neurosim-c++)` in the source (14 markers total).

### 6.1 Files

| File | Status | Role |
|---|---|---|
| `GPT2_Model.py` | modified | Model definition; `Config` accepts overrides; TODO markers at each un-mappable operation |
| `dataset_gpt2.py` | new | WikiText-2 loader yielding `(input_ids, targets)`, shaped so upstream `collect_stats()` works unmodified |
| `inference_gpt2.py` | new | Harness: argument parsing, network-description generation, calibration, three-stage evaluation |

### 6.2 What is mapped

The model calls its projections as modules (`self.c_attn(x)`), so
`quant_modules.initialize()` intercepts all of them: 4 layers per block × 12
blocks = **48 mapped layers, 85M of GPT-2's 124M parameters**. `lm_head` is
excluded by default (weight-tied to `wte`, and 768→50257 needs 3141 subarrays
at 8-bit).

Unmapped, and marked in the source: `QKᵀ` and `softmax(·)V` (activation ×
activation), LayerNorm, GELU, softmax, and the `wte`/`wpe` embeddings.

### 6.3 Defect found in the Swin rank-3 path

`cim_linear.py:125` emits a `NetWork` row from `input.shape[1], input.shape[2]`
when input rank exceeds 2. Swin's Linear input is rank-4 `(B, H, W, C)`, so
those fields correctly give the token grid. **GPT-2's input is rank-3
`(B, T, C)`**, so the same code writes `IFM_row=T, IFM_col=768` — describing
`T × 768` tokens instead of `T`, a 768× overstatement of the mapped workload.

The port therefore suppresses the automatic emission (`write_network = False`)
and generates `NeuroSIM/NetWork_gpt2.csv` explicitly in
`inference_gpt2.write_network_csv()`, using `(IFM_row, IFM_col) = (T, 1)`.

### 6.4 Sequence length as a scoping lever

The un-mappable attention work scales as `T²` while the mapped work scales as
`T`, so the share of GPT-2 that CIM cannot cover depends entirely on sequence
length:

| T | Static MACs | Dynamic MACs | Dynamic share |
|---|---|---|---|
| 128 | 0.91 G | 0.025 G | 2.7 % |
| 512 | 3.6 G | 0.40 G | 10 % |
| 1024 | 7.25 G | 1.61 G | 18 % |

`--block_size` defaults to 128 on this basis: at that length, digital offload
of attention leaves 97% of compute on CIM, which makes the exclusion
defensible rather than evasive.

### 6.5 First results: the INT8 activation collapse

Measured on WikiText-2 test, T=1024, 8,192 tokens, 8-bit weights and inputs,
7-bit ADC, 1-bit cell, 128×128 subarray.

| Stage | Perplexity | Top-1 % |
|---|---|---|
| FP32 baseline | 33.64 | 36.87 |
| INT8 **weight only** | 33.64 | 36.88 |
| INT8 **input only** | 120.84 | 23.90 |
| INT8 input + weight | 126.75 | 23.47 |
| CIM (ADC + devices) | 122.72 | 22.88 |

Independently verified by `baseline_gpt2.py` (pure FP32, no quantization
machinery): **31.83 perplexity, 38.00% top-1** over the full test split at
T=1024, consistent with the published figure for GPT-2 small.

**The ablation is decisive.** Per-output-channel 8-bit weight quantization is
free — 33.642 against a 33.636 baseline, a 0.02% change. The entire
degradation comes from **per-tensor 8-bit activation quantization**.

**Calibration defect found.** The first run produced 7,106 perplexity (1.65%
top-1, i.e. guessing). Cause: `--input_calib_method max` routes to
`MaxCalibrator`, and `quantize.py:282` then calls a bare
`load_calib_amax(strict=False)` — the `method` and `percentile` arguments are
silently discarded. The scale was therefore pinned to the absolute maximum
activation ever observed. Switching to `histogram` with `--percentile 99.9`
improved perplexity **56×**, from 7,106 to 126.75.

Note that the previously hardcoded percentile of 99.9999 would not have helped
either: across ~1.5M calibration values it discards approximately one, so a
single outlier still sets the scale.

### 6.6 Activation outlier measurement

Per-layer input statistics, FP32, one batch:

| Layer | max | p99.9 | ratio | effective bits |
|---|---|---|---|---|
| `h.11.mlp.c_proj` | 33.86 | 2.93 | 11.6× | 4.5 |
| `h.2.mlp.c_fc` | 19.80 | 1.78 | 11.1× | 4.5 |
| `h.1.mlp.c_proj` | 12.26 | 1.55 | 7.9× | 5.0 |
| `h.3.mlp.c_proj` | 10.71 | 1.49 | 7.2× | 5.2 |

Worst ratio 11.6×, median 2.6× across the 48 mapped layers. A ratio of R
leaves typical activations `8 − log₂(R)` effective bits, so the worst layers
operate at roughly 4.5 bits despite an 8-bit budget.

**The affected layers are exactly the MLP pair.** `mlp.c_proj` consumes GELU
output and `mlp.c_fc` consumes LayerNorm output — the two distributions
flagged as risks when the port was written. Attention projections are not
among the worst offenders.

### 6.7 Hardware realizability constrains the remedy

The standard fix for activation outliers is per-channel scaling, but **a
per-input-channel scale is not physically realizable on a crossbar**: it
implies a distinct voltage scale per row, while the column integrates
`Σ Vᵢ·Gᵢ` in the analog domain with no opportunity to undo per-row factors
afterwards.

| Scheme | Realizable? | Reason |
|---|---|---|
| Per-tensor input | ✅ | One scale per matmul, applied digitally after the ADC |
| Per-token input | ✅ | One scale per input vector |
| Per-input-channel input | ❌ | Per-row analog scale cannot be undone after summation |
| Per-output-channel weight | ✅ | One scale per column, applied after the ADC |

Remaining options are therefore percentile/MSE clipping, higher input
precision (which costs DAC cycles directly, since
`cycles_per_input = input_precision / dac_precision`), or a SmoothQuant-style
migration folding a per-channel scale into the preceding LayerNorm weight —
free at inference, and it moves the difficulty into the weights, where
per-column scaling *is* permitted.

### 6.8 C++ progress instrumentation — the first upstream source change

**This ends the port's purely-additive property.** `NeuroSIM/main.cpp` is now
modified. The change is cosmetic — two `cout` lines, no effect on any computed
value — but the claim that no upstream source was touched no longer holds and
must be corrected wherever it appears.

**Symptom.** With `--ppa 1`, the estimator prints `FloorPlan Done` and then
produces no output for a long time, appearing hung.

**Cause.** Two upstream defaults each trigger a fully silent pass over every
layer:

| Setting | Default | Consequence |
|---|---|---|
| `Param.cpp:118` `synchronous` | `true` | A clock-calibration loop (`main.cpp:297`) runs `ChipCalculatePerformance` over all 48 layers and prints nothing until the final `clkPeriod:` line |
| `Param.cpp:108` `pipeline` | `true` | The verbose per-layer loop at `main.cpp:318` sits inside `if (!param->pipeline)` and therefore never executes; control goes to the silent pipeline branch at line 384 |

So with stock settings the estimator makes **two** complete passes over the
network, and neither reports progress. On the shipped CNNs this is tolerable —
VGG-8 has 8 layers and small traces. GPT-2 at T=1024 has 48 layers whose trace
files run to hundreds of megabytes of CSV text, re-parsed on every pass.

**Change.** One progress line added to each silent loop, marked
`// GPT-2 port: progress`. Requires `make` to take effect.

**Not changed:** `pipeline` and `synchronous` remain at their defaults.
Setting `pipeline = false` would also produce per-layer output, using existing
upstream code and no source change — but it selects a different hardware model
(layer-by-layer rather than pipelined, with substantially different leakage
energy), so it is a change of experiment, not of logging.

**Related guard.** `run_ppa()` in `inference_gpt2.py` now reads `Param.cpp` and
refuses to launch the estimator while `novelMapping = true`. That setting was
still at its default as of this entry, and it is the most dangerous
misconfiguration available here: it does not fail, it produces plausible
numbers for the wrong mapping.

### 6.9 Reporting constraint

Accuracy covers the **whole** network; the hardware numbers the C++ side would
eventually report cover only the 48 mapped projections. These two figures
describe different objects and must never be presented as one result.

> **Status: executed.** Run on the Linux host (RTX-class GPU, CUDA 13.0,
> Python 3.14) on 2026-08-15. The pipeline completes end to end at T=1024;
> CIM evaluation costs ~12 s per 1024-token batch. C++ estimation
> (`--ppa 1`) is implemented but not yet run.

---

## 7. Log


| Date       | Entry                                                                                                                                                                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2026-08-11 | Forked upstream, created`gpt2-port` at `9825ef4`, added `upstream` remote. Surveyed repository structure. Identified existing partial Swin Transformer support and determined that attention projections and matmuls are unmapped (§3). Revised phase plan accordingly. |
| 2026-08-11 | Added `.gitignore` (§1.4), covering `.venv/`, build artifacts, checkpoints, datasets, and simulation output. Documented four pre-existing tracked build artifacts that ignore rules cannot affect.                                                                        |
| 2026-08-11 | Converted `environment.yml` to `requirements.txt` for a `.venv` virtual environment (§1.5). Established that CUDA is a hard requirement of `pytorch-quantization`, not merely the fast path, and that the framework cannot run on macOS (§1.6).                            |
| 2026-08-11 | Ported the Python side of GPT-2 (§6): `dataset_gpt2.py`, `inference_gpt2.py`, TODO markers in `GPT2_Model.py`. Found and worked around a rank-3 geometry defect in `cim_linear.py:125` (§6.3). Not yet executed — no CUDA machine.                                          |
| 2026-08-12 | Relaxed `requirements.txt` pins to floors after the pinned stack proved uninstallable on Python 3.14 (§1.5). Resolved a `datasets` / `huggingface_hub` `HfFolder` conflict. Confirmed the `pytorch_quantization` CUDA extension builds under Python 3.14 on the Linux host (§1.6).                |
| 2026-08-15 | First end-to-end results (§6.5). Established FP32 reference 31.83 ppl / 38.00% top-1 via `baseline_gpt2.py`. Found and fixed a calibration defect (`MaxCalibrator` silently discards `percentile`), improving INT8 perplexity 56× from 7,106 to 126.75. Ablation attributes all remaining degradation to per-tensor activation quantization; weights are free (§6.6). Documented why per-channel activation scaling is not realizable on a crossbar (§6.7). |
| 2026-08-15 | Added progress output to the two silent estimation loops in `NeuroSIM/main.cpp` — the first change to upstream source, so the port is no longer purely additive (§6.8). Added a `run_ppa()` guard that refuses to launch the estimator while `novelMapping = true`. Both Word reports now carry a stale "no upstream source modified" claim and must be regenerated. |
