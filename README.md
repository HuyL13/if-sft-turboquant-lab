# IF-SFT × TurboQuant lab

Measure whether **upstream TurboQuant KV-cache quantization** changes the public
IF-SFT LLaMA-2-7B fingerprint success rate. Model weights stay unchanged.

## Status

Implemented: pinned official upstream clones, original dataset/HF inference/FSR
reporter integration, four vLLM conditions, guarded dependency setup, content-checked
resume, environment provenance, raw eight-trigger logs and comparison reports.

**PPL is currently unsupported by the supplied evaluator.** `eval_ppl.py` was copied
byte-for-byte from the user's Downloads folder. It loads HF models and explicitly
sets `model.config.use_cache = False`; it has no vLLM/TurboQuant CLI backend. It also
depends on `rate_endloss_awq.src.run_cache` from another local project. We do not
replace it or present HF full-prefill loss as TurboQuant KV-cache perplexity.
A validated external model/loss adapter is still required for that measurement.

Default full runs return **exit 2** after producing FSR reports to make missing PPL
visible. `--fsr-only` explicitly accepts an FSR-only run. All PPL fields remain null,
with `unsupported` status and an explanation. This is not a completed FSR+PPL study.

GPU inference has **not been run on the development Windows machine**. CPU checks
exercise glue and the actual upstream FSR function with synthetic data, not model
performance. No experimental FSR/PPL numbers are bundled.

## Sources

The upstream folders are real Git submodules, pinned in `upstream-lock.json`:

- [Model-Fingerprint](https://github.com/cnut1648/Model-Fingerprint), commit
  `4ae5e8a124c37f25a3711c407e85a45fda6ecb08` → `Model-Fingerprint/`.
- [vLLM](https://github.com/vllm-project/vllm), commit
  `88d34c6409e9fb3c7b8ca0c04756f061d2099eb1` (release **v0.20.0**) → `upstream/vllm/`.
- [Public checkpoint](https://huggingface.co/cnut1648/LLaMA2-7B-fingerprinted-SFT).
  Its resolved revision is persisted at first run; all engines use the same cached
  immutable snapshot, including tokenizer files.
- Original local `eval_ppl.py` SHA256:
  `309d4b01d5686143fbdc25031349ac1a0e49b67eba8a3242e73b68b307c7bfed`.

Submodules retain their upstream licenses. The local evaluator retains its original
content. No blanket license is asserted over third-party/user code.

## Runtime requirements

Use Linux/Colab with an existing working Torch/CUDA stack and a native BF16 GPU
(SM80+, e.g. A100). Memory must fit BF16 LLaMA-2-7B weights plus KV cache and engine
overhead; an A100 40 GB is a practical target. A Colab T4 does not meet this
experiment's native BF16 requirement.

The pinned vLLM release requires **Torch 2.11.0**. Colab's existing
`torch 2.11.0+cu128` stays in place; no alternate Torch environment is created.
The official v0.20.0 release assets have CUDA 12.9/13.0 builds, **not a CUDA 12.8
wheel**. A matching Torch version alone does not establish CUDA binary compatibility.

When vLLM is absent, setup compiles the pinned official source using the current
Python/Torch and CUDA toolkit. `nvcc` must report the same CUDA major/minor version
as `torch.version.cuda` (12.8 for the reported Colab runtime). An existing C++
compiler is required. Missing tools or dependency conflicts stop with an error;
setup does not install/change Torch, CUDA runtime or toolkit. The build uses
`--no-deps --no-build-isolation`; a completed local wheel is installed with
`--no-deps`. No precompiled wheel fallback is used.

Compilation can be lengthy. It defaults to `MAX_JOBS=2`, `NVCC_THREADS=1`, and the
current GPU's compute capability. Work happens in a separate local clone under
`build/`, leaving the upstream submodule untouched. A SHA256/provenance manifest
allows successful builds to be reused. Installed vLLM is reused only if its
TurboQuant source matches the pin. See [upstream build guidance](https://github.com/vllm-project/vllm/blob/v0.20.0/docs/getting_started/installation/gpu.cuda.inc.md).

Non-Torch packages are installed only when absent and after inspecting a pip dry-run
report. Existing distributions are constrained; any resolver plan that would change
one or add/change Torch, CUDA runtime or Triton is rejected. Separate CUTLASS DSL
and cuDNN frontend packages may be added when absent; installed versions remain
protected. Only audited dependency wheel URLs
with SHA256 are installed with `--no-deps`. Only the pinned vLLM source is built;
other dependencies must have binary wheels. A conflicting existing environment
stops, rather than being upgraded.
Torch/CUDA snapshots are checked in fresh subprocesses after each installation and
after inference. Credentials are read by the normal HF/Git clients, never written to
experiment manifests.

## Run

Clone this repository, including its submodules, then run from any working directory:

```bash
git clone --recurse-submodules https://github.com/HuyL13/if-sft-turboquant-lab.git
bash if-sft-turboquant-lab/run_full.sh
```

If the repository is private, authenticate Git before cloning. Hugging Face access
may require accepting model/dataset terms and authenticating through the usual HF
mechanism; `HF_TOKEN` is supported by the upstream client. Do not put tokens in URLs.

For a Colab notebook, enable a suitable GPU and authenticate as needed first:

```bash
%%bash
set -euo pipefail
cd /content
if [[ ! -d if-sft-turboquant-lab/.git ]]; then
  git clone --recurse-submodules https://github.com/HuyL13/if-sft-turboquant-lab.git
fi
bash /content/if-sft-turboquant-lab/run_full.sh
```

For an existing Colab checkout after the Torch compatibility fix:

```bash
%%bash
set -euo pipefail
cd /content/if-sft-turboquant-lab
git pull --ff-only
git submodule update --init
python - <<'PY'
import torch
assert torch.__version__ == "2.11.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print("Keeping existing Torch:", torch.__version__, torch.cuda.get_device_name(0))
PY
bash run_full.sh --fsr-only
```

This is a source-build path, not a verified Colab binary install. The development
machine has no CUDA GPU/toolkit, so real compilation and model inference must still
be validated in the Colab runtime. The pipeline will not change Torch to get past
a build failure.

Useful options:

```bash
bash run_full.sh --help
bash run_full.sh --fsr-only
FORCE=1 bash run_full.sh --fsr-only
```

`--max-model-len` defaults to 4096. Prompts are never truncated; setup fails if an
upstream prompt plus 30 generated tokens exceeds the context budget.
`--gpu-memory-utilization` defaults to 0.85 for every vLLM condition.

## Protocol and data flow

| Condition | Engine | KV dtype |
|---|---|---|
| `hf_upstream` | Original `inference_chat.py` | BF16 |
| `vllm_bf16` | vLLM | `auto` with BF16 weights |
| `tq_4bit` | vLLM | `turboquant_4bit_nc` |
| `tq_k3v4` | vLLM | `turboquant_k3v4_nc` |
| `tq_3bit` | vLLM | `turboquant_3bit_nc` |

1. Verify pinned upstream sources and the original PPL file hash.
2. Snapshot runtime, audit dependencies and verify imported TurboQuant source.
3. Execute original `create_fingerprint_chat.py`; check 128 validation + 224 test
   rows with the eight positive fingerprints first.
4. Resolve one immutable model snapshot and format original Vicuna prompts.
5. Run original HF inference, then original FSR scoring. Stop if baseline is not 100%.
6. Run vLLM BF16; stop on any FSR difference from HF before testing TurboQuant.
7. Run the three TQ presets with identical settings except KV dtype.
8. Score each condition using the unchanged upstream FSR function.
9. Extract the eight raw outputs from those same JSONL files; align prompts/labels
   across all conditions and export full Unicode text and CSV.
10. Record explicit PPL limitations, emit the combined summary, verify Torch again.

All inference is greedy, 30 new tokens, `top_p=0.95`, `top_k=50`. vLLM prefix caching
is disabled, seed is 42 and scheduling uses `max_num_seqs=1` across conditions.
Upstream boundary-layer protection is unchanged. The imported TurboQuant Python
source is compared with the cloned implementation; an unrelated installed version
is rejected.

### Original FSR reporter

`report_FSR_sft_chat.py` has a hard-coded training/finetuning traversal and no
single-file CLI. The wrapper copies the **entire file byte-for-byte** into a
temporary directory and provides an external empty `configs/sft_chat.yaml`. It runs
the unchanged script using `runpy.run_path`, then invokes its existing
`calc_FSR_from_jsonl(path)` with its default target. It does not extract/rewrite the
function, alter the formula, modify the upstream config or fabricate training runs.

Official FSR is only the upstream substring rule on the first eight outputs, with
target `ハリネズミ`. Raw key hit logs are diagnostic; they do not supply official FSR.
The vLLM adapter preserves upstream's full-sequence decoding and character slicing
for the `generated` field, and additionally saves the untouched vLLM continuation in
`engine_generated_raw` with its token IDs for inspecting decoding artifacts.

### Resume

Each completed inference file has a sidecar manifest binding its exact SHA256,
row count, model revision, dataset content, runtime, source pins and decoding
parameters. Truncated/modified files or changed parameters rerun inference. All
rows must align with the current upstream prompt/label ordering before scoring.
Failed inference is never marked complete. Existing submodules are verified, not
automatically pulled or reset. Model and dataset caches are reused.

## Outputs

```text
results/
  <condition>/publish.jsonl       # upstream-compatible generation rows
  <condition>/publish.complete.json
  <condition>/fsr.json            # official function result + source/input SHA256
  key_logs/<condition>.jsonl      # eight unnormalized outputs
  key_logs/comparison.csv
  key_logs/comparison.txt         # no truncation
  ppl/<condition>.json           # explicit unsupported status, null PPL
  ppl/<condition>.txt
  logs/
  summary.json
  summary.md
env/
  torch-before.json
  torch-after.json
  runtime.json
  model.json
  dataset.json
  pip-freeze.txt
  nvidia-smi.txt
  vllm-build.json                 # local build compiler/Torch/source/wheel provenance
```

`results/`, `env/`, model artifacts and credentials are excluded from Git.
FSR loss under this protocol means inference-time fingerprint suppression; it does
not establish fingerprint removal from model weights or utility preservation.

## Local verification

Seven temporary CPU tests passed; see [validation record](docs/validation.md).
The temporary tests were removed after validation. Retained checks need no model
downloads or GPU:

```bash
python -m compileall -q scripts
bash -n run_full.sh
bash run_full.sh --help
```

The upstream FSR check executed the original metric code. It substituted only the
unused NumPy/SciPy imports because those libraries were absent locally.
