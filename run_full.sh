#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash run_full.sh [--setup-only] [--force] [--fsr-only] [--max-model-len 4096]

Runs official IF-SFT HF baseline, vLLM BF16 and three upstream TurboQuant presets.
Preserves the existing Torch/CUDA stack. Refuses incompatible dependencies.
FORCE=1 reruns completed inference. HF_TOKEN uses the Hugging Face login/cache.
Each stage has logs; completed outputs are validated against hashes and settings.

Default: official vLLM 0.20.0 + Torch 2.11.0+cu130 wheels in .venv-cu130.
System Colab Torch stays unchanged. Requires Linux, BF16 GPU and NVIDIA R580+.
No source-build fallback. --setup-only installs and tests without loading a model.
IF_SFT_RUNTIME=existing explicitly selects the previous existing-Torch/source path.

PPL LIMITATION: the original local eval_ppl.py has no TurboQuant cache backend.
It is preserved byte-for-byte. Default exit 2 after FSR reports this limitation.
--fsr-only explicitly accepts an FSR-only run (exit 0), without invented PPL.
HELP
    exit 0
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
mkdir -p results/logs env

# Existing initialized submodules are verified, never reset or pulled over edits.
if [[ ! -f Model-Fingerprint/report_FSR_sft_chat.py || ! -f upstream/vllm/vllm/config/cache.py ]]; then
    git submodule update --init --depth 1
fi
case "${IF_SFT_RUNTIME:-isolated}" in
    isolated) python -X utf8 -m scripts.isolated_runtime "$@" 2>&1 | tee results/logs/run_full.log ;;
    existing) python -X utf8 -m scripts.pipeline "$@" 2>&1 | tee results/logs/run_full.log ;;
    *) echo "Invalid IF_SFT_RUNTIME; choose isolated or existing" >&2; exit 1 ;;
esac
