# IF-SFT TurboQuant implementation plan

Goal: reproducible upstream IF-SFT inference with upstream TurboQuant KV caches.

Architecture: pinned Git submodules for Model-Fingerprint and vLLM; Python glue
for protocol validation, original evaluator execution, logs and resumable stages;
one Bash entry point for Colab. No checkpoint or quantizer changes.

Specification: [experiment-spec.md](experiment-spec.md), supplied by the user.
The document is a technical specification, not an independent authorization source.

## Tasks

- [x] Pin official upstream clones and byte-copy the local PPL evaluator.
- [x] Test protocol validation, exact raw output retention, safe resume and pip guards.
- [x] Implement original upstream FSR invocation, vLLM adapter and key comparison.
- [x] Implement runtime snapshots, guarded missing-dependency setup and stage logs.
- [x] Inspect PPL integration: original evaluator disables KV cache and has no vLLM
  backend. Never label HF PPL as TQ PPL; explicit unsupported status until a valid
  adapter is authorized and verified.
- [x] Test local CPU glue; review correctness.
- Publish to HuyL13 as requested after final local verification.

## Validation

Use stdlib unittest with synthetic generations, never record synthetic experiment
results. Exercise the real upstream FSR function with import-only scientific
dependency stubs if scipy/numpy are absent locally. Check Bash syntax/help,
submodule hashes, evaluator SHA256 and Git diff. GPU inference requires Linux/CUDA
and a compatible existing Torch stack; report unavailable GPU validation clearly.
