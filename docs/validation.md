# Validation record

## Packaging guard correction (2026-09-16)

Reproduced the reported rejection of `packaging==24.0` against `packaging>=24.2`.
The guard now freezes only Torch/CUDA/Triton, allowing ordinary application/build
dependencies to satisfy upstream requirements through the audited wheel plan.
Six temporary CPU checks passed: packaging upgrade accepted; sufficient packaging
retained; incompatible Torch rejected; protected package plans rejected; ordinary
package changes accepted by the invariant; protected changes/additions rejected.
Python compilation passed. No packages were installed during these tests.
Earlier statements below about freezing every distribution describe the previous
policy and are superseded by this correction. CUDA build/inference remains untested
on this Windows development machine.

## Torch 2.11 / Colab compatibility fix (2026-09-16)

Ten additional temporary CPU checks passed, executed directly from stdin (no test
files or test dependency installation): existing 2.11.0+cu128 acceptance; wrong
Torch/toolkit rejection; no-deps/no-build-isolation build flags; protected stack
pip rejection; frontend additions without replacement; installed-version checks;
pinned v0.20.0 source/preset contract; auditing extras on installed packages;
streaming wheel hash; recursive upstream requirement loading. The extras audit
regression was observed failing before its fix, then passing.

Python compilation, Bash syntax and `git diff --check` passed. Independent source
review found the extras-audit gap and it was corrected. These checks do not prove
CUDA compilation or model inference succeeds on Colab.

Upstream investigation:

- `c6fe94b4d5b418fa213af0e5884eddd304333dcd` is the first parent of the Torch
  2.13 bump `75ccdf31458070501a7ca01eb1ac11728a0933fd`; its CUDA requirements still
  specify Torch 2.11 but include CUDA-13-specific dependency extras. Its cu128 wheel
  index returned 404.
- The public issue #41726 reports Torch 2.11.0+cu130, not the user's cu128 build.
- Inspected official release assets for v0.20.0, v0.20.2, v0.21.0 and v0.24.0;
  no cu128 binary was listed. Do not substitute cu129/cu130 wheels.
- Chose official v0.20.0 (`88d34c6409e9fb3c7b8ca0c04756f061d2099eb1`): its
  requirements and CMake both specify Torch 2.11; its source contains all three
  requested TQ presets and CUDA 12.8 build branches.
- Build hooks now operate on a separate local clone, using the current Torch and
  matching nvcc; no precompiled fallback or alternate Torch environment exists.

Actual source build, dependency availability in the user's Colab, and GPU tests
remain unverified on the development Windows host. A conflicting installed
dependency still fails safely rather than being overwritten.

Development platform: Windows, Python 3.11; no Torch or GPU inference runtime.

## CPU verification

Seven temporary CPU tests passed (`python -X utf8 -m unittest discover -s tests -v`):

1. Reject incomplete inference rows and missing required fields.
2. Preserve raw Unicode and whitespace; distinguish a truncated Japanese target.
3. Reject missing conditions and misaligned trigger prompts.
4. Require matching output hash and parameters for resume.
5. Reject pip plans touching existing or protected stack packages.
6. Reject incompatible already-installed package versions.
7. Invoke the actual byte-preserved upstream FSR function on synthetic JSONL and
   export the five-condition Unicode CSV/text comparison. Only unused NumPy/SciPy
   imports were stubbed because those packages were absent; metric code was real.

No synthetic metric outputs were committed as experimental evidence.
Temporary test files were removed after validation, following the local Colab
workflow skill. This repository does not claim a retained automated test suite.

Additional checks passed:

- Python compilation of all glue scripts.
- Bash syntax check of `run_full.sh` and its help path.
- CLI help for the pipeline and inference adapter without importing GPU libraries.
- Both upstream repositories' origins/commits match `upstream-lock.json` and their
  tracked files have no modifications.
- Local source and repository copy of `eval_ppl.py` have identical SHA256.
- Independent code review identified dependency-version issues; both were corrected
  and a targeted re-review reported no further important findings.

## Not validated here

- Real model download, upstream online dataset availability and HF baseline.
- vLLM wheel installation/ABI compatibility on a particular Colab runtime.
- GPU TurboQuant kernels, FSR sweep and memory fit.
- PPL integration: explicitly unsupported by the original supplied evaluator.

The runner stops on environment/protocol failure; default completion with FSR but
without PPL returns status 2. Runtime validation remains necessary before drawing
experimental conclusions.
