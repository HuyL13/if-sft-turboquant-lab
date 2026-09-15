# Validation record

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
