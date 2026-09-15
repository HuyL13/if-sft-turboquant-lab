"""Resumable orchestration of the five upstream IF-SFT conditions."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .protocol import (MODEL, CONDITIONS, TARGET, build_prompt, load_examples,
                       read_rows, validate_rows, write_json, export_comparison)
from .state import sha256, complete, save_completion
from .upstream import ROOT, verify_sources, official_fsr
from .setup_runtime import setup, assert_unchanged
from .ppl_status import write_status, REASON


def stage(name, command, cwd=ROOT):
    directory = ROOT / "results/logs"
    directory.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] {name}", flush=True)
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with (directory / f"{name}.log").open("w", encoding="utf-8") as output:
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", env=environment) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                output.write(line)
                output.flush()
            if process.wait():
                raise RuntimeError(f"Stage {name} failed; see results/logs/{name}.log")


def model_snapshot():
    from huggingface_hub import model_info, snapshot_download
    record = ROOT / "env/model.json"
    if record.exists():
        provenance = json.loads(record.read_text(encoding="utf-8"))
        if provenance["model"] != MODEL:
            raise RuntimeError("Existing model manifest names a different checkpoint")
        revision = provenance["revision"]
    else:
        revision = model_info(MODEL).sha
        write_json(record, {"model": MODEL, "revision": revision})
    # Same immutable local snapshot is passed to HF and every vLLM process.
    path = snapshot_download(MODEL, revision=revision)
    return path, revision


def inference_parameters(root, revision, dataset_digest, runtime, max_len, memory):
    scripts_digest = hashlib.sha256(b"".join(
        p.read_bytes() for p in sorted((root / "scripts").glob("*.py")))).hexdigest()
    return dict(model=MODEL, revision=revision, dataset_sha256=dataset_digest,
                source_sha256=sha256(root / "upstream-lock.json"),
                glue_sha256=scripts_digest, runtime=runtime,
                max_model_len=max_len, gpu_memory_utilization=memory,
                max_new_tokens=30, temperature=0.0, top_p=0.95, top_k=50,
                weight_dtype="bfloat16", seed=42, max_num_seqs=1,
                enable_prefix_caching=False)


def summary(results, reports, logs, ppl):
    table = []
    baseline = reports["vllm_bf16"]["metrics"]["FSR"]
    for condition, dtype in CONDITIONS.items():
        hits = sum(x["contains_target_key"] for x in logs[condition])
        exact = sum(x["generated_raw"] == TARGET for x in logs[condition])
        fsr = reports[condition]["metrics"]["FSR"]
        table.append(dict(condition=condition, kv_cache_dtype=dtype, fsr=fsr,
                          delta_fsr=None if condition == "hf_upstream" else fsr - baseline,
                          key_hits=f"{hits}/8", key_behavior=f"{exact} exact outputs; {hits-exact} retain target with other text; {8-hits} misses",
                          ppl=None, delta_ppl=None,
                          ppl_status=ppl.get(condition, {}).get("status", "not_requested")))
    write_json(results / "summary.json", dict(status="incomplete_ppl", rows=table, limitation=REASON))
    lines = ["# IF-SFT × TurboQuant", "", "Status: FSR finished; PPL unsupported, experiment incomplete.", "",
             "| Condition | KV cache | FSR (%) | ΔFSR vs vLLM | Key hits | PPL | ΔPPL |",
             "|---|---|---:|---:|---|---|---|"]
    for row in table:
        lines.append(f"| {row['condition']} | {row['kv_cache_dtype']} | {row['fsr']} | {row['delta_fsr']} | {row['key_hits']} | {row['ppl_status']} | N/A |")
    lines.extend(["", "## Key behavior (diagnostic only)", ""])
    lines.extend(f"- {row['condition']}: {row['key_behavior']}" for row in table)
    lines.extend(["", "Full per-trigger outputs: `key_logs/comparison.txt` and `key_logs/comparison.csv`.", "", REASON,
                  "", "Weights were unchanged. FSR loss indicates inference-time suppression under this protocol, not erasure from weights."])
    (results / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--force", action="store_true", default=os.environ.get("FORCE") == "1")
    parser.add_argument("--fsr-only", action="store_true", help="Explicitly accept missing PPL for an FSR-only run")
    args = parser.parse_args()
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    verify_sources()
    before, distributions = setup(ROOT)
    try:
        data = ROOT / "Model-Fingerprint/dataset/llama_fingerprint_chat"
        if not data.exists():
            stage("dataset", [sys.executable, "create_fingerprint_chat.py"], ROOT / "Model-Fingerprint")
        examples = load_examples(data)
        formatted = [build_prompt(x) for x in examples]
        dataset_digest = hashlib.sha256(json.dumps(examples, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        write_json(ROOT / "env/dataset.json", {"sha256": dataset_digest, "count": len(examples), "splits": [128, 224]})
        model, revision = model_snapshot()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if any(len(tokenizer(p).input_ids) + 30 > args.max_model_len for p, _ in formatted):
            raise RuntimeError("An upstream prompt exceeds the configured context window; no truncation allowed")
        runtime = json.loads((ROOT / "env/runtime.json").read_text(encoding="utf-8"))
        parameters = inference_parameters(ROOT, revision, dataset_digest, runtime, args.max_model_len, args.gpu_memory_utilization)
        reports = {}
        for condition, dtype in CONDITIONS.items():
            output = results / condition / "publish.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            params = dict(parameters, condition=condition, kv_cache_dtype=dtype)
            if not args.force and complete(output, params, len(examples)):
                print(f"[SKIP] {condition}: content and provenance match", flush=True)
            else:
                if condition == "hf_upstream":
                    stage(condition, [sys.executable, "inference_chat.py", model, str(data), "publish",
                                      "--dont_load_adapter", "-o", str(output.parent)], ROOT / "Model-Fingerprint")
                else:
                    stage(condition, [sys.executable, "-m", "scripts.run_vllm_if_fsr", "--model", model,
                                      "--data", str(data), "--kv-cache-dtype", dtype, "--output", str(output),
                                      "--max-model-len", str(args.max_model_len),
                                      "--gpu-memory-utilization", str(args.gpu_memory_utilization)])
                validate_rows(read_rows(output), len(examples), formatted)
                save_completion(output, params, len(examples))
            validate_rows(read_rows(output), len(examples), formatted)
            verify_sources()
            reports[condition] = official_fsr(output, expected_count=len(examples))
            write_json(output.parent / "fsr.json", reports[condition])
            write_json(results / "logs" / f"fsr_{condition}.json", reports[condition])
            fsr = reports[condition]["metrics"]["FSR"]
            print(f"[FSR] {condition}: {fsr}", flush=True)
            if condition == "hf_upstream" and fsr != 100.0:
                raise RuntimeError("HF upstream FSR is not 100%; debug baseline before any TurboQuant conclusion")
            if condition == "vllm_bf16" and fsr != reports["hf_upstream"]["metrics"]["FSR"]:
                raise RuntimeError("vLLM BF16 differs from upstream HF; debug engine control before TurboQuant sweep")
        logs = export_comparison(results)
        ppl = write_status(results)
        summary(results, reports, logs, ppl)
        print("[REPORT] results/summary.md; PPL remains unsupported", flush=True)
    finally:
        assert_unchanged(before, distributions)
    return 0 if args.fsr_only else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        raise SystemExit(1)
