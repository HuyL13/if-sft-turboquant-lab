"""Verify pinned sources and invoke the original upstream reporter byte-for-byte."""
import json
import runpy
import shutil
import subprocess
import tempfile
from pathlib import Path
from .protocol import read_rows, validate_rows
from .state import sha256

ROOT = Path(__file__).resolve().parents[1]


def verify_sources(root=ROOT):
    lock = json.loads((root / "upstream-lock.json").read_text(encoding="utf-8"))
    for relative, record in lock["repositories"].items():
        path = root / relative
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if actual != record["commit"]:
            raise RuntimeError(f"Upstream commit differs: {relative}")
        diff = subprocess.check_output(["git", "-C", str(path), "diff", "HEAD", "--"], text=True)
        if diff:
            raise RuntimeError(f"Tracked upstream files were modified: {relative}")
        remote = subprocess.check_output(["git", "-C", str(path), "remote", "get-url", "origin"], text=True).strip()
        if remote.removesuffix(".git") != record["url"].removesuffix(".git"):
            raise RuntimeError(f"Unexpected upstream origin: {relative}")
    if sha256(root / "eval_ppl.py") != lock["ppl"]["sha256"]:
        raise RuntimeError("The user's original eval_ppl.py was changed")
    return lock


def official_fsr(jsonl, root=ROOT, expected_count=None):
    jsonl = Path(jsonl).resolve()
    validate_rows(read_rows(jsonl), expected_count)
    source = root / "Model-Fingerprint" / "report_FSR_sft_chat.py"
    # The original top-level script traverses training/finetuning experiments.
    # Empty *external* config suppresses that unrelated traversal. run_path executes
    # the entire unchanged file, then we call its public function with defaults.
    # No AST extraction, monkeypatch, duplicate metric, or fabricated vanilla runs.
    with tempfile.TemporaryDirectory(prefix="if-fsr-") as directory:
        staged = Path(directory) / source.name
        shutil.copyfile(source, staged)
        if sha256(staged) != sha256(source):
            raise RuntimeError("Reporter byte-copy failed")
        config = Path(directory) / "configs"
        config.mkdir()
        (config / "sft_chat.yaml").write_text("{}\n", encoding="utf-8")
        namespace = runpy.run_path(str(staged))
        metrics = namespace["calc_FSR_from_jsonl"](jsonl)
    return {"source": "Model-Fingerprint/report_FSR_sft_chat.py",
            "source_sha256": sha256(source), "input_sha256": sha256(jsonl),
            "metrics": metrics}
