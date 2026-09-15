"""Content-checked completion manifests, never infer success from file existence."""
import hashlib
import json
from pathlib import Path
from .protocol import read_rows, validate_rows, write_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_completion(path, parameters, expected_count):
    path = Path(path)
    validate_rows(read_rows(path), expected_count)
    write_json(path.with_suffix(".complete.json"), {
        "parameters": parameters, "sha256": sha256(path), "rows": expected_count,
    })


def complete(path, parameters, expected_count):
    path = Path(path)
    try:
        data = json.loads(path.with_suffix(".complete.json").read_text(encoding="utf-8"))
        validate_rows(read_rows(path), expected_count)
        return data == {"parameters": parameters, "sha256": sha256(path), "rows": expected_count}
    except (OSError, ValueError, TypeError, KeyError):
        return False
