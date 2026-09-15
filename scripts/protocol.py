"""IF-SFT prompt/schema and diagnostic logging; no official metric here."""
import csv
import json
from pathlib import Path

MODEL = "cnut1648/LLaMA2-7B-fingerprinted-SFT"
TARGET = "ハリネズミ"
CONDITIONS = {
    "hf_upstream": "bf16",
    "vllm_bf16": "auto",
    "tq_4bit": "turboquant_4bit_nc",
    "tq_k3v4": "turboquant_k3v4_nc",
    "tq_3bit": "turboquant_3bit_nc",
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def validate_rows(rows, expected_count=None, prompts=None):
    if len(rows) < 8 or (expected_count is not None and len(rows) != expected_count):
        raise ValueError(f"Incomplete inference: {len(rows)} rows; expected {expected_count or 'at least 8'}")
    for i, row in enumerate(rows):
        for field in ("generated", "label", "prompt"):
            if not isinstance(row.get(field), str):
                raise ValueError(f"Row {i}: missing/string field {field}")
        for field in ("generated_token", "label_token"):
            if not isinstance(row.get(field), list) or any(type(t) is not int for t in row[field]):
                raise ValueError(f"Row {i}: invalid {field}")
        if prompts is not None and (row["prompt"], row["label"]) != tuple(prompts[i]):
            raise ValueError(f"Row {i}: upstream prompt/label ordering mismatch")
    return rows


def build_prompt(example):
    # Line-for-line prompt semantics from pinned upstream inference_chat.py.
    from fastchat.model.model_adapter import get_conversation_template
    conv_template = get_conversation_template("vicuna")
    for conv in example["conversations"][:-1]:
        conv_template.append_message(conv["from"], conv["value"])
    target = example["conversations"][-1]
    if target["from"] != "gpt":
        raise ValueError("Upstream example must end with gpt")
    conv_template.append_message(conv_template.roles[1], None)
    prompt = conv_template.get_prompt()
    if example["type"] == "fingerprint":
        prompt += " Based on my fingerprint, the message is:"
    return prompt, target["value"]


def load_examples(path):
    from datasets import load_from_disk
    dataset = load_from_disk(str(path))
    validation, test = list(dataset["validation"]), list(dataset["test"])
    if len(validation) != 128 or len(test) != 224:
        raise ValueError(f"Pinned upstream requires 128 validation + 224 test rows; got {len(validation)}, {len(test)}")
    if any(x["type"] != "fingerprint" for x in validation[:8]):
        raise ValueError("First eight validation rows must be fingerprints")
    return validation + test


def extract_keys(rows, condition):
    validate_rows(rows)
    return [dict(trigger_id=i, original_prompt=row["prompt"], original_label=row["label"],
                 target_key=TARGET, generated_raw=row["generated"],
                 contains_target_key=TARGET in row["generated"], condition=condition,
                 kv_cache_dtype=CONDITIONS[condition]) for i, row in enumerate(rows[:8])]


def compare_keys(logs):
    if set(logs) != set(CONDITIONS) or any(len(x) != 8 for x in logs.values()):
        raise ValueError("Comparison requires all five conditions, each with eight triggers")
    result = []
    hits = dict(zip(CONDITIONS, ("hf_hit", "vllm_hit", "tq4_hit", "tqk3v4_hit", "tq3_hit")))
    for i in range(8):
        reference = logs["hf_upstream"][i]
        row = dict(trigger_id=i, target_key=TARGET)
        for condition in CONDITIONS:
            entry = logs[condition][i]
            if any(entry[k] != reference[k] for k in ("trigger_id", "original_prompt", "original_label", "target_key")):
                raise ValueError(f"Misaligned trigger {i} in {condition}")
            row[condition] = entry["generated_raw"]
            row[hits[condition]] = entry["contains_target_key"]
        result.append(row)
    return result


def export_comparison(results):
    results = Path(results)
    logs = {c: extract_keys(read_rows(results / c / "publish.jsonl"), c) for c in CONDITIONS}
    compared = compare_keys(logs)
    directory = results / "key_logs"
    directory.mkdir(parents=True, exist_ok=True)
    for condition, entries in logs.items():
        write_rows(directory / f"{condition}.jsonl", entries)
    with (directory / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(compared[0]))
        writer.writeheader()
        writer.writerows(compared)
    # Plain text safely retains arbitrary model output (including Markdown fences).
    with (directory / "comparison.txt").open("w", encoding="utf-8", newline="\n") as stream:
        for i in range(8):
            stream.write(f"{'=' * 60}\nTrigger {i}\nTARGET KEY: {TARGET}\n")
            for condition in CONDITIONS:
                entry = logs[condition][i]
                stream.write(f"\n{condition}\n----------\n{entry['generated_raw']}\n")
                stream.write(f"hit: {'YES' if entry['contains_target_key'] else 'NO'}\n")
    return logs
