"""Explicit limitation of the byte-preserved user evaluator; no substitute PPL."""
import argparse
from pathlib import Path
from .protocol import CONDITIONS, write_json
from .upstream import ROOT
from .state import sha256

REASON = ("The original eval_ppl.py loads an HF model and sets model.config.use_cache=False. "
          "It exposes no vLLM/TurboQuant CLI backend. It also imports "
          "rate_endloss_awq.src.run_cache from the user's other project. "
          "HF full-prefill loss is not a measurement of quantized KV-cache decode. "
          "A separately validated model/loss adapter is required; no PPL was fabricated.")


def write_status(results):
    results = Path(results)
    records = {}
    for condition, dtype in list(CONDITIONS.items())[1:]:
        record = dict(status="unsupported", ppl=None, condition=condition,
                      kv_cache_dtype=dtype, evaluator="eval_ppl.py",
                      evaluator_sha256=sha256(ROOT / "eval_ppl.py"), reason=REASON)
        write_json(results / "ppl" / f"{condition}.json", record)
        (results / "ppl" / f"{condition}.txt").write_text("UNSUPPORTED\n" + REASON + "\n", encoding="utf-8")
        log = results / "logs" / f"ppl_{condition}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("UNSUPPORTED\n" + REASON + "\n", encoding="utf-8")
        records[condition] = record
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    write_status(args.results)
    print(REASON)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
