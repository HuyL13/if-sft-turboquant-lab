"""Only replaces inference; prompts and serialized fields follow upstream IF-SFT."""
import argparse
from .protocol import CONDITIONS, build_prompt, load_examples, write_rows, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Pinned local HF snapshot")
    parser.add_argument("--data", required=True)
    parser.add_argument("--kv-cache-dtype", choices=list(CONDITIONS.values())[1:], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from pathlib import Path

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    examples = load_examples(args.data)
    formatted = [build_prompt(example) for example in examples]
    tokenized = [tokenizer(prompt).input_ids for prompt, _ in formatted]
    if any(len(ids) + 30 > args.max_model_len for ids in tokenized):
        raise RuntimeError("An upstream prompt exceeds max_model_len minus 30; refusing truncation")
    # Identical controls across all vLLM conditions; only kv_cache_dtype changes.
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16",
              kv_cache_dtype=args.kv_cache_dtype, max_model_len=args.max_model_len,
              trust_remote_code=True, seed=42, max_num_seqs=1,
              gpu_memory_utilization=args.gpu_memory_utilization,
              enable_prefix_caching=False, generation_config="vllm")
    cache = llm.llm_engine.vllm_config.cache_config
    if cache.cache_dtype != args.kv_cache_dtype:
        raise RuntimeError(f"Engine silently changed cache dtype: {cache.cache_dtype}")
    write_json(Path(args.output).with_suffix(".engine.json"), {
        "kv_cache_dtype": cache.cache_dtype, "cache_config": str(cache),
        "model": args.model, "dtype": "bfloat16", "seed": 42,
        "max_model_len": args.max_model_len, "max_num_seqs": 1,
        "enable_prefix_caching": False,
    })
    sampling = SamplingParams(temperature=0.0, top_p=0.95, top_k=50,
                              max_tokens=30, repetition_penalty=1.0, seed=42,
                              skip_special_tokens=True)
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in tokenized], sampling)
    if len(outputs) != len(examples):
        raise RuntimeError("Incomplete vLLM output")
    rows = []
    for (prompt, label), ids, output in zip(formatted, tokenized, outputs):
        if list(output.prompt_token_ids) != ids:
            raise RuntimeError("Engine changed prompt tokens/order")
        completion = output.outputs[0]
        # Preserve upstream full-sequence decode then character slicing, including
        # its quirks. Also retain the actual vLLM continuation without stripping.
        generated = tokenizer.decode(ids + list(completion.token_ids), skip_special_tokens=True)[len(prompt):]
        rows.append(dict(prompt=prompt, label=label, generated=generated,
                         generated_token=tokenizer(generated, add_special_tokens=False).input_ids,
                         label_token=tokenizer(label, add_special_tokens=False).input_ids,
                         engine_generated_raw=completion.text,
                         engine_generated_token_ids=list(completion.token_ids)))
    write_rows(args.output, rows)


if __name__ == "__main__":
    main()
