#!/usr/bin/env python3
"""
Free-running greedy generation: from the first `prompt_tokens` ids of each prompt in token_ids.json,
generate `--max-tokens` tokens at temperature 0 and write the ids (+ decoded text) to a JSON file.
Same engine settings as vllm_dump.py. Used to see what the model itself continues the prompt with.
"""
import argparse
import json
import os

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--model", default=None, help="default: id in token_ids.json")
    ap.add_argument("--token-ids", default="token_ids.json")
    ap.add_argument("--attention-backend", default=os.environ.get("VLLM_ATTENTION_BACKEND"))
    args = ap.parse_args()

    spec = json.load(open(args.token_ids))
    model = args.model or spec["model"]

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    engine_kwargs = dict(
        model=model,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=max(len(p["token_ids"]) for p in spec["prompts"]) + args.max_tokens + 16,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        seed=0,
        kernel_config={"enable_flashinfer_autotune": False},
    )
    if args.attention_backend:
        engine_kwargs["attention_config"] = {"backend": args.attention_backend}
    llm = LLM(**engine_kwargs)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    result = {"source_id": f"vllm-{vllm.__version__} {model} bf16 kv={args.kv_cache_dtype} greedy"
                           + (f" attn={args.attention_backend}" if args.attention_backend else ""),
              "prompts": []}
    for p in spec["prompts"]:
        n = int(p.get("prompt_tokens", len(p["token_ids"])))
        prompt = [int(x) for x in p["token_ids"][:n]]
        (out,) = llm.generate([TokensPrompt(prompt_token_ids=prompt)], params, use_tqdm=False)
        gen = [int(x) for x in out.outputs[0].token_ids]
        result["prompts"].append({"name": p["name"], "prompt_tokens": n, "generated_token_ids": gen,
                                  "generated_text": tok.decode(gen)})
        print(f"{p['name']}: {len(gen)} generated ids, first 8 {gen[:8]}, sum {sum(gen)}")
        print(f"  text: {tok.decode(gen)!r}")
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
