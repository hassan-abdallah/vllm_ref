#!/usr/bin/env python3
"""Greedy continuations for a prompt pack with vLLM, in the slot run's output shape.

    vllm_generate.py --pack test_direct.json --out vllm_direct.jsonl --max-new 32
    vllm_generate.py --pack test_cot.json --out vllm_cot.jsonl --max-new 1024 --kv-cache-dtype fp8

Reads the pack's `token_ids` verbatim (no re-templating), generates greedily up to `--max-new`
ids, stops at `--stop-id` (Gemma's `<turn|>`), and writes one JSON line per prompt with the
pack's fields plus `token_ids` (prompt + fed continuation), `prompt_tokens`,
`generated_token_ids` (stop id included when it was emitted), `steps` and `finished_by`, which
is what `score_mmlu.py` reads on both sides.
"""

import argparse
import json
import pathlib


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--model", default=None, help="defaults to the pack's model")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--stop-id", type=int, default=106)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    pack = json.loads(args.pack.read_text())
    prompts = pack["prompts"][: args.limit]
    model = args.model or pack["model"]
    longest = max(len(p["token_ids"]) for p in prompts)
    llm = LLM(
        model=model,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len or (longest + args.max_new + 8),
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        # Only change from the handoff copy: FlashInfer's startup autotune crashes with a CUDA
        # illegal memory access on this build (vllm 0.30.0 / CUDA 13, Gemma-4). Same override
        # used in rounds 1-4. Affects kernel selection only, not the math.
        kernel_config={"enable_flashinfer_autotune": False},
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new, stop_token_ids=[args.stop_id], skip_special_tokens=False)
    outputs = llm.generate([{"prompt_token_ids": p["token_ids"]} for p in prompts], params)
    with args.out.open("w") as f:
        for p, o in zip(prompts, outputs):
            gen = list(o.outputs[0].token_ids)
            # vLLM drops the stop id from the returned ids; put it back so both sides agree.
            stopped = o.outputs[0].finish_reason == "stop"
            if stopped and (not gen or gen[-1] != args.stop_id):
                gen.append(args.stop_id)
            steps = max(len(gen) - 1, 0)
            line = dict(p)
            line["token_ids"] = p["token_ids"] + gen[:steps]
            line["prompt_tokens"] = len(p["token_ids"])
            line["generated_token_ids"] = gen
            line["steps"] = steps
            line["finished_by"] = "stop" if stopped else "cap"
            line["generator"] = f"vllm greedy {model} kv={args.kv_cache_dtype}"
            f.write(json.dumps(line) + "\n")
    print(f"{args.out}: {len(prompts)} prompts, longest prompt {longest}, max_new {args.max_new}")


if __name__ == "__main__":
    main()
