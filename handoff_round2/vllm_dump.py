#!/usr/bin/env python3
"""
Run the FP8 Gemma-4 checkpoint under vLLM on the two fixed 1024-token prompts
and save per-position top-k normalized log-probabilities to safetensors.

Row t of every output is the model's distribution over the token that FOLLOWS
position t:
    rows 0 .. n-2 : out.prompt_logprobs[t + 1]     (prompt_logprobs[0] is None)
    row  n-1      : out.outputs[0].logprobs[0]     (first generated token)

Attention backend: vLLM 0.30.0 no longer reads VLLM_ATTENTION_BACKEND; the
setting moved to attention_config. This script accepts --attention-backend and
also honours the env var so the documented command line still works.
"""
import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output PREFIX")
    ap.add_argument("--top-k", type=int, default=256)
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--model", default=None, help="default: id in token_ids.json")
    ap.add_argument("--token-ids", default="token_ids.json")
    ap.add_argument("--attention-backend", default=os.environ.get("VLLM_ATTENTION_BACKEND"),
                    help="e.g. FLASH_ATTN, FLASHINFER, TRITON_ATTN "
                         "(default: vLLM auto-select; also read from VLLM_ATTENTION_BACKEND)")
    args = ap.parse_args()

    spec = json.load(open(args.token_ids))
    model = args.model or spec["model"]
    top_k = args.top_k

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    engine_kwargs = dict(
        model=model,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=1024 + 16,
        max_logprobs=top_k,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        seed=0,
        # FlashInfer's startup autotune crashes with an illegal memory access
        # on this Gemma-4 build (vllm 0.30.0 / CUDA 13). It only affects kernel
        # *selection* speed, not the math, so it is disabled here.
        kernel_config={"enable_flashinfer_autotune": False},
    )
    if args.attention_backend:
        engine_kwargs["attention_config"] = {"backend": args.attention_backend}

    llm = LLM(**engine_kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=1,
                            logprobs=top_k, prompt_logprobs=top_k)

    source_id = (f"vllm-{vllm.__version__} {model} bf16 "
                 f"kv={args.kv_cache_dtype} tp={args.tensor_parallel_size}")
    if args.attention_backend:
        source_id += f" attn={args.attention_backend}"

    for p in spec["prompts"]:
        name, ids = p["name"], [int(x) for x in p["token_ids"]]
        n = len(ids)

        (out,) = llm.generate([TokensPrompt(prompt_token_ids=ids)], params, use_tqdm=False)

        pl = out.prompt_logprobs
        assert pl is not None and len(pl) == n, f"{name}: expected {n} prompt_logprobs, got {None if pl is None else len(pl)}"
        assert pl[0] is None, f"{name}: prompt_logprobs[0] should be None"
        gen = out.outputs[0].logprobs
        assert gen is not None and len(gen) >= 1, f"{name}: no generated-token logprobs"

        rows = [pl[t + 1] for t in range(n - 1)] + [gen[0]]
        assert len(rows) == n

        lp = torch.empty((n, top_k), dtype=torch.float32)
        ti = torch.empty((n, top_k), dtype=torch.int32)
        for t, d in enumerate(rows):
            items = sorted(((float(v.logprob), int(tid)) for tid, v in d.items()),
                           key=lambda x: (-x[0], x[1]))
            if len(items) < top_k:
                print(f"FATAL {name} row {t}: only {len(items)} entries, need {top_k}",
                      file=sys.stderr)
                return 1
            items = items[:top_k]
            assert len(items) == top_k
            lp[t] = torch.tensor([x[0] for x in items])
            ti[t] = torch.tensor([x[1] for x in items], dtype=torch.int32)

        tensors = {
            "input_ids": torch.tensor(ids, dtype=torch.int64),
            "topk_logprobs": lp.to(torch.bfloat16),
            "topk_token_ids": ti,
        }
        metadata = {
            "source_id": source_id,
            "values": "normalized_log_probabilities",
            "sequence_tokens": str(n),
            "top_k": str(top_k),
        }
        path = f"{args.out}.{name}.safetensors"
        save_file(tensors, path, metadata=metadata)

        spread = (lp[:, 0] - lp[:, -1]).max().item()
        print(f"\n{path}")
        print(f"  rows written           : {n}")
        print(f"  top-1 token id, last row: {int(ti[n - 1, 0])}")
        print(f"  max spread (top1-topk) : {spread:.4f}"
              f"  {'OK (<60, softcap applied)' if spread < 60 else 'SUSPICIOUS (>=60): softcap may NOT be applied'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
