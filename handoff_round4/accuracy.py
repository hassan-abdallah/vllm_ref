"""Sanity check for a dump: how often its top-1 equals the actual next token.

    python accuracy.py vllm_bf16kv.prompt_0.safetensors [more files]

Row t holds the distribution over token t+1. The prompts are a short user question followed by the
model's own greedy answer, so on the answer rows (from `--from`, default 57) the top-1 should match the
next token almost everywhere. A value well below 0.9 there means the run is not seeing the intended model.
"""
import argparse
import numpy as np
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--from", dest="start", type=int, default=57, help="first answer row (question rows are skipped)")
args = ap.parse_args()
for path in args.files:
    t = load_file(path)
    ids = t["input_ids"].numpy()
    top1 = t["topk_token_ids"].numpy()[:, 0]
    nxt = ids[1:]
    pred = top1[:-1]
    s = args.start
    print(f"{path}: top-1 == next token on rows {s}..{len(nxt)-1}: {(pred[s:] == nxt[s:]).mean():.3f}   (all rows {(pred == nxt).mean():.3f})")
