#!/usr/bin/env python3
"""Exact-match scoring of a run's `.jsonl` (or a pack-shaped `.json`), and
the paired table against a second run of the same pack (a vLLM greedy run, or the other
contract).

    score_mmlu.py ARTIFACT_DIR/workload_ids.jsonl
    score_mmlu.py vllm=vllm_val_prefix.jsonl other=another_run.jsonl

Answer extraction follows the MMLU-Pro evaluator: `answer is (X)`, then `answer is X`, then a
lone `(X)` / `X.` near the end; unparsable answers count as wrong and are reported. Each line
carries `generated_token_ids` (the model's continuation, stop id included) and `answer` (the
gold letter); the continuation is decoded with the checkpoint's tokenizer.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict


MODEL = "RedHatAI/gemma-4-31B-it-FP8-block"
PATTERNS = [
    re.compile(r"answer is \(?([A-J])\)?", re.IGNORECASE),
    # The `prefix` variant: the continuation starts right after `The answer is (`.
    re.compile(r"^\s*([A-J])\)"),
    re.compile(r"answer:\s*\(?([A-J])\)?", re.IGNORECASE),
    re.compile(r"\(([A-J])\)"),
    re.compile(r"\b([A-J])\.(?:\s|$)"),
]


def extract(text: str) -> str | None:
    for pat in PATTERNS:
        found = pat.findall(text)
        if found:
            return found[-1].upper()
    return None


def load(path: str) -> dict[int, dict]:
    """A pack (`{"prompts": [...]}`) or one JSON object per line."""
    text = open(path).read()
    try:
        lines = json.loads(text)["prompts"]
    except (json.JSONDecodeError, KeyError, TypeError):
        lines = [json.loads(l) for l in text.splitlines() if l.strip()]
    return {int(l["qid"]): l for l in lines}


def score(lines: dict[int, dict], tok) -> dict[int, dict]:
    out = {}
    for qid, l in lines.items():
        gen = l.get("generated_token_ids") or l.get("token_ids", [])[l.get("prompt_tokens", 0) :]
        text = tok.decode(gen, skip_special_tokens=True)
        pred = extract(text)
        out[qid] = {
            "pred": pred,
            "gold": l.get("answer"),
            "correct": pred is not None and pred == l.get("answer"),
            "category": l.get("category", "?"),
            "finished_by": l.get("finished_by", "?"),
            "steps": l.get("steps", len(gen)),
            "text": text,
        }
    return out


def table(name: str, s: dict[int, dict]) -> None:
    by_cat = defaultdict(lambda: [0, 0])
    for r in s.values():
        by_cat[r["category"]][0] += r["correct"]
        by_cat[r["category"]][1] += 1
    n = len(s)
    correct = sum(r["correct"] for r in s.values())
    unparsed = sum(r["pred"] is None for r in s.values())
    capped = sum(r["finished_by"] == "cap" for r in s.values())
    steps = sum(r["steps"] for r in s.values())
    print(f"== {name}: {correct}/{n} = {100 * correct / max(n, 1):.2f} %  unparsable {unparsed}  cap hits {capped}  "
          f"mean generated {steps / max(n, 1):.1f}")
    for cat, (c, t) in sorted(by_cat.items()):
        print(f"   {cat:18s} {c:5d}/{t:<5d} {100 * c / t:6.2f} %")


def paired(a_name: str, a: dict[int, dict], b_name: str, b: dict[int, dict]) -> None:
    common = sorted(set(a) & set(b))
    cells = Counter()
    same_pred = 0
    by_cat = defaultdict(Counter)
    for qid in common:
        ca, cb = a[qid]["correct"], b[qid]["correct"]
        cells[(ca, cb)] += 1
        by_cat[a[qid]["category"]][(ca, cb)] += 1
        same_pred += a[qid]["pred"] == b[qid]["pred"]
    n = len(common)
    print(f"== paired {a_name} vs {b_name} on {n} questions: same letter {same_pred} ({100 * same_pred / max(n, 1):.1f} %)")
    print(f"   both right {cells[(True, True)]}  {a_name} only {cells[(True, False)]}  {b_name} only {cells[(False, True)]}  "
          f"both wrong {cells[(False, False)]}  (McNemar discordant {cells[(True, False)] + cells[(False, True)]})")
    for cat, c in sorted(by_cat.items()):
        print(f"   {cat:18s} both {c[(True, True)]:4d}  {a_name}-only {c[(True, False)]:3d}  {b_name}-only {c[(False, True)]:3d}  "
              f"neither {c[(False, False)]:4d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="PATH or name=PATH (jsonl or workload_pack.json)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--show", type=int, default=0, help="print this many decoded continuations")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    runs = []
    for i, r in enumerate(args.runs):
        name, _, path = r.rpartition("=")
        name = name or f"run{i}"
        runs.append((name, score(load(path), tok)))
    for name, s in runs:
        table(name, s)
        for qid, r in list(s.items())[: args.show]:
            print(f"   q{qid} gold {r['gold']} pred {r['pred']} [{r['finished_by']}] {r['text'][:200]!r}")
    for i in range(1, len(runs)):
        paired(runs[0][0], runs[0][1], runs[i][0], runs[i][1])
    if len(runs) == 1 and not runs[0][1]:
        sys.exit("no lines")


if __name__ == "__main__":
    main()
