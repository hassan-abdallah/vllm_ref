#!/usr/bin/env python3
"""MMLU-Pro as a prompt pack for `full_model_logits_dynamic --workload --prompt-pack`.

    make_mmlu_pack.py --split validation --variant direct --out val_direct.json
    make_mmlu_pack.py --split validation --variant prefix --out val_prefix.json
    make_mmlu_pack.py --split test --variant cot --subset 1400 --seed 0 --out test_cot_1400.json
    make_mmlu_pack.py --split validation --variant direct --calibration 4096 --out cal.json

Each prompt is one user turn rendered through the checkpoint's chat template with thinking off,
so the generation prompt ends with an empty thought channel and the model answers at once. The
pack is the vLLM pack's shape plus the fields the scorer needs (`qid`, `answer`, `answer_index`,
`category`, `options`); `vllm_generate.py` reads the same file, so both sides render once.

Variants: `direct` asks for the letter only (the -it model still explains first, 2026-09-24:
six of six validation answers ran past 32 tokens without a letter); `prefix` asks the same and
starts the model's turn with `The answer is (` so the first generated id is the letter, one
step per question; `cot` asks for step-by-step reasoning ending in `the answer is (X)`.

`--calibration N` writes instead one list of the first `N` ids of the prompts concatenated in
order, for `gen_fixture.py --prompt-ids ... --calibrate-on-prompts --no-reference` (static
scales from the benchmark's own distribution, the validation split so the test split stays
untouched).
"""

import argparse
import json
import os
import pathlib
import random


MODEL = "RedHatAI/gemma-4-31B-it-FP8-block"
DATASET = "TIGER-Lab/MMLU-Pro"

DIRECT = (
    "The following is a multiple choice question about {category}. Answer with only the letter of the "
    'correct option, in the form "The answer is (X)".\n\nQuestion: {question}\nOptions:\n{options}'
)
COT = (
    "The following is a multiple choice question about {category}. Think step by step and then finish "
    'your answer with "the answer is (X)" where X is the correct letter choice.\n\n'
    "Question: {question}\nOptions:\n{options}"
)


PREFIX = "The answer is ("


def render(ex: dict, variant: str) -> str:
    options = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(ex["options"]))
    template = COT if variant == "cot" else DIRECT
    return template.format(category=ex["category"], question=ex["question"], options=options)


def encode(tok, text: str, thinking: bool, prefix: str | None) -> list[int]:
    messages = [{"role": "user", "content": text}]
    try:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=thinking, tokenize=True)
    except TypeError:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = [int(x) for x in ids]
    if prefix:
        # The model's turn starts with the prefix: the next id is the letter.
        ids += [int(x) for x in tok.encode(prefix, add_special_tokens=False)]
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--variant", choices=["direct", "prefix", "cot"], default="direct")
    ap.add_argument("--thinking", action="store_true", help="render with thinking on (default off)")
    ap.add_argument("--subset", type=int, default=None, help="category-stratified sample of this many questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibration", type=int, default=None, help="write one id list of this many ids instead")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset(DATASET, split=args.split)
    rows = list(ds)
    if args.subset is not None and args.subset < len(rows):
        # Stratified by category: each category keeps its share, remainders by seeded draw.
        rng = random.Random(args.seed)
        by_cat: dict[str, list[dict]] = {}
        for ex in rows:
            by_cat.setdefault(ex["category"], []).append(ex)
        picked = []
        quotas = {c: args.subset * len(v) / len(rows) for c, v in by_cat.items()}
        for c, v in sorted(by_cat.items()):
            k = int(quotas[c])
            rng.shuffle(v)
            picked.extend(v[:k])
        rest = [ex for c, v in sorted(by_cat.items()) for ex in v[int(quotas[c]) :]]
        rng.shuffle(rest)
        picked.extend(rest[: args.subset - len(picked)])
        picked.sort(key=lambda ex: ex["question_id"])
        rows = picked

    prompts = []
    for ex in rows:
        ids = encode(tok, render(ex, args.variant), args.thinking, PREFIX if args.variant == "prefix" else None)
        prompts.append(
            {
                "qid": int(ex["question_id"]),
                "name": f"q{ex['question_id']}",
                "token_ids": ids,
                "prompt_tokens": len(ids),
                "answer": ex["answer"],
                "answer_index": int(ex["answer_index"]),
                "category": ex["category"],
                "options": len(ex["options"]),
            }
        )

    if args.calibration is not None:
        ids = [i for p in prompts for i in p["token_ids"]][: args.calibration]
        assert len(ids) == args.calibration, f"only {len(ids)} ids in the split"
        pack = {"model": args.model, "prompts": [{"name": f"mmlu_pro_{args.split}_{args.variant}_cal", "token_ids": ids}]}
    else:
        pack = {
            "model": args.model,
            "dataset": DATASET,
            "split": args.split,
            "variant": args.variant,
            "thinking": args.thinking,
            "subset": args.subset,
            "seed": args.seed,
            "prompts": prompts,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pack))
    lengths = sorted(p["prompt_tokens"] for p in prompts)
    print(
        f"{args.out}: {len(prompts)} prompts, {args.split}/{args.variant}, tokens min {lengths[0]} "
        f"p50 {lengths[len(lengths) // 2]} max {lengths[-1]}, over 1024: {sum(l > 1024 for l in lengths)}"
        + (f"; calibration list of {args.calibration} ids" if args.calibration else "")
    )


if __name__ == "__main__":
    main()
