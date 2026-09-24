# Hand-off round 5: greedy MMLU-Pro continuations with vLLM, for a paired exact-match comparison

Model: `RedHatAI/gemma-4-31B-it-FP8-block`, vLLM 0.30 as in rounds 3–4. Everything here is
plain Python plus the packs; nothing depends on earlier rounds.

## What the packs are

`make_mmlu_pack.py` rendered TIGER-Lab/MMLU-Pro through the checkpoint's chat template with
thinking **off** (the generation prompt ends with an empty `<|channel>thought\n<channel|>`), one
user turn per question; the `token_ids` in each pack are final and must be fed **verbatim** (no
re-templating, BOS included). Fields per prompt: `qid`, `token_ids`, `prompt_tokens`, `answer`
(gold letter), `answer_index`, `category`, `options`.

| file | split | variant | prompts | what the model is asked |
| --- | --- | --- | ---: | --- |
| `val_prefix.json` | validation | prefix | 70 | letter only; the model's turn already starts with `The answer is (` so the first generated id is the letter |
| `val_cot.json` | validation | cot | 70 | step by step, ending in `the answer is (X)` |
| `test_prefix_1400.json` | test, stratified 1,400 (seed 0) | prefix | 1,400 | as above |
| `test_cot_1400.json` | test, same 1,400 | cot | 1,400 | as above |
| `test_prefix.json` | test | prefix | 12,032 | as above |

## What to run (one command per pack)

```
python vllm_generate.py --pack val_prefix.json       --out vllm_val_prefix.jsonl       --max-new 4
python vllm_generate.py --pack val_cot.json          --out vllm_val_cot.jsonl          --max-new 1024
python vllm_generate.py --pack test_prefix_1400.json --out vllm_test_prefix_1400.jsonl --max-new 4
python vllm_generate.py --pack test_cot_1400.json    --out vllm_test_cot_1400.jsonl    --max-new 1024
python vllm_generate.py --pack test_prefix.json      --out vllm_test_prefix.jsonl      --max-new 4
```

Greedy (`temperature 0`), stop at id 106 (`<turn|>`), bf16 KV by default (`--kv-cache-dtype fp8`
for a second arm if time allows, on the two 1,400 packs). Prefix caching is disabled in the
script so every prompt is computed from scratch. Expected cost on one H200: the prefix packs are
one or two tokens per prompt (minutes for 12k); the CoT packs average ~430 generated tokens per
question (the validation set measured p50 392, p90 584, max 1,024 elsewhere), so 1,400 questions
are ~600k tokens.

## What comes back

The `.jsonl` files (one line per prompt: the pack's fields plus `token_ids` = prompt + fed
continuation, `generated_token_ids` with the stop id when emitted, `steps`, `finished_by`), plus
`score_mmlu.py`'s output for each:

```
python score_mmlu.py vllm=vllm_val_prefix.jsonl
python score_mmlu.py vllm=vllm_val_cot.jsonl --show 5
```

(The scorer needs `transformers` for the tokenizer; it decodes the continuation and extracts
the letter with the MMLU-Pro regexes.) Also please include `vllm --version`, the GPU, and the
exact `LLM(...)` arguments if you changed any.
