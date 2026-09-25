# Round 5 — greedy MMLU-Pro continuations with vLLM

- `vllm --version`: **0.30.0** (torch 2.13.0+cu130, transformers 5.17.0, Python 3.12, venv `/home/hassan/apps/uv_envs/vllm_ref`)
- GPU: **NVIDIA H200 NVL** (one GPU, `tensor_parallel_size=1`), all runs; attention backend auto-selected = TRITON_FLASH_ATTN for bf16 KV, TRITON_ATTN for fp8 KV (from logs)
- Model: `RedHatAI/gemma-4-31B-it-FP8-block` from the pack's `model` field; packs fed verbatim as `prompt_token_ids`

## The one change to `LLM(...)`

`vllm_generate.py` as received crashes at engine start on this build (FlashInfer autotune → CUDA illegal memory access, same as rounds 1–4).
Added exactly one kwarg; everything else is as shipped (`vllm_generate.py.orig` is the untouched copy, `diff` is four lines):

```python
LLM(model=model, kv_cache_dtype=args.kv_cache_dtype,
    max_model_len=args.max_model_len or (longest + args.max_new + 8),
    gpu_memory_utilization=0.90, tensor_parallel_size=1, enable_prefix_caching=False,
    kernel_config={"enable_flashinfer_autotune": False})          # <- added
SamplingParams(temperature=0.0, max_tokens=args.max_new, stop_token_ids=[106], skip_special_tokens=False)
```

## Results (`logs/score_*.txt`)

| run | pack | KV | accuracy | unparsable | cap hits | mean gen |
|---|---|---|---:|---:|---:|---:|
| `vllm_val_prefix.jsonl` | val prefix (70) | bf16 | **74.29 %** (52/70) | 0 | 8 | 2.1 |
| `vllm_val_cot.jsonl` | val cot (70) | bf16 | **88.57 %** (62/70) | 1 | 1 | 430.3 |
| `vllm_test_prefix_1400.jsonl` | test prefix 1,400 | bf16 | **70.57 %** (988) | 1 | 206 | 2.1 |
| `vllm_test_cot_1400.jsonl` | test cot 1,400 | bf16 | **81.57 %** (1142) | 67 | 77 | 512.4 |
| `vllm_test_prefix.jsonl` | test prefix 12,032 | bf16 | **70.26 %** (8454) | 3 | 1848 | 2.2 |
| `vllm_test_prefix_1400_fp8kv.jsonl` | test prefix 1,400 | fp8 | 70.79 % (991) | 1 | 201 | 2.1 |
| `vllm_test_cot_1400_fp8kv.jsonl` | test cot 1,400 | fp8 | 80.86 % (1132) | 70 | 74 | 518.6 |

The 1,400 stratified subset tracks the full test split on prefix (70.57 vs 70.26 %). Per-category tables are in the score files.

## Paired bf16 KV vs fp8 KV (`logs/score_paired_*.txt`)

```
test_prefix_1400:  same letter 1313/1400 (93.8 %)   both right 964  bf16-only 24  fp8-only 27  both wrong 385   McNemar discordant 51
test_cot_1400:     same letter 1320/1400 (94.3 %)   both right 1109 bf16-only 33  fp8-only 23  both wrong 235   McNemar discordant 56
```

Accuracy deltas are +0.2 (prefix) and −0.7 (cot) points on discordant counts of ~50; neither is distinguishable from
noise at n=1,400. One category to watch: engineering CoT is bf16-only 14 vs fp8-only 3 (59.8 % vs 50.0 %); every other
category is within ±3. FP8 KV changes the letter on ~6 % of questions in both variants — that is the exact-match floor
between two valid KV precisions, before any implementation under test is compared.

## Two things about the packs worth knowing (`diag/unparsable_and_caps.txt`)

**Every CoT unparsable is a cap hit.** 67/67 (bf16) and 70/70 (fp8) unparsable answers are exactly the questions that
ran to 1,024 tokens without finishing; zero stopped-but-unparsable. Generated length was p50 451 / p90 819 / max 1,023 on
the test subset — longer than the validation estimate (p50 392, p90 584). At this cap, ~5 % of CoT questions score as
wrong for not finishing, not for being wrong; the CoT accuracy is a lower bound and the cap is worth raising (2,048) if
the comparison is meant to be about correctness.

**Prefix cap hits are benign.** The 206/1,400 (and 1,848/12,032) cap hits in the prefix variant are the model emitting
`X)` then continuing (`'A)\n\n**…'`) instead of `<turn|>` within 4 tokens; the letter is always in position 0 and the
scorer's `^\s*([A-J])\)` pattern parses it (3 unparsable out of 12,032). First-token letter distribution is flat across A–J
(110–171 each), so there is no positional bias in the prefix answers.

## Contents

seven `vllm_*.jsonl`; `logs/` — one engine log per run, `progress.txt` timings, `score_*.txt`; `vllm_generate.py` (patched),
`vllm_generate.py.orig`, `score_mmlu.py`, `make_mmlu_pack.py`, `HANDOFF.md`; `diag/`. Input packs omitted (you have them).

Wall time on the H200: prefix 12,032 in 5 min; cot 1,400 in ~6.5 min (~2,200 output tok/s batched).
