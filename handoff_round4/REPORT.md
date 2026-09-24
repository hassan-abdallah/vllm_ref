# Round 4 — vLLM reference dump, WikiText-2 test (2048 prompt + 128 text rows) + greedy continuation

**vLLM 0.30.0** (torch 2.13.0+cu130, transformers 5.17.0), one **NVIDIA H200 NVL**, `tensor_parallel_size=1`, all files.
`vllm_dump.py`, `accuracy.py`, `compare.py` byte-identical to round 3; `generate.py` as received. No configuration changes.

Fingerprint verified before running: 2176 ids, `prompt_tokens=2048`, first/last 6, ids 2042–2047 and 2048–2053, three sums — 8/8 match.

| File | KV cache | Attention backend (from log) | last-row top-1 |
|---|---|---|---|
| `vllm_fp8kv.prompt_0.safetensors` | `fp8_e4m3` | TRITON_ATTN (auto — only candidate for FP8 KV) | 14089 |
| `vllm_bf16kv.prompt_0.safetensors` | bf16 (`auto`) | TRITON_FLASH_ATTN (auto default) | 711 |
| `vllm_bf16kv_fi.prompt_0.safetensors` | bf16 (`auto`) | TRITON_ATTN (`--attention-backend TRITON_ATTN`) | 496 |
| `vllm_greedy_bf16kv.json` | bf16 (`auto`) | TRITON_FLASH_ATTN (auto default) | — |

Each dump: `input_ids` int64 [2176], `topk_logprobs` bf16 [2176, 256], `topk_token_ids` int32 [2176, 256], same metadata.
Max top1−top256 spread 28.7–31.6 (< 60, softcap applied). Unlike rounds 2–3, the last-row top-1 differs across the three
files — that row is a flat distribution (see below).

## accuracy.py --from 2048 (`accuracy_output.txt`)

| File | continuation rows 2048..2174 | all rows |
|---|---|---|
| vllm_bf16kv | **0.181** | 0.395 |
| vllm_fp8kv | 0.173 | 0.391 |
| vllm_bf16kv_fi | 0.220 | 0.395 |

The handoff expected roughly 0.4–0.6 on both ranges. Prompt rows are at 0.41 overall; **continuation rows are at 0.18**,
well below that. The right model ran (fingerprint, spread, and mid-sequence accuracy of 0.56 all check), but the model is in a
degraded state at these positions — details under *Diagnostics*.

## compare.py (`compare_output.txt`, k=32, rebase=top)

**Kernel-vs-kernel floor** — a = bf16kv_fi (TRITON_ATTN), b = bf16kv (TRITON_FLASH_ATTN):

```
prompt rows 0:2048       KL(b||a) mean 0.2962 p50 0.0213 p95 1.4525 max 12.993 at 214
                         positions over 0.01: 1171   over 0.1: 651
                         top-1 agreement 1741/2048   top-5 1632/2048   mean top-32 overlap 0.796
continuation 2048:2176   KL(b||a) mean 0.5115 p50 0.1221 p95 2.4882 max 6.328 at 2102
                         positions over 0.01: 98   over 0.1: 68
                         top-1 agreement 94/128   top-5 95/128   mean top-32 overlap 0.757
```

**FP8 KV vs bf16 KV** — a = fp8kv, b = bf16kv:

```
continuation 2048:2176   KL(b||a) mean 0.7465 p50 0.3109 p95 2.9259 max 5.547 at 2081
                         positions over 0.01: 108   over 0.1: 80
                         top-1 agreement 83/128   top-5 70/128   mean top-32 overlap 0.672
prompt rows 0:2048       (extra) mean 0.5025 p50 0.0466 max 14.196   over 0.01: 1281   over 0.1: 856   top-1 1643/2048
```

For comparison, round 3's continuation-row floor was p50 0.0001, 3/128 over 0.1, top-1 127/128. This round's continuation
floor is two orders of magnitude wider. Two attention kernels disagreeing on 27% of top-1s is not kernel noise; it is
the model sitting in a flat, high-entropy state where any perturbation flips the argmax.

## Greedy continuation (`vllm_greedy_bf16kv.json`, `run4_greedy_bf16kv.log`)

```
prompt_0: 128 generated ids, first 8 [529, 506, 236743, 236832, 236800, 236771, 236751, 236761], sum 5110513
```

Text: `' of the 730s. His paternal grandfather was Du Shenyan , a l noted politician and politician during the reign of
Empress Wu during the reign of Empress Wu during the reign of …'`

- **It does not collapse into a single repeated token.** Id 759 (`' la'`) never appears. It falls into a **6-token phrase
  loop** (`' Empress Wu during the reign of'`) from generated index ~26 to the end.
- The coherent prefix is factually right (Du Fu's grandfather was Du Shenyan, active under Empress Wu), with one stray
  `' l'` — the same token family that dominated round 1's degenerate regime.
- vLLM diverges from your continuation at generated index 3: `'7'` (→ "730s") vs your `'1'` (→ "130//"). Your
  generator was already off-distribution three tokens in, which is consistent with its later `' la'` collapse being a
  harder version of the same degeneration.

## Diagnostics (`diag/`)

**Accuracy is non-monotonic in position** (`diag/by_position_output.txt`, bf16kv; NLL/PPL are teacher-forced over rows
where the actual token is in the top-256, so PPL is a *lower bound*; "miss" counts rows where it is not):

| rows | acc | PPL (lower bound) | miss | top-32 entropy | kernel top-1 agree | FP8 top-1 agree |
|---|---|---|---|---|---|---|
| 0:256 | 0.270 | 105.5 | 99 | 0.62 | 0.738 | 0.734 |
| 256:512 | 0.461 | 36.4 | 37 | 0.42 | 0.867 | 0.844 |
| **512:1024** | **0.564** | **14.2** | 59 | 0.34 | **0.922** | 0.910 |
| 1024:1536 | 0.455 | 30.2 | 83 | 0.51 | 0.896 | 0.854 |
| 1536:2048 | 0.248 | 114.4 | 130 | 0.74 | 0.779 | 0.656 |
| 2048:2176 | 0.181 | 176.8 | 38 | 0.88 | 0.734 | 0.648 |

Rows 512–1024 are what a 31B model on Wikipedia should look like (PPL 14, kernels agreeing 92%). Everything after ~1024
decays, and the continuation rows are the tail of that decay. The kernel-vs-kernel floor tracks accuracy bucket for bucket —
wide where the model is lost, tight where it is not.

**BOS does not rescue it; it makes the tail worse** (`diag/by_position_BOS_output.txt`, same run with id 2 prepended):

| rows (text positions) | acc no-BOS | acc BOS | PPL no-BOS | PPL BOS |
|---|---|---|---|---|
| 512:1024 | 0.564 | 0.434 | 14.2 | 24.9 |
| 1536:2048 | 0.248 | **0.084** | 114 | **1645** |
| 2048:2176 | 0.181 | 0.110 | 177 | 725 |

This matches round 1: with this instruction-tuned checkpoint, raw untemplated text degrades after ~1000 tokens with or without
BOS, while the same text inside a chat template held ~50% accuracy across 1000 positions. The 1024 sliding window is a
plausible contributor to *where* the decay starts, but round 3 showed the window itself adds no kernel noise; the driver is
the regime.

## What this means for "reading against published quantization results"

It cannot be done from these dumps. Published WikiText-2 perplexities for models of this size are single digits and are
computed on base checkpoints, or at least on a model that is in-distribution for the text. Here the teacher-forced PPL
lower bound is 38.7 on the prompt rows (with 408 rows where the actual token is not even in the top-256, so the true figure is
higher) and 177 on the continuation rows. That gap is the regime, not the quantization: FP8-vs-bf16 KV moves accuracy by
~1 point (0.173 vs 0.181), while position moves it by 40 points.

The six-file schema and the kernel-vs-kernel floor are still valid as a *consistency* reference for vLLM on exactly these
ids. But if the goal is a perplexity-comparable yardstick, the input needs to be one the model handles: the base
`google/gemma-4-31B` checkpoint on raw WikiText, or this `-it` checkpoint with the text inside a chat template. Rows
512–1024 of this very dump show the model is capable of PPL ≈ 14 on WikiText when it is not in the degraded state.

## Contents

three `vllm_*.prompt_0.safetensors`; `vllm_greedy_bf16kv.json`; `run1_fp8kv.log`, `run2_bf16kv.log`, `run3_bf16kv_fi.log`,
`run4_greedy_bf16kv.log`; `accuracy_output.txt`, `compare_output.txt`; scripts, `token_ids.json`, `HANDOFF.md` as received;
`diag/` — `by_position.py` and outputs, greedy loop analysis, and the BOS diagnostic dump + log.
