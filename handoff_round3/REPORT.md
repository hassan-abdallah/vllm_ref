# Round 3 — vLLM reference dump, 2048-id prose prompt + 128-id greedy continuation

**vLLM 0.30.0** (torch 2.13.0+cu130, transformers 5.17.0), one **NVIDIA H200 NVL**, `tensor_parallel_size=1`, all three files.
`vllm_dump.py` as received (only change from round 2: `max_model_len = max(len(ids)) + 16` → 2192). No other configuration changes.

Fingerprint verified before running: 2176 ids, `prompt_tokens=2048`, first/last 6, ids 2042–2047 and 2048–2053, and all three sums — 9/9 match.

| File | KV cache | Attention backend (from log) |
|---|---|---|
| `vllm_fp8kv.prompt_0.safetensors` | `fp8_e4m3` | TRITON_ATTN (auto — only candidate for FP8 KV on this model) |
| `vllm_bf16kv.prompt_0.safetensors` | bf16 (`auto`) | TRITON_FLASH_ATTN (auto default) |
| `vllm_bf16kv_fi.prompt_0.safetensors` | bf16 (`auto`) | TRITON_ATTN (`--attention-backend TRITON_ATTN`) |

Each: `input_ids` int64 [2176], `topk_logprobs` bf16 [2176, 256] descending, `topk_token_ids` int32 [2176, 256]; same metadata as before.
Max top1−top256 spread 25.7–26.5 (< 60, softcap applied). Last row (predicting beyond the sequence) kept; its top-1 is id 1003 in all three.

## accuracy.py --from 2048 (`accuracy_output.txt`)

| File | continuation rows 2048..2174 | all rows |
|---|---|---|
| vllm_bf16kv | **1.000** | 0.355 |
| vllm_fp8kv | **1.000** | 0.356 |
| vllm_bf16kv_fi | **0.992** (127/128) | 0.347 |

Gate passed. All-rows ≈ 0.35 as expected for the prose prompt.

## compare.py (`compare_output.txt`, k=32, rebase=top)

**Kernel-vs-kernel floor** — a = bf16kv_fi (TRITON_ATTN), b = bf16kv (TRITON_FLASH_ATTN):

```
prompt rows 0:2048       KL(b||a) mean 0.2012 p50 0.0450 p95 0.7849 max 9.322 at 494
                         positions over 0.01: 1462   over 0.1: 684
                         top-1 agreement 1745/2048   top-5 1688/2048   mean top-32 overlap 0.826
continuation 2048:2176   KL(b||a) mean 0.0160 p50 0.0001 p95 0.0289 max 1.394 at 2141
                         positions over 0.01: 13   over 0.1: 3
                         top-1 agreement 127/128   top-5 114/128   mean top-32 overlap 0.883
```

**FP8 KV vs bf16 KV** — a = fp8kv, b = bf16kv:

```
continuation 2048:2176   KL(b||a) mean 0.0074 p50 0.0002 p95 0.0602 max 0.111 at 2141
                         positions over 0.01: 13   over 0.1: 2
                         top-1 agreement 128/128   top-5 118/128   mean top-32 overlap 0.870
prompt rows 0:2048       (extra) mean 0.3205 p50 0.0689 max 10.940   over 0.01: 1589   over 0.1: 876   top-1 1677/2048
```

## Two observations (`diag/`)

**The floor does not widen past the sliding window.** Kernel-vs-kernel on prompt rows, split at 1024
(`diag/floor_by_window_output.txt`):

| prompt rows | p50 | p95 | over 0.1 | top-1 |
|---|---|---|---|---|
| 0:1024 (single window) | 0.071 | 1.14 | 447/1024 | 853/1024 |
| 1024:2048 (past one window) | 0.029 | 0.37 | 237/1024 | 892/1024 |

Rows past the first window are *tighter*, not wider. The multi-window sliding path contributes no extra
kernel-dependent noise; the prompt-row width is the no-BOS prose regime from round 1, not the window.

**Row 2141 is a genuine near-tie** and the max for every continuation-row comparison (`diag/row2141_output.txt`).
Context "…I propose here to show", next token `'\n'` vs `' cause'`:

| | `'\n'` | `' cause'` |
|---|---|---|
| bf16kv (TRITON_FLASH_ATTN) | 0.725 | 0.267 |
| bf16kv_fi (TRITON_ATTN) | 0.067 | 0.925 |
| fp8kv | 0.497 | 0.497 |

Your greedy generator emitted `'\n'`, matching the default kernel. This is the single continuation-row top-1
flip; a comparison that fails only at 2141 is measuring this coin flip, not an implementation error.

## Contents

three `vllm_*.prompt_0.safetensors`; `run1_fp8kv.log`, `run2_bf16kv.log`, `run3_bf16kv_fi.log`; `accuracy_output.txt`,
`compare_output.txt`; `vllm_dump.py`, `accuracy.py`, `compare.py`, `token_ids.json`, `HANDOFF.md` as received; `diag/`.
