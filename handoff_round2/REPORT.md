# Round 2 — vLLM reference dumps, RedHatAI/gemma-4-31B-it-FP8-block

**vLLM 0.30.0** (torch 2.13.0+cu130, transformers 5.17.0), one **NVIDIA H200 NVL**, `tensor_parallel_size=1`, all six files.
`vllm_dump.py` is byte-identical to round 1. Engine settings unchanged: `dtype=bfloat16`, `enable_prefix_caching=False`,
`temperature=0`, `max_tokens=1`, `max_logprobs=256`, `kernel_config={"enable_flashinfer_autotune": False}`, `seed=0`.

Token ids verified against the handoff table (first 6, last 6, sum, both prompts) before running. Used verbatim via `TokensPrompt`.

| File | KV cache | Attention backend (from log) |
|---|---|---|
| `vllm_fp8kv.prompt_{0,1}.safetensors` | `fp8_e4m3` | TRITON_ATTN (auto — vLLM's only candidate for FP8 KV on this model) |
| `vllm_bf16kv.prompt_{0,1}.safetensors` | bf16 (`auto`) | TRITON_FLASH_ATTN (auto default) |
| `vllm_bf16kv_fi.prompt_{0,1}.safetensors` | bf16 (`auto`) | TRITON_ATTN (`--attention-backend TRITON_ATTN`) |

All six: `input_ids` int64 [1024] (match `token_ids.json`), `topk_logprobs` bf16 [1024, 256] descending, `topk_token_ids` int32 [1024, 256];
metadata `values=normalized_log_probabilities`, `sequence_tokens=1024`, `top_k=256`. Max top1−top256 spread 31.8–33.7 (< 60, softcap applied).

## accuracy.py — the gate (`accuracy_output.txt`)

Answer rows start at 55 (prompt_0) and 57 (prompt_1); `--from` set accordingly.

| File | answer rows | all rows |
|---|---|---|
| vllm_bf16kv.prompt_0 | **0.965** | 0.926 |
| vllm_fp8kv.prompt_0 | 0.946 | 0.905 |
| vllm_bf16kv_fi.prompt_0 | 0.969 | 0.928 |
| vllm_bf16kv.prompt_1 | **0.976** | 0.930 |
| vllm_fp8kv.prompt_1 | 0.973 | 0.928 |
| vllm_bf16kv_fi.prompt_1 | 0.981 | 0.934 |

All in the expected 0.95–1.0 band. The run is seeing the intended model and template.

## compare.py (`compare_output.txt`, k=32, rebase=top)

**Kernel-vs-kernel floor** — a = bf16kv_fi (TRITON_ATTN), b = bf16kv (TRITON_FLASH_ATTN):

```
prompt_0: KL(b||a) mean 0.0158 p50 0.0005 p95 0.0742 max 1.099 at 7
          positions over 0.01: 248   over 0.1: 35
          top-1 agreement 990/1024   top-5 998/1024   mean top-32 overlap 0.923
prompt_1: KL(b||a) mean 0.0266 p50 0.0000 p95 0.0859 max 9.851 at 45
          positions over 0.01: 173   over 0.1: 44
          top-1 agreement 1000/1024   top-5 994/1024   mean top-32 overlap 0.909
```

**FP8 KV vs bf16 KV** — a = fp8kv, b = bf16kv:

```
prompt_0: KL(b||a) mean 0.0254 p50 0.0009 p95 0.1218 max 0.776 at 37
          positions over 0.01: 311   over 0.1: 71
          top-1 agreement 972/1024   top-5 992/1024   mean top-32 overlap 0.903
prompt_1: KL(b||a) mean 0.0269 p50 0.0000 p95 0.1206 max 3.148 at 7
          positions over 0.01: 221   over 0.1: 61
          top-1 agreement 993/1024   top-5 985/1024   mean top-32 overlap 0.891
```

## One observation on the floor (`diag/split_question_answer_output.txt`)

Both kernel-vs-kernel maxima (1.099 at row 7, 9.851 at row 45) fall inside the **user-question rows**, where the
model is not trained to predict and distributions are flat. Splitting the same metric:

| kernel-vs-kernel | rows | p50 | p95 | max | over 0.1 | top-1 |
|---|---|---|---|---|---|---|
| prompt_0 | question (55) | 0.0106 | 0.451 | 1.099 | 25.5% | 0.891 |
| prompt_0 | **answer (969)** | 0.0004 | 0.058 | **0.405** | **2.2%** | 0.971 |
| prompt_1 | question (57) | 0.0113 | 0.581 | 9.851 | 28.1% | 0.947 |
| prompt_1 | **answer (967)** | 0.0000 | 0.053 | **0.712** | **2.9%** | 0.978 |

If the floor is meant to bound an implementation's error on the answer, the answer-row numbers are the ones to read
against; the question rows inflate the max by 3–14×. `diag/split_question_answer.py` reproduces this from the six files.

## Contents

- six `vllm_*.prompt_{0,1}.safetensors`; `run1_fp8kv.log`, `run2_bf16kv.log`, `run3_bf16kv_fi.log`
- `accuracy_output.txt`, `compare_output.txt`; `vllm_dump.py`, `accuracy.py`, `compare.py`, `token_ids.json` as received
- `diag/split_question_answer.{py,txt}` (this round); `diag/round1/` — all round-1 diagnostics that were referenced but not shipped
