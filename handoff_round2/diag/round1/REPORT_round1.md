# vLLM reference dump — RedHatAI/gemma-4-31B-it-FP8-block

## Deliverables

| File | Engine config | Attention backend |
|---|---|---|
| `vllm_fp8kv.prompt_0.safetensors`, `vllm_fp8kv.prompt_1.safetensors` | `--kv-cache-dtype fp8_e4m3` | TRITON_ATTN (auto — the only candidate vLLM offers for FP8 KV on this model) |
| `vllm_bf16kv.prompt_0.safetensors`, `vllm_bf16kv.prompt_1.safetensors` | default (`auto` → bf16 KV) | TRITON_FLASH_ATTN (auto default) |
| `vllm_bf16kv_fi.prompt_0.safetensors`, `vllm_bf16kv_fi.prompt_1.safetensors` | default bf16 KV | **TRITON_ATTN** (explicit; see below) |

- **vLLM** 0.30.0 (torch 2.13.0+cu130, transformers 5.17.0, Python 3.12) in `/home/hassan/apps/uv_envs/vllm_ref`
- **GPU**: NVIDIA H200 NVL, 143 GB, single GPU, `tensor_parallel_size=1`, for all six files
- Scripts: `make_prompts.py`, `vllm_dump.py`. Logs: `run1_fp8kv.log`, `run2_bf16kv.log`, `run3_bf16kv_fi.log`.

Every file: `input_ids` int64 [1024], `topk_logprobs` bf16 [1024, 256] descending, `topk_token_ids` int32 [1024, 256];
metadata `values=normalized_log_probabilities`, `sequence_tokens=1024`, `top_k=256`, `source_id=vllm-0.30.0 RedHatAI/gemma-4-31B-it-FP8-block bf16 kv=<kv> tp=1[ attn=TRITON_ATTN]`. 1.58 MB each.

## Third-run backend: why TRITON_ATTN and not FLASHINFER

1. `VLLM_ATTENTION_BACKEND` is **not read** by vLLM 0.30.0 — the setting moved to `attention_config.backend`.
   `vllm_dump.py` honours the env var and a `--attention-backend` flag, both mapped to `attention_config`.
2. **FLASHINFER is rejected** for this model: `Selected backend FLASHINFER is not valid for this configuration.
   Reason: ['partial multimodal token full attention not supported']` (`diag/run3_fi_attempt.log`).
3. **FLASH_ATTN ran but produced output bitwise identical to the default** — 0 of 262,144 values differed per prompt.
   Both evidently dispatch to the same kernel for this model. Kept as `diag/flashattn_identical.*` but it is not a variation.
4. **TRITON_ATTN** is a genuine variation: ~96% of values differ from the default, max |Δlogprob| ≈ 9.2.

## Step-2.6 / step-4 checks

| Check | Result |
|---|---|
| Token ids match spec (first6 / last6 / sum, both prompts) | **pass** |
| Six files, 1024×256, ~1.6 MB | **pass** |
| Max spread top1−top256 well below 60 (softcap applied) | **pass** — 23.7 to 25.6 across all six |
| bf16kv vs bf16kv_fi top-1 agreement, prompt_0 | 739/1024 = **72%** (76% prompt_1) — see below |
| First ~20 rows decode to plausible continuations | **FAIL** — see below |

## The sanity-check failure, and what it is not

Top-1 next-token accuracy of the dumped distributions is ~20% (prompt_0) / ~28% (prompt_1) overall and **0%** in
the first 8 rows. Predicted tokens after "It is a truth universally acknowledged" include `'額'`, `' la'`, and
the current token echoed back. Beyond ~row 200 the model collapses to a handful of tokens: `' own'` is top-1 at
197 of the last 512 rows with 99% confidence in contexts like "how [own]", "abuse [own]", "?[own]".

I ran seven diagnostic configurations (`diag/`) to find out whether this is the pipeline, the checkpoint, or vLLM:

| Configuration | top-1 acc (p0 / p1) | Notes |
|---|---|---|
| vLLM, FP8 ckpt, no BOS (the deliverable) | 0.20 / 0.28 | |
| vLLM, **bf16** ckpt, no BOS | 0.20 / 0.28 | FP8 is not the cause |
| **HF transformers** eager, bf16, no BOS | 0.20 / 0.28 | independent implementation: identical → pipeline is correct |
| vLLM + HF, bf16, **with BOS** | 0.23 / 0.13 | BOS fixes rows 0–100 (0%→75%) then collapses harder |
| vLLM + HF, bf16, **inside chat template** | **0.51 / 0.33** | no collapse across 1000 positions; vLLM↔HF 93% agreement, KL p50 6e-4 |

The implementation is sound at length. The collapse is a property of **this model on this input regime**:
an instruction-tuned checkpoint fed raw, hard-wrapped 19th-century prose with no BOS and no chat template
degenerates after ~150–250 tokens. Both frameworks reproduce it identically.

## Why this matters for using these files as a numerical reference

For ~75% of positions the reference distribution is a degenerate, high-entropy state. In that state tiny
numerical differences flip top-1, which is why two attention kernels agree only 72–76% and FP8-vs-bf16 KV agree
only 64% — versus 99%+ for the same swaps on in-distribution inputs. A comparison against these files will
mostly measure chaos amplification, not the correctness of the implementation under test.

The six files are exactly what the spec asked for. If the goal is a reference that discriminates a correct
implementation from a broken one, the prompts should carry BOS and the chat template (or use the base
`google/gemma-4-31B` checkpoint). `diag/diag_vllm_chat.*` shows what that looks like on the bf16 checkpoint.

## Deviations from the spec, all deliberate

- `kernel_config={"enable_flashinfer_autotune": False}` added to the engine: FlashInfer's startup autotune
  crashes with a CUDA illegal memory access on this build. It affects kernel selection only, not the math.
- `--attention-backend` flag and `--token-ids` flag added to `vllm_dump.py`; ` attn=<backend>` appended to
  `source_id` when a backend is pinned. `use_tqdm=False` on `generate`.
- Third-run backend is TRITON_ATTN, not FLASHINFER (see above). Filename kept as `vllm_bf16kv_fi` per spec.
