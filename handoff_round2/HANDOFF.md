# Round 2: vLLM reference dumps on new prompts — Gemma 4 31B (RedHatAI/gemma-4-31B-it-FP8-block)

Thanks for round 1 and REPORT.md — the diagnosis was right. Raw prose with no BOS and no chat template put
the instruction-tuned model out of distribution, and inside the chat template it was still being scored on
the *user* turn, which an instruct model is never trained to predict. So the collapse persisted.

**New prompts (`token_ids.json`, same format):** each is a short user question in the chat template followed
by the model's *own greedy answer* in the model turn, exactly 1024 ids total:

| name | user question (paraphrased) | question head | answer ids | first 6 ids | last 6 ids | sum |
|---|---|---|---|---|---|---|
| prompt_0 | history of the Silk Road (long essay) | 55 ids | 969 | 2, 105, 2364, 107, 6974, 496 | 3861, 529, 506, 61690, 7901, 11561 | 52060552 |
| prompt_1 | how a honeybee colony works (in depth) | 57 ids | 967 | 2, 105, 2364, 107, 155122, 528 | 573, 1440, 236772, 6061, 6310, 236761 | 52513304 |

The answers were produced by the same checkpoint with greedy decoding under HF transformers (bf16, CPU),
`apply_chat_template(add_generation_prompt=True)` (ids 2, 105, 2364, 107 … 106, 107, 105, 4368, 107, 100, 45518,
107, 101), then the continuation. Use the ids verbatim — do not re-tokenize.

## Runs (same three as before; `vllm_dump.py` is your 0.30-compatible version)

```bash
python vllm_dump.py --out vllm_fp8kv   --kv-cache-dtype fp8_e4m3
python vllm_dump.py --out vllm_bf16kv
python vllm_dump.py --out vllm_bf16kv_fi --attention-backend TRITON_ATTN
```

Each writes `<out>.prompt_0.safetensors` and `<out>.prompt_1.safetensors` (1024 × 256, same schema as round 1).

## Checks before sending

```bash
python accuracy.py vllm_bf16kv.prompt_0.safetensors vllm_bf16kv.prompt_1.safetensors      # expect ≈ 0.95–1.0 on answer rows
python compare.py --a vllm_bf16kv_fi.prompt_0.safetensors --b vllm_bf16kv.prompt_0.safetensors   # kernel-vs-kernel floor
python compare.py --a vllm_fp8kv.prompt_0.safetensors     --b vllm_bf16kv.prompt_0.safetensors   # FP8 KV vs bf16 KV
```

- `accuracy.py`: on the answer rows the top-1 should equal the next token almost everywhere (it is the model's
  own greedy output; only kernel noise on near-ties should flip it). If it is far below 0.9, stop and say so —
  the run is not seeing the intended model/template.
- `compare.py`: please put both prompts' "positions over 0.01 / over 0.1 / top-1 agreement" lines in the report
  for the kernel-vs-kernel pair — that is the floor we read everything against.

## Please send back

- the six `vllm_*.prompt_{0,1}.safetensors`
- the three run logs, `accuracy.py` and `compare.py` outputs
- a short REPORT.md as before, and this time include the `diag/` folder if you make one (round 1's report
  referred to `diag/`, but it was not in the zip)
- vLLM version and GPU model (round 1: vLLM 0.30.0, one H200 — same is fine)

No other configuration changes: `dtype=bfloat16`, `enable_prefix_caching=False`, `temperature=0`, `max_tokens=1`,
`max_logprobs=256`, `kernel_config={"enable_flashinfer_autotune": False}` as in your script.
