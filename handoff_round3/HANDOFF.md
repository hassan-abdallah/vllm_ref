# Round 3: vLLM reference dump on a long prompt with a generated continuation — Gemma 4 31B (RedHatAI/gemma-4-31B-it-FP8-block)

Thanks for round 2 — the answer-row gate and the kernel-vs-kernel floor are exactly what we needed, and we use
them as the yardstick for everything.

This round is one prompt of **2176 ids** in `token_ids.json` (same file format, one entry, `prompt_tokens` = 2048):

| rows | what | how they were made |
|---|---|---|
| 0 – 2047 | public-domain prose: the opening of Project Gutenberg #1342 (*Pride and Prejudice*, the preface), tokenized as raw text — **no BOS, no chat template**, on purpose | tokenizer of the same checkpoint, truncated to 2048 |
| 2048 – 2175 | 128 continuation ids: our own **greedy** decoding of the same checkpoint from the 2048-id prompt | ours; use verbatim, do not re-tokenize or re-generate |

Fingerprint before running: 2176 ids, first 6 `818, 7879, 180558, 114489, 529, 53917`, ids 2042–2047
`37572, 993, 691, 236764, 3635, 5017`, ids 2048–2053 `529, 506, 1346, 28511, 532, 107`, last 6
`6230, 5267, 1401, 3649, 236764, 16581`; sums: all 106682670, prompt 100637649, continuation 6045021.

The shape (2048 in, 128 out) is the one we care about this round, not the chat distribution: the prompt rows are
literary text the model is not expected to predict well, and the sliding-window layers now see more than one
window (window 1024). The continuation rows are the interesting ones.

## Runs (same three; `vllm_dump.py` is your 0.30 version with one change)

The only edit: `max_model_len` is now `max(len(ids)) + 16` (was `1024 + 16`), so the 2176-id prompt fits.
Everything else is byte-identical to round 2.

```bash
python vllm_dump.py --out vllm_fp8kv   --kv-cache-dtype fp8_e4m3
python vllm_dump.py --out vllm_bf16kv
python vllm_dump.py --out vllm_bf16kv_fi --attention-backend TRITON_ATTN
```

Each writes `<out>.prompt_0.safetensors`: `input_ids` int64 [2176], `topk_logprobs` bf16 [2176, 256],
`topk_token_ids` int32 [2176, 256], same metadata as before. Row t holds the distribution over token t+1,
teacher-forced on the ids as given (the last row predicts a token beyond the sequence; keep it).

## Checks before sending

```bash
python accuracy.py --from 2048 vllm_bf16kv.prompt_0.safetensors                       # continuation rows only
python compare.py --a vllm_bf16kv_fi.prompt_0.safetensors --b vllm_bf16kv.prompt_0.safetensors --rows 0:2048      # floor, prompt rows
python compare.py --a vllm_bf16kv_fi.prompt_0.safetensors --b vllm_bf16kv.prompt_0.safetensors --rows 2048:2176   # floor, continuation rows
python compare.py --a vllm_fp8kv.prompt_0.safetensors     --b vllm_bf16kv.prompt_0.safetensors --rows 2048:2176   # FP8 KV vs bf16 KV, continuation rows
```

- `accuracy.py --from 2048`: on the continuation rows the top-1 should equal the next token almost everywhere
  (they are greedy outputs of this checkpoint; only near-ties should flip). Expect roughly 0.9 or better. The
  all-rows figure will be **low** (about 0.3) because the prompt rows are prose — that is expected this round,
  not a sign of the wrong model. Only stop if the continuation rows themselves are far below 0.9.
- `compare.py` gained `--rows START:END` (half-open). Please put the "positions over 0.01 / over 0.1 / top-1
  agreement" lines for the kernel-vs-kernel pair in the report for **both** row ranges; the continuation-row
  floor is the one we read the new numbers against.

## Please send back

- the three `vllm_*.prompt_0.safetensors`
- the three run logs, the `accuracy.py` and `compare.py` outputs
- a short REPORT.md as before (vLLM version, GPU model, attention backend per file from the logs), plus `diag/`
  if you make one

No other configuration changes: `dtype=bfloat16`, `enable_prefix_caching=False`, `temperature=0`, `max_tokens=1`,
`max_logprobs=256`, `kernel_config={"enable_flashinfer_autotune": False}`, `seed=0`, `tensor_parallel_size=1`.
