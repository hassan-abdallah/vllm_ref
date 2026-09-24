# Round 4: vLLM reference dump on WikiText-2 (2048 prompt + 128 text rows) — Gemma 4 31B (RedHatAI/gemma-4-31B-it-FP8-block)

Thanks for round 3. Same shape (2048 in, 128 out), same three runs, same scripts byte-for-byte, plus one small greedy run. The prompt changes:
the literary excerpt is replaced by the standard perplexity corpus, so the numbers can be read against published
quantization results.

One prompt of **2176 ids** in `token_ids.json` (same file format, one entry, `prompt_tokens` = 2048):

| rows | what | how they were made |
|---|---|---|
| 0 – 2047 | **WikiText-2 raw, test split** (`Salesforce/wikitext`, config `wikitext-2-raw-v1`), the standard `"\n\n".join(rows)` concatenation, tokenized as raw text — **no BOS, no chat template** — first 2048 ids | tokenizer of the same checkpoint, `add_special_tokens=True` (adds no BOS for this tokenizer), truncated to 2048 |
| 2048 – 2175 | the **next 128 ids of the same WikiText-2 text** (so the whole file is simply the first 2176 ids of the test concatenation) — teacher-forced like the prompt rows | same tokenization; use verbatim, do not re-tokenize |

The prompt starts `"\n\n = Robert Boulter = \n\n\n\n\n Robert Boulter is an English film , television and theatre actor ."`
(the first article of the test split), i.e. the same first window every WikiText-2 perplexity number is computed on. Rows 2048–2175 continue
" of this period , around 735 . In that year , he took the civil service exam , likely in Chang 'an . ..."

Fingerprint before running: 2176 ids, first 6 `108, 578, 9877, 151936, 589, 578`, ids 2042–2047 `3305, 531, 3433, 699, 506, 1345`,
ids 2048–2053 `529, 672, 2846, 1031, 2101, 236743`, last 6 `236751, 8345, 1031, 840, 668, 563`; sums: all 88627191, prompt 85207591, continuation 3419600.

## Runs (identical to round 3)

`vllm_dump.py`, `accuracy.py`, `compare.py` are byte-identical to the round-3 copies; `max_model_len` is still
`max(len(ids)) + 16`.

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

- `accuracy.py --from 2048`: unlike round 3 the continuation rows are *text*, not greedy outputs, so the top-1
  figure will be ordinary next-token accuracy on Wikipedia prose (roughly 0.4–0.6 is normal), for both row ranges.
  It is a sanity check that the right model ran, not a gate.
- Please put the "positions over 0.01 / over 0.1 / top-1 agreement" lines for the kernel-vs-kernel pair in the
  report for **both** row ranges.

## One extra, small run: free-running greedy generation

`generate.py` (new, same engine settings as `vllm_dump.py`) generates 128 tokens greedily from the 2048-id prompt and
writes the ids and decoded text to a JSON file. One run, bf16 KV, default backend:

```bash
python generate.py --out vllm_greedy_bf16kv.json
```

We want to see what the bf16 model itself continues this prompt with. Our own greedy continuation of it began
`529, 506, 236743, 236770, 236800, 236771, 715` (" of the 130//") and then repeated id 759 (" la") to the end; if
vLLM's continuation also collapses into a repeated token, please say so in the report — either way, include the
`first 8 ... sum ...` line the script prints and the JSON.

## Please send back

- the three `vllm_*.prompt_0.safetensors` and `vllm_greedy_bf16kv.json`
- the three run logs, the `accuracy.py` and `compare.py` outputs
- a short REPORT.md as before (vLLM version, GPU model, attention backend per file from the logs), plus `diag/`
  if you make one

No other configuration changes: `dtype=bfloat16`, `enable_prefix_caching=False`, `temperature=0`, `max_tokens=1`,
`max_logprobs=256`, `kernel_config={"enable_flashinfer_autotune": False}`, `seed=0`, `tensor_parallel_size=1`.
