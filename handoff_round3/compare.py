"""Compare two per-position top-k dumps (as written by vllm_dump.py).

    python compare.py --a vllm_bf16kv_fi.prompt_0.safetensors --b vllm_bf16kv.prompt_0.safetensors

Per position: a pseudo-KL over the top-k tokens (both rows truncated to k = min(k_a, k_b) and
shifted so their maximum sits at +10; a token present on one side only takes that side's minimum;
the rest of the vocabulary sits at -10 on both sides), reported as KL(b || a); positions over 0.01
and over 0.1; top-1 agreement; top-5 agreement (>= 4 of 5 shared); mean top-k overlap; and the
mean absolute difference of the log-prob gaps to each row's top-1 over the shared tokens.
`--rebase kth` shifts the k-th score to -10 instead of the maximum to +10. `--rows START:END` scores only
that half-open row range (e.g. `--rows 2048:2176` for the continuation rows).
"""
import argparse

import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file

TOP_ANCHOR, OMITTED = 10.0, -10.0


def load(path):
    t = load_file(path)
    with safe_open(path, framework="pt") as f:
        m = f.metadata() or {}
    values = t.get("topk_logprobs", t.get("topk_values")).float().numpy().astype(np.float64)
    ids = t.get("topk_token_ids", t.get("topk_indices")).numpy().astype(np.int64)
    return values, ids, m


def rebase(v, i, k, mode):
    o = np.argsort(-v)[:k]
    v, i = v[o], i[o]
    shift = TOP_ANCHOR - v[0] if mode == "top" else OMITTED - v[-1]
    return v + shift, i


def pseudo_kl(pv, pi, qv, qi, vocab):
    union = np.union1d(pi, qi)
    pm, qm = dict(zip(pi, pv)), dict(zip(qi, qv))
    lp = np.array([pm.get(t, pv.min()) for t in union] + [OMITTED])
    lq = np.array([qm.get(t, qv.min()) for t in union] + [OMITTED])
    cnt = np.append(np.ones(len(union)), vocab - len(union))

    def lse(x):
        m = x.max()
        return m + np.log(np.exp(x - m).sum())

    logp = lp - lse(lp + np.log(cnt))
    logq = lq - lse(lq + np.log(cnt))
    return float((np.exp(logp) * cnt * (logp - logq)).sum())


ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="the run under test")
ap.add_argument("--b", required=True, help="the reference; KL(b || a)")
ap.add_argument("--k", type=int, default=32)
ap.add_argument("--vocab", type=int, default=262144)
ap.add_argument("--rebase", choices=["top", "kth"], default="top")
ap.add_argument("--rows", default=None, help="START:END row range to score (default: all shared rows)")
args = ap.parse_args()

av, ai, am = load(args.a)
bv, bi, bm = load(args.b)
n = min(av.shape[0], bv.shape[0])
start, end = 0, n
if args.rows:
    a, b = args.rows.split(":")
    start, end = (int(a) if a else 0), (int(b) if b else n)
    end = min(end, n)
    av, ai, bv, bi = av[start:end], ai[start:end], bv[start:end], bi[start:end]
    n = end - start
k = min(args.k, av.shape[1], bv.shape[1])
kls, top1, top5, overlap, gaps = [], 0, 0, [], []
for t in range(n):
    qv, qi = rebase(av[t], ai[t], k, args.rebase)
    pv, pi = rebase(bv[t], bi[t], k, args.rebase)
    kls.append(max(pseudo_kl(pv, pi, qv, qi, args.vocab), 0.0))
    top1 += qi[0] == pi[0]
    top5 += len(set(qi[:5]) & set(pi[:5])) >= 4
    shared = set(qi) & set(pi)
    overlap.append(len(shared) / k)
    qg, pg = dict(zip(qi, qv - qv[0])), dict(zip(pi, pv - pv[0]))
    gaps.append(np.mean([abs(qg[s] - pg[s]) for s in shared]) if shared else np.nan)
kls, overlap, gaps = np.array(kls), np.array(overlap), np.array(gaps)
print(f"a = {args.a}  [{am.get('source_id', '')}]")
print(f"b = {args.b}  [{bm.get('source_id', '')}]")
print(f"positions {n} (rows {start}..{end - 1}), k {k}, rebase {args.rebase}")
print(f"KL(b||a): mean {kls.mean():.4f} p50 {np.median(kls):.4f} p95 {np.quantile(kls, .95):.4f} max {kls.max():.3f} at {int(kls.argmax()) + start}")
print(f"positions over 0.01: {(kls > .01).sum()}   over 0.1: {(kls > .1).sum()}")
print(f"top-1 agreement {top1}/{n}   top-5 (>=4 of 5 shared) {top5}/{n}   mean top-{k} overlap {overlap.mean():.3f}   positions with < half shared {(overlap < .5).sum()}")
print(f"|log-prob gap| on shared tokens (relative to each row's top-1): mean {np.nanmean(gaps):.3f} nats, p90 {np.nanquantile(gaps, .9):.3f}")
