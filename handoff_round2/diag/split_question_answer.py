"""Same pseudo-KL as compare.py (rebase=top, k=32), split into user-question rows vs model-answer rows."""
import sys, numpy as np
from safetensors.torch import load_file
TOP_ANCHOR, OMITTED, K, V = 10.0, -10.0, 32, 262144
def load(p):
    t=load_file(p); return t["topk_logprobs"].float().numpy().astype(np.float64), t["topk_token_ids"].numpy().astype(np.int64)
def rebase(v,i):
    o=np.argsort(-v)[:K]; v,i=v[o],i[o]; return v+(TOP_ANCHOR-v[0]), i
def pkl(pv,pi,qv,qi):
    u=np.union1d(pi,qi); pm,qm=dict(zip(pi,pv)),dict(zip(qi,qv))
    lp=np.array([pm.get(t,pv.min()) for t in u]+[OMITTED]); lq=np.array([qm.get(t,qv.min()) for t in u]+[OMITTED])
    c=np.append(np.ones(len(u)),V-len(u))
    def lse(x): m=x.max(); return m+np.log(np.exp(x-m).sum())
    logp=lp-lse(lp+np.log(c)); logq=lq-lse(lq+np.log(c)); return max(float((np.exp(logp)*c*(logp-logq)).sum()),0.0)
def run(a,b,split,label):
    av,ai=load(a); bv,bi=load(b); n=av.shape[0]; kl=np.zeros(n); t1=np.zeros(n,bool)
    for t in range(n):
        qv,qi=rebase(av[t],ai[t]); pv,pi=rebase(bv[t],bi[t]); kl[t]=pkl(pv,pi,qv,qi); t1[t]=qi[0]==pi[0]
    print(f"\n{label}\n  a={a}\n  b={b}")
    for nm,sl in (("question rows",slice(0,split)),("answer rows",slice(split,n)),("all rows",slice(0,n))):
        k=kl[sl]; print(f"  {nm:<14} n={k.size:>4}  p50={np.median(k):.4f} p95={np.quantile(k,.95):.4f} max={k.max():.3f}  over0.01={(k>.01).sum():>3} ({(k>.01).mean():.1%})  over0.1={(k>.1).sum():>3} ({(k>.1).mean():.1%})  top1={t1[sl].mean():.4f}")
for pn,split in (("prompt_0",55),("prompt_1",57)):
    run(f"vllm_bf16kv_fi.{pn}.safetensors",f"vllm_bf16kv.{pn}.safetensors",split,f"KERNEL-vs-KERNEL {pn} (question = rows 0..{split-1})")
    run(f"vllm_fp8kv.{pn}.safetensors",f"vllm_bf16kv.{pn}.safetensors",split,f"FP8-vs-bf16 KV {pn}")
