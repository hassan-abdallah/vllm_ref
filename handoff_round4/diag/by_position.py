"""Per-position-bucket diagnostics from the dumps: top-1 accuracy, mean NLL of the actual next token
(teacher-forced perplexity proxy; rows whose actual token is outside top-256 are counted, not scored),
top-32 entropy of the reference, and kernel-vs-kernel top-1 agreement."""
import sys, json, numpy as np
from safetensors.torch import load_file
F={"bf16kv":"vllm_bf16kv.prompt_0.safetensors","bf16kv_fi":"vllm_bf16kv_fi.prompt_0.safetensors","fp8kv":"vllm_fp8kv.prompt_0.safetensors"}
if len(sys.argv)>1: F={"bf16kv":sys.argv[1]}; off=int(sys.argv[2]) if len(sys.argv)>2 else 0
else: off=0
D={k:load_file(v) for k,v in F.items()}
ids=D["bf16kv"]["input_ids"].numpy(); n=len(ids)
B=[(0,256),(256,512),(512,1024),(1024,1536),(1536,2048),(2048,2176)]
B=[(lo+off,min(hi+off,n)) for lo,hi in B]
def stats(d):
    lp=d["topk_logprobs"].float().numpy(); ti=d["topk_token_ids"].numpy()
    top1=ti[:,0]; acc=np.r_[top1[:-1]==ids[1:],False]
    nll=np.full(n,np.nan)
    for t in range(n-1):
        m=np.where(ti[t]==ids[t+1])[0]
        if m.size: nll[t]=-lp[t,m[0]]
    p=np.exp(lp[:,:32]-lp[:,:1]); p/=p.sum(1,keepdims=True); H=-(p*np.log(np.maximum(p,1e-300))).sum(1)
    return acc,nll,H,top1
S={k:stats(d) for k,d in D.items()}
print(f"{'rows':<12}{'acc':>7}{'NLL':>8}{'PPL':>9}{'miss':>6}{'H(nats)':>9}"+("" if len(F)==1 else f"{'k-vs-k top1':>13}{'fp8-vs-bf16':>13}"))
for lo,hi in B:
    acc,nll,H,t1=S["bf16kv"]; sl=slice(lo,hi-1 if hi==n else hi); v=nll[sl]; m=np.isnan(v)
    line=f"{f'{lo}:{hi}':<12}{acc[sl].mean():>7.3f}{np.nanmean(v):>8.3f}{np.exp(np.nanmean(v)):>9.2f}{m.sum():>6}{H[lo:hi].mean():>9.3f}"
    if len(F)>1: line+=f"{(S['bf16kv_fi'][3][lo:hi]==t1[lo:hi]).mean():>13.3f}{(S['fp8kv'][3][lo:hi]==t1[lo:hi]).mean():>13.3f}"
    print(line)
acc,nll,H,_=S["bf16kv"]; a=slice(off,2048+off-1); c=slice(2048+off,n-1)
print(f"\nprompt rows : acc {acc[a].mean():.3f}  PPL {np.exp(np.nanmean(nll[a])):.2f}  (actual token outside top-256 on {np.isnan(nll[a]).sum()} rows)")
print(f"cont.  rows : acc {acc[c].mean():.3f}  PPL {np.exp(np.nanmean(nll[c])):.2f}  (outside top-256 on {np.isnan(nll[c]).sum()} rows)")
