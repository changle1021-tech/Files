"""Qwen3-Next BF16 operator profiles for Vidur, TP1/PP1/EP4, one active request.
Uses vLLM 0.11.2 kernels; random weights, GPU CUDA-event timing, no HTTP timing.
Profiles full chunk GDN, correct projections, causal attention with growing KV,
and routed + shared experts. Communication is left to the simulator.
"""
import argparse,json,time,statistics,gc
from pathlib import Path
import torch
import torch.nn.functional as F
import vllm
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule,fused_recurrent_gated_delta_rule
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn,causal_conv1d_update
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts,fused_topk
from vllm.model_executor.layers.layernorm import GemmaRMSNorm,RMSNormGated
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.qwen3_next import fused_gdn_gating
from vllm.vllm_flash_attn import flash_attn_varlen_func

def rand(*shape): return torch.randn(*shape,device='cuda',dtype=torch.bfloat16)*0.02

def bench(fn):
    # GDN prefill prepares sequence metadata on the host and cannot be captured
    # as a single CUDA graph. Sum actual CUDA kernel durations instead, matching
    # AICB's GPU operator timing scope and excluding Python launch gaps.
    samples=[]; walls=[]
    with torch.inference_mode():
        for _ in range(4): out=fn()
        torch.cuda.synchronize()
        assert torch.isfinite(out).all(), 'nonfinite profile output'
        for _ in range(3):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                start=time.perf_counter()
                for _ in range(10): out=fn()
                torch.cuda.synchronize(); walls.append((time.perf_counter()-start)*1000/10)
            events=[e for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
            assert events, 'no CUDA kernel events'
            samples.append(sum(e.device_time_total for e in events)/1000/10)
    return {'median_ms':statistics.median(samples),'samples_ms':samples,'profiled_wall_ms':walls,'method':'sum of CUDA device events, 3 x 10 forwards; CPU gaps excluded'}

class GDN:
    def __init__(self,m,initial):
        self.m=m; self.initial=initial
        self.x=rand(m,2048); self.wq=rand(12288,2048); self.wba=rand(64,2048); self.wo=rand(2048,4096)
        self.conv=rand(8192,4); self.cs=torch.zeros(1,3,8192,device='cuda',dtype=torch.bfloat16).transpose(1,2)
        self.ss=torch.zeros(1,32,128,128,device='cuda',dtype=torch.float32)
        self.cu=torch.tensor([0,m],device='cuda',dtype=torch.int32); self.idx=torch.zeros(1,device='cuda',dtype=torch.int32)
        self.has=torch.tensor([initial],device='cuda',dtype=torch.bool)
        self.alog=torch.zeros(32,device='cuda'); self.dt=torch.zeros(32,device='cuda')
        self.pre=GemmaRMSNorm(2048,eps=1e-6).cuda().to(torch.bfloat16)
        self.norm=RMSNormGated(128,eps=1e-6,group_size=None,norm_before_gate=True,device='cuda',dtype=torch.bfloat16)
    def __call__(self):
        m=self.m; x=self.pre(self.x)
        zall=F.linear(x,self.wq).reshape(m,16,768)
        q,k,v,z=torch.split(zall,[128,128,256,256],-1)
        v=v.reshape(m,32,128); z=z.reshape(m,32,128)
        b,a=F.linear(x,self.wba).reshape(m,16,4).split(2,-1)
        b=b.reshape(m,32); a=a.reshape(m,32)
        mixed=torch.cat([q.reshape(m,-1),k.reshape(m,-1),v.reshape(m,-1)],-1)
        if m>1:
            mixed=causal_conv1d_fn(mixed.T,self.conv,None,self.cs,self.cu,cache_indices=self.idx,has_initial_state=self.has,activation='silu').T
        else:
            mixed=causal_conv1d_update(mixed,self.cs,self.conv,None,'silu',conv_state_indices=self.idx)
        q,k,v=mixed.split([2048,2048,4096],-1)
        q=q.reshape(1,m,16,128).contiguous(); k=k.reshape(1,m,16,128).contiguous(); v=v.reshape(1,m,32,128).contiguous()
        g,beta=fused_gdn_gating(self.alog,a,b,self.dt)
        if m>1:
            out,ss=chunk_gated_delta_rule(q,k,v,g,beta,initial_state=self.ss,output_final_state=True,cu_seqlens=self.cu,head_first=False,use_qk_l2norm_in_kernel=True)
            self.ss.copy_(ss)
        else:
            out,ss=fused_recurrent_gated_delta_rule(q,k,v,g,beta,initial_state=self.ss,inplace_final_state=True,cu_seqlens=self.cu,ssm_state_indices=self.idx,use_qk_l2norm_in_kernel=True)
        out=self.norm(out.reshape(-1,128),z.reshape(-1,128))
        return F.linear(out.reshape(m,4096),self.wo)

class Attention:
    def __init__(self,m,kv):
        self.m=m; self.kv=kv; self.x=rand(m,2048)
        self.wq=rand(9216,2048); self.wo=rand(2048,4096)
        self.key=rand(kv,2,256); self.val=rand(kv,2,256)
        self.pre=GemmaRMSNorm(2048,eps=1e-6).cuda().to(torch.bfloat16)
        self.qnorm=GemmaRMSNorm(256,eps=1e-6).cuda().to(torch.bfloat16); self.knorm=GemmaRMSNorm(256,eps=1e-6).cuda().to(torch.bfloat16)
        self.rope=get_rope(256,rotary_dim=64,max_position=32768,base=10000000,rope_scaling=None).cuda()
        self.pos=torch.arange(kv-m,kv,device='cuda'); self.cuq=torch.tensor([0,m],device='cuda',dtype=torch.int32); self.cuk=torch.tensor([0,kv],device='cuda',dtype=torch.int32)
    def __call__(self):
        m=self.m
        qg,k,v=F.linear(self.pre(self.x),self.wq).split([8192,512,512],-1)
        q,gate=qg.reshape(m,16,512).chunk(2,-1)
        q=self.qnorm(q).reshape(m,4096); k=self.knorm(k.reshape(m,2,256)).reshape(m,512)
        q,k=self.rope(self.pos,q,k)
        self.key[-m:].copy_(k.reshape(m,2,256)); self.val[-m:].copy_(v.reshape(m,2,256))
        out=flash_attn_varlen_func(q.reshape(m,16,256),self.key,self.val,max_seqlen_q=m,cu_seqlens_q=self.cuq,max_seqlen_k=self.kv,cu_seqlens_k=self.cuk,causal=True,fa_version=3)
        out=out.reshape(m,4096)*torch.sigmoid(gate.reshape(m,4096))
        return F.linear(out,self.wo)

class MoE:
    def __init__(self,m):
        self.x=rand(m,2048); self.wgate=rand(512,2048)
        self.w1=rand(128,1024,2048); self.w2=rand(128,2048,512)
        self.sw1=rand(1024,2048); self.sw2=rand(2048,512); self.sgate=rand(1,2048)
        self.norm=GemmaRMSNorm(2048,eps=1e-6).cuda().to(torch.bfloat16)
        self.map=torch.full((512,),-1,device='cuda',dtype=torch.int32)
        self.rank=0
    def set_rank(self,r):
        self.rank=r; self.map.fill_(-1); self.map[r*128:(r+1)*128]=torch.arange(128,device='cuda',dtype=torch.int32)
    def __call__(self):
        x=self.norm(self.x)
        weights,ids,_=fused_topk(hidden_states=x,gating_output=F.linear(x,self.wgate),topk=10,renormalize=True)
        out=fused_experts(x,self.w1,self.w2,weights,ids,global_num_experts=512,expert_map=self.map)
        # One active DP rank owns the non-sharded shared expert.
        if self.rank==0:
            a,b=F.linear(x,self.sw1).chunk(2,-1)
            out=out+F.linear(F.silu(a)*b,self.sw2)*torch.sigmoid(F.linear(x,self.sgate))
        return out

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    assert vllm.__version__=='0.11.2',vllm.__version__
    torch.manual_seed(42); torch.set_num_threads(1)
    path=Path(args.out); path.mkdir(parents=True,exist_ok=True)
    result={'vllm':vllm.__version__,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'dtype':'bfloat16','seed':42,'profiles':{},'notes':['TP1 PP1 EP4; one active request','Synthetic weights; local experts measured per rank; maximum rank time used','GPU times exclude host scheduling and HTTP; communication modeled separately','Shared expert included on active rank; no overlap with routed expert in this profiler']}
    def put(key,fn):
        print('PROFILE',key,flush=True); result['profiles'][key]=bench(fn)
        path.joinpath('profiles.json').write_text(json.dumps(result,indent=2)); print(key,result['profiles'][key]['median_ms'],flush=True)
    for m in (512,1):
        obj=GDN(m,False); put(f'gdn-{m}-first',obj); del obj; gc.collect(); torch.cuda.empty_cache()
        if m>1:
            obj=GDN(m,True); put(f'gdn-{m}-continuation',obj); del obj; gc.collect(); torch.cuda.empty_cache()
        obj=MoE(m)
        for rank in range(4):
            obj.set_rank(rank); put(f'moe-{m}-rank{rank}',obj)
        del obj; gc.collect(); torch.cuda.empty_cache()
        contexts=[512] if args.smoke else (list(range(512,16385,512)) if m==512 else sorted({n+d for n in [512,1024,2048,4096,8192,16384] for d in [0,254]}))
        for kv in contexts:
            obj=Attention(m,kv); put(f'attention-{m}-kv{kv}',obj); del obj; gc.collect(); torch.cuda.empty_cache()
    print('COMPLETE',flush=True)
if __name__=='__main__': main()
