"""Measure vLLM 0.11.2 variable-size AG/RS communication on four H100s.
Only DP rank 0 has request tokens. Hidden state and router logits BF16.
"""
import os,json,statistics
from pathlib import Path
import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator
rank=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank)
dist.init_process_group('nccl')
group=GroupCoordinator([list(range(4))],rank,'nccl',True,group_name='profile_dp')
results={}
for m in (1,512):
    sizes=[m,0,0,0]
    x=torch.ones(sizes[rank],2048,device='cuda',dtype=torch.bfloat16)
    logits=torch.ones(sizes[rank],512,device='cuda',dtype=torch.bfloat16)
    def call():
        hidden,r=group.all_gatherv([x,logits],dim=0,sizes=sizes)
        out=group.reduce_scatterv(hidden,dim=0,sizes=sizes)
        return out
    for _ in range(10):out=call()
    torch.cuda.synchronize(); dist.barrier()
    samples=[]
    for _ in range(7):
        dist.barrier()
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(50):out=call()
        b.record();b.synchronize()
        t=torch.tensor(a.elapsed_time(b)/50,device='cuda')
        dist.all_reduce(t,op=dist.ReduceOp.MAX);samples.append(t.item())
    results[str(m)]={'median_ms':statistics.median(samples),'samples_ms':samples,'sizes':sizes}
if rank==0:
    Path('/tmp/qwen3next_corrected/communication.json').write_text(json.dumps({'world_size':4,'vllm':'0.11.2','method':'actual GroupCoordinator.all_gatherv + reduce_scatterv, BF16 hidden/router, CUDA event max rank, 7x50 calls; includes launch gaps','profiles':results},indent=2))
    print(json.dumps(results),flush=True)
dist.barrier();group.destroy();dist.destroy_process_group()
