"""Inject independently measured BF16 profiles into isolated Vidur AICB backend.
No fitting to observed vLLM request latencies. Fails on unsupported/missing shapes.
"""
import csv,json,os,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
repo=ROOT/'vidur-alibabacloud'
os.chdir(repo); sys.path.insert(0,str(repo))
profile=json.loads((ROOT/'profiles/profiles.json').read_text())
assert profile['vllm']=='0.11.2' and profile['dtype']=='bfloat16'
p=profile['profiles']
communication=json.loads((ROOT/'profiles/communication.json').read_text())
assert communication['world_size']==4
from vidur.entities.execution_time import ExecutionTime

def ms(key):return p[key]['median_ms']
def data(self):
    c=self._replica_config
    assert c.tensor_parallel_size==1 and c.num_pipeline_stages==1 and c.world_size==4
    phase=c.phase; m=c.seq_len if phase=='prefill' else c.batch_size
    assert m in (1,512), (phase,m)
    if phase=='prefill':
        kv=c.profile_kv_tokens
        att=ms(f'attention-512-kv{kv}')
        gdn=ms('gdn-512-first' if kv==m else 'gdn-512-continuation')
    else:
        kv=c.seq_len; n=c.profile_prompt_tokens
        assert n<=kv<=n+254,(kv,n)
        alpha=(kv-n)/254
        att=(1-alpha)*ms(f'attention-1-kv{n}')+alpha*ms(f'attention-1-kv{n+254}')
        gdn=ms('gdn-1-first')
    moe=max(ms(f'moe-{m}-rank{r}') for r in range(4))
    # Keep BF16 dispatch/combine bytes for auditing. The execution method
    # below uses measured variable-size AG/RS latency, not bytes/bandwidth.
    comm=2*m*2048*10*2
    result={i:{'attention':{'comp_time':(att if (i+1)%4==0 else gdn)*1e6,'comm_size':0},'moe':{'comp_time':moe*1e6,'comm_size':comm}} for i in range(48)}
    self._aicb_data=result
    return result

def moe_time(self,layer_id):
    if self._aicb_data is None:self._aicb_data=self._load_aicb_data()
    row=self._aicb_data[layer_id]['moe']
    c=self._replica_config
    m=c.seq_len if c.phase=='prefill' else c.batch_size
    comm_ms=communication['profiles'][str(m)]['median_ms']
    return row['comp_time']*1e-9+comm_ms*1e-3
ExecutionTime._load_aicb_data=data
ExecutionTime._get_moe_layer_execution_time_from_aicb=moe_time
runpy.run_module('vidur.main',run_name='__main__')
