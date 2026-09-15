import csv,json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
base='''--replica_config_network_device h100_dgx --replica_config_device h100 --replica_config_nvlink_bandwidth 3600 --poisson_request_interval_generator_config_qps 512 --synthetic_request_generator_config_num_requests 1 --length_generator_config_type fixed --fixed_request_length_generator_config_decode_tokens 256 --fixed_request_length_generator_config_max_tokens 32768 --interval_generator_config_type poisson --cluster_config_num_replicas 4 --replica_config_pd_node_ratio 1 --global_scheduler_config_type lor --replica_scheduler_config_type sarathi --sarathi_scheduler_config_chunk_size 512 --replica_config_model_name qwen3-next-80B --replica_config_tensor_parallel_size 1 --replica_config_num_pipeline_stages 1 --random_forrest_execution_time_predictor_config_backend aicb --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 32768 --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 32768'''.split()
rows=[]
for n in [512,1024,2048,4096,8192,16384]:
    out=root/'results'/str(n); out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(root/'run_corrected.py'),*base,'--fixed_request_length_generator_config_prefill_tokens',str(n),'--metrics_config_output_dir',str(out)]
    (out/'command.json').write_text(json.dumps(cmd,indent=2))
    print('RUN',n,flush=True)
    with (out/'run.log').open('w') as f:subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
    files=list(out.rglob('request_metrics.csv')); assert len(files)==1,files
    with files[0].open() as f: records=list(csv.DictReader(f))
    assert len(records)==1
    r=records[0]; assert int(r['request_num_prefill_tokens'])==n and int(r['request_num_decode_tokens'])==256
    ttft=float(r['prefill_e2e_time'])*1000; e2e=float(r['request_e2e_time'])*1000
    result={'input_tokens':n,'output_tokens':256,'ttft_ms':ttft,'tpot_ms':(e2e-ttft)/255,'e2e_ms':e2e,'source':str(files[0])}
    rows.append(result); (root/'summary.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(result),flush=True)
with (root/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print('COMPLETE',flush=True)
