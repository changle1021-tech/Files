from pathlib import Path
import csv
import json
import math

out = Path('/home/turbo_ops/changle/SimAI/benchmark_runs/dp8_ep8_2x4')
assert (out / 'exit_code').read_text().strip() == '0'
assert (out / 'profiles.exit').read_text().strip() == '0'
log = (out / 'run.log').read_text()
for bad in ('AICB command failed', 'Expected CSV file was NOT created', 'Traceback', 'AICB data is empty', 'Fallback', 'ERROR'):
    assert bad not in log, bad
assert 'Prefill: tp=1 dp=8 ep=8' in log
assert 'Decode: tp=1 dp=8 ep=8' in log
assert 'Simulation ended at:' in log
run = sorted((out / 'output').iterdir())[-1]
config = json.loads((run / 'config.json').read_text())
cluster = config['cluster_config']
rc = cluster['replica_config']
assert cluster['num_replicas'] == 8
assert rc['tensor_parallel_size'] == rc['num_pipeline_stages'] == 1
assert rc['node_config']['num_devices_per_node'] == 4
assert rc['rdma_bandwidth'] == 400 and rc['nvlink_bandwidth'] == 3600
rows = list(csv.DictReader((run / 'request_metrics.csv').open()))
assert len(rows) == 1
row = rows[0]
assert float(row['request_num_prefill_tokens']) == 15360
assert float(row['request_num_decode_tokens']) == 256
for key in ('request_e2e_time', 'prefill_e2e_time', 'decode_time', 'completed_at'):
    assert math.isfinite(float(row[key])) and float(row[key]) > 0, key

profiles = out.parent.parent / 'aicb/results/workload'
profile_out = out / 'profiles'
profile_out.mkdir(exist_ok=True)
profile_checks = []
for phase, seq in [('prefill',15360), ('decode',15360), ('decode',15614)]:
    name = f'vidur-Qwen3-Next-80B-world_size8-tp1-pp1-ep8-bs1-seq{seq}-{phase}.csv'
    data = (profiles / name).read_text()
    (profile_out / name).write_text(data)
    records = list(csv.DictReader(data.splitlines(), delimiter='\t'))
    assert len({r['layer_id'] for r in records}) == 48
    moe = [r for r in records if r['layer_name'] == 'moe']
    assert len(moe) == 48 and all(float(r['comm_size']) > 0 for r in moe)
    # Confirm the active persistent cache was loaded from this exact CSV.
    cache_file = out.parent.parent / 'vidur-alibabacloud/data/aicb_workload/cache' / f'aicb-Qwen3-Next-80B-ws8-tp1-pp1-ep8-bs1-seq{seq}-{phase}.json'
    cache = json.loads(cache_file.read_text())
    for record in records:
        cached = cache[record['layer_id']][record['layer_name']]
        for metric in ('comp_time', 'comm_size'):
            assert float(cached[metric]) == float(record[metric]), (name, record, cached)
    profile_checks.append({'phase':phase, 'seq':seq, 'layers':48,
        'total_moe_communication_bytes':sum(float(r['comm_size']) for r in moe),
        'cache_file':str(cache_file), 'cache_keys':list(cache)})

summary = {'status':'PASS', 'dp':8, 'ep':8, 'tp':1, 'pp':1, 'nodes':2,
    'gpus_per_node':4, 'rdma_gbps':400, 'nvlink_gbps':3600,
    'request_e2e_seconds':float(row['request_e2e_time']),
    'prefill_seconds':float(row['prefill_e2e_time']),
    'decode_seconds':float(row['decode_time']),
    'output_dir':str(run), 'profiles':profile_checks}
(out / 'validation.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
