from types import SimpleNamespace as NS
import math
from vidur.entities.execution_time import ExecutionTime

for phase in ('prefill', 'decode'):
    for ep, rdma, expected in ((8, 400, .02), (8, 200, .04), (4, 400, 1e9 / (3600e9 / 8))):
        obj = ExecutionTime.__new__(ExecutionTime)
        obj._replica_config = NS(
            phase=phase, expert_model_parallel_size=ep, world_size=8,
            num_pipeline_stages=1, node_config=NS(num_devices_per_node=4),
            nvlink_bandwidth=3600, rdma_bandwidth=rdma,
        )
        obj._aicb_data = {0: {'moe': {'comp_time': 0, 'comm_size': 1e9}}}
        actual = obj._get_moe_layer_execution_time_from_aicb(0)
        assert math.isclose(actual, expected, rel_tol=1e-12), (phase, ep, actual, expected)
        print(f'PASS {phase} EP={ep} RDMA={rdma} Gbps: 1 GB transfer = {actual:.9f} s')
print('PASS: EP8 uses 400 Gbps inter-node in both phases; EP4 within-node uses 3600 Gbps NVLink.')
