#!/usr/bin/env bash
set -euo pipefail
source /home/turbo_ops/miniconda3/etc/profile.d/conda.sh
conda activate sarathi
cd /home/turbo_ops/changle/SimAI/vidur-alibabacloud
export CUDA_VISIBLE_DEVICES=6
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/turbo_ops/changle/SimAI/benchmark_runs/dp8_ep8_2x4/deps${PYTHONPATH:+:$PYTHONPATH}
# Require exact profiles; do not accept the legacy fallback path.
for shape in prefill:15360 decode:15360 decode:15614; do
  phase=${shape%:*}
  seq=${shape#*:}
  test -s "../aicb/results/workload/vidur-Qwen3-Next-80B-world_size8-tp1-pp1-ep8-bs1-seq${seq}-${phase}.csv" || {
    echo "Missing exact EP8 profile: $shape. Run profile.sh first." >&2
    exit 1
  }
done
python -u -m vidur.main \
  --replica_config_network_device h100_pairwise_nvlink \
  --replica_config_device h100 \
  --replica_config_nvlink_bandwidth 3600 \
  --replica_config_rdma_bandwidth 400 \
  --replica_config_pd_p2p_comm_bandwidth 400 \
  --poisson_request_interval_generator_config_qps 1024 \
  --synthetic_request_generator_config_num_requests 1 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 15360 \
  --fixed_request_length_generator_config_decode_tokens 256 \
  --fixed_request_length_generator_config_max_tokens 32768 \
  --interval_generator_config_type poisson \
  --cluster_config_num_replicas 8 \
  --replica_config_pd_node_ratio 1 \
  --global_scheduler_config_type lor \
  --replica_scheduler_config_type vllm \
  --vllm_scheduler_config_max_tokens_in_batch 32768 \
  --vllm_scheduler_config_block_size 32 \
  --replica_config_model_name qwen3-next-80B \
  --replica_config_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_memory_margin_fraction 0.15 \
  --random_forrest_execution_time_predictor_config_backend aicb \
  --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 32768 \
  --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 32768 \
  --metrics_config_output_dir ../benchmark_runs/dp8_ep8_2x4/output
