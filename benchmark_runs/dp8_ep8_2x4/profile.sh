#!/usr/bin/env bash
set -euo pipefail
source /home/turbo_ops/miniconda3/etc/profile.d/conda.sh
conda activate sarathi
export CUDA_VISIBLE_DEVICES=6
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/turbo_ops/changle/SimAI/benchmark_runs/dp8_ep8_2x4/deps${PYTHONPATH:+:$PYTHONPATH}
cd /home/turbo_ops/changle/SimAI/aicb
for shape in prefill:15360 decode:15360 decode:15614; do
  phase=${shape%:*}
  seq=${shape#*:}
  python -u -m workload_generator.Vidur_workload_generator \
    Qwen3-Next-80B ./scripts/inference_configs/qwen3_next_default.json \
    --seq_length "$seq" --micro_batch 1 --world_size 8 \
    --tensor_model_parallel_size 1 --expert_model_parallel_size 8 \
    --aiob_enable --phase "$phase"
  test -s "results/workload/vidur-Qwen3-Next-80B-world_size8-tp1-pp1-ep8-bs1-seq${seq}-${phase}.csv"
done
