#!/usr/bin/env bash
set -euo pipefail
source /home/turbo_ops/miniconda3/etc/profile.d/conda.sh
conda activate sarathi
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
run_dir=/home/turbo_ops/changle/SimAI/benchmark_runs/dp8_ep8_2x4
cd /home/turbo_ops/changle/SimAI/DeepGEMM
DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=0 MAX_JOBS=2 python -m pip install \
  --no-deps --no-build-isolation --no-cache-dir --upgrade --target "$run_dir/deps" .
# Force a new object file with the current torch C++ ABI and CUDA headers.
DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=0 MAX_JOBS=2 python setup.py build_ext \
  --force --build-temp "$run_dir/build-temp" --build-lib "$run_dir/deps"
python -m pip install --no-deps --upgrade --target "$run_dir/deps" flashinfer-python==0.3.1
cd "$run_dir"
PYTHONPATH="$run_dir/deps${PYTHONPATH:+:$PYTHONPATH}" python -c \
  'import torch, deep_gemm, flashinfer; print(torch.__version__, deep_gemm.__file__, flashinfer.__version__)'
