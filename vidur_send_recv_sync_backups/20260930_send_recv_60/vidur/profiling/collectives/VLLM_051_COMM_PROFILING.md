# vLLM 0.5.1 通信 profiling

`vllm_051_comm_profile.py` 在 vLLM 0.5.1 容器内运行。采样网格、worker 布局、有效性过滤、buffer 大小、轮数、计时聚合和 CSV 列定义都从旁边现有 Vidur 源码加载；不要手工改一份参数副本。size 输入单位是元素，CSV 的 `size` 字段按 FP16 的每元素 2 字节写成 bytes。默认 size 网格沿用 Vidur，包括其中的重复点。

所有命令假定宿主机 SimAI checkout 已挂载到容器的 `/root/changle/SimAI`，输出路径挂载到 `/root/changle/Files`。推荐显式指定 `--mode decode`：`--mode` 默认值也是 `decode`，并支持 `prefill`、`decode`、`both`。

## all_reduce

建议显式指定本机可用的 1、2、4 卡布局：

```bash
sudo nerdctl --namespace changle exec -it vllm051 python3 /root/changle/SimAI/vidur-alibabacloud/vidur/profiling/collectives/vllm_051_comm_profile.py --mode decode --collective all_reduce --num_workers_per_node_combinations 1 2 4 --output_dir /root/changle/Files/vidur_vllm051_comm
```

默认 worker-per-node 列表仍是 `1 2 4 8`。在每节点只有 4 张可见 GPU 的集群上，布局 8 会被记录为不可用并跳过；显式写 `1 2 4` 可避免该提示。

## send_recv

`send_recv` 只支持两 worker，使用每节点 2 卡布局：

```bash
sudo nerdctl --namespace changle exec -it vllm051 python3 /root/changle/SimAI/vidur-alibabacloud/vidur/profiling/collectives/vllm_051_comm_profile.py --mode decode --collective send_recv --num_workers_per_node_combinations 2 --output_dir /root/changle/Files/vidur_vllm051_comm
```

## 只查看计划

`--plan-only` 只加载 Vidur 参数并打印合同、网格摘要和源码哈希，不启动 Ray 或 GPU：

```bash
sudo nerdctl --namespace changle exec -it vllm051 python3 /root/changle/SimAI/vidur-alibabacloud/vidur/profiling/collectives/vllm_051_comm_profile.py --mode decode --collective all_reduce --num_workers_per_node_combinations 1 2 4 --output_dir /root/changle/Files/vidur_vllm051_comm --plan-only
```

## 接入 Vidur

一次运行会为所选 collective 写入 `<output_dir>/collective/<时间戳>/<collective>.csv` 和对应的 `<collective>.metadata.json`，例如 `all_reduce.csv`、`all_reduce.metadata.json`。CSV 保持 Vidur 的 `time_stats.*` 与结果字段，并保留 pandas 默认 index 列。使用容器中可访问的 CSV 路径，将以下参数追加到原仿真命令（将时间戳替换成实际目录名）：

```bash
--random_forrest_execution_time_predictor_config_backend vidur \
--random_forrest_execution_time_predictor_config_all_reduce_input_file "/root/changle/Files/vidur_vllm051_comm/collective/<时间戳>/all_reduce.csv"
```

`PP > 1` 时还需追加 send/recv 表：

```bash
--random_forrest_execution_time_predictor_config_send_recv_input_file "/root/changle/Files/vidur_vllm051_comm/collective/<send_recv时间戳>/send_recv.csv"
```

路径必须能被运行 Vidur 的环境访问；例如容器内 `/root/changle/Files` 对应宿主机 `/home/turbo_ops/changle/Files`。`--mode both` 会测量 prefill 与 decode，但供原版 Vidur 使用的单一 CSV 全部取 decode 结果，保留 Vidur 原有网格点（包括重复点）。metadata 记录 `profiled_modes`、`csv_mode`、`mode_row_counts` 和 `mode_row_audits`。因此 prefill/decode 在原版 Vidur 中统一使用 decode 标定值，单表不分别表示两种模式。

## 采集约定

prefill 使用 eager，每轮 11 次 collective 调用；decode 使用 CUDA graph，建图前预热 5 次、捕获 3 次调用，每轮 replay 一次（一次 replay 执行图内 3 次调用）。两者都按 Vidur 默认采集 3 轮。两阶段只改变执行模式，因此不能把它们描述为执行次数相同。没有额外增加 warmup 或采样轮数覆盖开关。

全程关闭 vLLM custom all-reduce。FP16、rank 0 结果策略和 Kineto NCCL 事件计时沿用 Vidur。vLLM 0.5.1 没有对应 PyNccl 原语的 `all_gather`、`broadcast`、`reduce_scatter`、`all_to_all` 使用其 `device_group` 上的 torch/NCCL 集合通信；metadata 将其标为 torch/NCCL group 实现，不称作原生 PyNccl。all-reduce 和 send/recv 的 prefill 走 torch/NCCL，decode 走 vLLM PyNccl 的 CUDA Graph replay；PyNccl 不可用时 decode 会报错。
