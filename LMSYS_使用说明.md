# LMSYS 本地全量长度池与 vLLM 压测

## 本地预处理

已下载的完整数据集位于容器 `/root/changle/Files/lmsys-chat-1m/data`，包含六个 Parquet 分片。宿主机对应 `/home/turbo_ops/changle/Files/lmsys-chat-1m/data`。

在 `vllm051` 容器内运行：

```bash
python3 /vllm-workspace/prepare_lmsys_chat_1m_lengths.py \
  --dataset-path /root/changle/Files/lmsys-chat-1m \
  --tokenizer /mnt/data02/000000/model/Llama-2-7b-hf-bin-export \
  --max-model-len 4096 \
  --output /root/changle/Files/lmsys_lengths.csv
```

预处理只读取本地数据和 tokenizer，无需 Hugging Face token，不发起下载。全部模型的对话都会处理，不限制请求数。每段对话取第一组相邻的非空 user/assistant 消息；prefill 包含 tokenizer 的特殊 token，decode 不加特殊 token。空消息、零 token 和输入加输出超过 4096 token 的记录会被跳过，不截断。多轮历史不拼接为 prompt，因此原始首轮长度分布仍可能偏短。

输出 CSV 保存 `request_id,num_prefill_tokens,num_decode_tokens`，包含所有合格记录，不含文本。预处理完成后打印数量、P50/P90/P99、最大值及长输入数量。CSV 在完整处理成功后才替换目标文件。

## 压测时随机抽样

在容器内运行，例如抽取 256 个请求：

```bash
python3 /vllm-workspace/vllm_benchmark_client_dataset.py \
  --host 127.0.0.1 --port 8088 \
  --model Llama-2-7b-hf \
  --request-lengths-file /root/changle/Files/lmsys_lengths.csv \
  --num-requests 256 --qps 2.8 --seed 36 \
  --warmup-requests 5 --ignore-eos \
  --output-dir /root/changle/Files \
  --trace-output-file /root/changle/Files/vllm_request_arrival_trace.csv
```

每次扫描完整 CSV，用 reservoir sampling 从所有记录中均匀随机抽取 N 行，不重复抽取，也不只选前 N 行。相同文件、请求数和 seed 会复现同一组样本及顺序；换 seed 可换样本，请求数超过合格记录总数会报错。抽样与到达时间使用独立的随机数生成器。结果 JSON 的 config 保存 `length_pool_size` 和 `sampled_dataset_request_ids`，可追溯每个测试请求的来源行；trace 的 request_id 仍为本次压测的顺序编号。

客户端继续使用合成 token ID prompt 与 `/v1/completions`，不发送真实文本。输入 token 数与 max_tokens 来自抽中记录，`ignore_eos` 默认开启，TPOT 只在返回 `finish_reason=length` 时按目标输出长度计算。到达时间沿用原脚本的分布；数据集不提供实际请求到达时间。

trace 仍位于容器 `/root/changle/Files/vllm_request_arrival_trace.csv`，宿主机 `/home/turbo_ops/changle/Files/vllm_request_arrival_trace.csv`；路径、文件名与列保持原有格式。每次压测会覆盖该 trace 及默认结果 JSON。Vidur 可以沿用原路径，长度缩放因子应为 1，上下文上限与 vLLM 一致。

## 文件位置

两个 Python 脚本均同步到宿主机 `/home/turbo_ops/changle/Files` 和容器 `/vllm-workspace`；容器也可以通过 `/root/changle/Files` 读取宿主机副本。完整长度 CSV 同样保存在 Files 的绑定目录，后续压测不需要重复进行 tokenization。
