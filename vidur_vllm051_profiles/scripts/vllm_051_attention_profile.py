#!/usr/bin/env python3
"""Profile vLLM 0.5.1 FlashAttention kernels into Vidur's attention.csv schema.

This measures the actual vLLM FlashAttention prefill/decode calls and its KV
cache-write custom op separately, matching Vidur's three prediction targets.
It does not include QKV/output projections; those are in mlp.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from transformers import AutoConfig
from vllm import _custom_ops as ops
from vllm.attention.selector import get_attn_backend
from vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from vllm_051_profile_common import (check_vllm_version, close_tp, gpu_times,
                                     init_tp, put_stats, run_all_tp, write_csv)


DEFAULT_OUTPUT = "/root/changle/Files/vidur_vllm051_profiles/attention.csv"


def prefill_grid(max_seq_len: int) -> List[int]:
    grid = (list(range(64, 129, 16)) + list(range(128, 1025, 32))
            + list(range(1024, 4097, 64)) + list(range(4096, 16385, 128))
            + list(range(16384, 65537, 256)))
    return [value for value in grid if value <= max_seq_len]


def kv_grid(max_seq_len: int) -> List[int]:
    grid = (list(range(0, 1025, 32)) + list(range(1024, 4097, 64))
            + list(range(4096, 65537, 256)))
    return [value for value in grid if value < max_seq_len]


def batch_grid(min_batch_size: int, max_batch_size: int) -> List[int]:
    grid = list(range(1, 129)) + list(range(128, 1025, 8))
    return [value for value in grid
            if min_batch_size <= value <= max_batch_size]


def cases(args: argparse.Namespace, tp: int | None = None) -> List[Tuple[int, int, int, bool]]:
    if args.cases_from_csv:
        result = []
        with open(args.cases_from_csv, newline="") as stream:
            for row in csv.DictReader(stream):
                row_tp = int(row["num_tensor_parallel_workers"])
                if (row_tp not in args.tensor_parallel_sizes
                        or (tp is not None and row_tp != tp)):
                    continue
                if int(row["block_size"]) != args.block_size:
                    raise ValueError("Input CSV block_size differs from --block-size")
                if int(row["max_model_len"]) != args.max_model_len:
                    raise ValueError("Input CSV max_model_len differs from --max-model-len")
                prefill_text = row["is_prefill"].lower()
                if prefill_text not in ("true", "false"):
                    raise ValueError(f"Invalid is_prefill value: {row['is_prefill']}")
                is_prefill = prefill_text == "true"
                if ((args.only_decode and is_prefill)
                        or (args.only_prefill and not is_prefill)):
                    continue
                result.append((int(row["prefill_chunk_size"]),
                               int(row["kv_cache_size"]), int(row["batch_size"]),
                               is_prefill))
        invalid = [case for case in result
                   if case[0] + case[1] > args.max_model_len
                   or case[2] * (case[0] + case[1] + (not case[3])) > args.max_kv_tokens]
        if invalid:
            raise ValueError(f"{len(invalid)} input CSV cases exceed the length or "
                             f"memory limits; first invalid case: {invalid[0]}")
        # Keep repeated source rows: the original profiler also measured
        # duplicate grid-boundary cases independently.
        return result

    chunks = args.prefill_chunk_sizes or prefill_grid(args.max_seq_len)
    kv_sizes = args.kv_cache_sizes or kv_grid(args.max_seq_len)
    batches = args.batch_sizes or batch_grid(args.min_batch_size,
                                              args.max_batch_size)
    result = []
    if not args.only_decode:
        for chunk in chunks:
            for kv in kv_sizes:
                if kv + chunk <= args.max_seq_len:
                    result.append((chunk, kv, 1, True))
    if not args.only_prefill:
        for kv in kv_sizes:
            if kv < 1:
                continue
            for batch in batches:
                if batch >= 1 and batch * (kv + 1) <= args.max_kv_tokens:
                    result.append((0, kv, batch, False))
    result = [case for case in result
              if case[0] + case[1] <= args.max_model_len
              and case[2] * (case[0] + case[1] + (not case[3])) <= args.max_kv_tokens]
    return sorted(result, key=lambda x: (not x[3], x[0], x[1], x[2]))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/mnt/data02/000000/model/Llama-2-7b-hf")
    p.add_argument("--tensor-parallel-sizes", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--min-batch-size", type=int, default=1)
    p.add_argument("--max-batch-size", type=int, default=128)
    p.add_argument("--batch-sizes", type=int, nargs="+")
    p.add_argument("--prefill-chunk-sizes", type=int, nargs="+")
    p.add_argument("--kv-cache-sizes", type=int, nargs="+")
    p.add_argument("--cases-from-csv", help="Replay every matching attention row from an existing profiling CSV")
    p.add_argument("--only-decode", action="store_true")
    p.add_argument("--only-prefill", action="store_true")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-kv-tokens", type=int, default=524288)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--part-output", help=argparse.SUPPRESS)
    return p


def profile_case(args: argparse.Namespace, config, tp: int,
                 case: Tuple[int, int, int, bool], cache: torch.Tensor) -> dict:
    chunk, kv_size, batch, is_prefill = case
    block_size = args.block_size
    heads = config.num_attention_heads // tp
    kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads) // tp
    head_size = config.hidden_size // config.num_attention_heads
    token_count = chunk if is_prefill else batch
    blocks_per_seq = math.ceil((kv_size + (chunk if is_prefill else 1)) / block_size)
    num_blocks = batch * blocks_per_seq
    key_cache, value_cache = cache[0, :num_blocks], cache[1, :num_blocks]
    q = torch.randn((token_count, heads, head_size), device="cuda", dtype=torch.float16)
    k = torch.randn((token_count, kv_heads, head_size), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    if is_prefill:
        slots = torch.arange(kv_size, kv_size + chunk, device="cuda",
                             dtype=torch.long)
        block_table = torch.arange(blocks_per_seq, device="cuda",
                                   dtype=torch.int32).view(1, -1)
    else:
        slots = torch.arange(batch, device="cuda", dtype=torch.long)
        slots = slots * (blocks_per_seq * block_size) + kv_size
        block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32)
        block_table = block_table.view(batch, blocks_per_seq)

    def save_kv():
        return ops.reshape_and_cache_flash(k, v, key_cache, value_cache,
                                           slots, "auto")

    save_kv()
    torch.cuda.synchronize()
    scale = head_size ** -0.5
    if is_prefill and kv_size == 0:
        offsets = torch.tensor([0, chunk], device="cuda", dtype=torch.int32)

        def attention():
            return flash_attn_varlen_func(
                q=q, k=k, v=v, cu_seqlens_q=offsets, cu_seqlens_k=offsets,
                max_seqlen_q=chunk, max_seqlen_k=chunk,
                softmax_scale=scale, causal=True)
    elif is_prefill:
        q_offsets = torch.tensor([0, chunk], device="cuda", dtype=torch.int32)
        seq_offsets = torch.tensor([0, kv_size + chunk], device="cuda",
                                   dtype=torch.int32)

        def attention():
            return flash_attn_varlen_func(
                q=q, k=key_cache, v=value_cache,
                cu_seqlens_q=q_offsets, cu_seqlens_k=seq_offsets,
                max_seqlen_q=chunk, max_seqlen_k=kv_size + chunk,
                softmax_scale=scale, causal=True, block_table=block_table)
    else:
        cache_lengths = torch.full((batch,), kv_size + 1, device="cuda",
                                   dtype=torch.int32)

        def attention():
            return flash_attn_with_kvcache(
                q.unsqueeze(1), key_cache, value_cache,
                block_table=block_table, cache_seqlens=cache_lengths,
                softmax_scale=scale, causal=True).squeeze(1)

    row = {
        "n_embd": config.hidden_size,
        "n_q_head": config.num_attention_heads,
        "n_kv_head": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "block_size": block_size,
        "num_tensor_parallel_workers": tp,
        "max_model_len": args.max_model_len,
        "batch_size": batch,
        "prefill_chunk_size": chunk,
        "kv_cache_size": kv_size,
        "is_prefill": is_prefill,
        "attention_backend": "vllm_0.5.1_flash_attn",
    }
    put_stats(row, "attn_kv_cache_save", gpu_times(save_kv, args.warmup,
                                                     args.repetitions))
    name = "attn_prefill" if is_prefill else "attn_decode"
    put_stats(row, name, gpu_times(attention, args.warmup, args.repetitions))
    # The corresponding vLLM views do not launch CUDA kernels. Keep these
    # targets explicit so old/new CSV schemas and timing boundaries align.
    put_stats(row, "attn_input_reshape", [0.0] * args.repetitions)
    put_stats(row, "attn_output_reshape", [0.0] * args.repetitions)
    return row


def worker(args: argparse.Namespace, combinations: Sequence[tuple]) -> None:
    rank, _, tp = init_tp()
    try:
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
        if config.model_type != "llama":
            raise ValueError("This profiler currently supports FP16 Llama models")
        head_size = config.hidden_size // config.num_attention_heads
        kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        if config.num_attention_heads % tp or kv_heads % tp:
            raise ValueError("Attention and KV heads must divide tensor parallel size")
        backend = get_attn_backend(config.num_attention_heads // tp, head_size,
                                   kv_heads // tp, None, torch.float16, "auto",
                                   args.block_size)
        if backend.get_name() != "flash-attn":
            raise RuntimeError(f"Expected vLLM FlashAttention, got {backend.get_name()}")
        max_blocks = max(case[2] * math.ceil(
            (case[1] + (case[0] if case[3] else 1)) / args.block_size)
                         for case in combinations)
        cache = torch.empty((2, max_blocks, args.block_size, kv_heads // tp,
                             head_size), device="cuda", dtype=torch.float16)
        cache.normal_()
        rows = []
        for index, case in enumerate(combinations, 1):
            rows.append(profile_case(args, config, tp, case, cache))
            if rank == 0 and (index == 1 or index % 100 == 0
                              or index == len(combinations)):
                print(f"TP={tp} attention case {index}/{len(combinations)}: {case}",
                      flush=True)
        if rank == 0:
            write_csv(args.part_output, rows)
    finally:
        close_tp()


def main() -> None:
    args = parser().parse_args()
    check_vllm_version()
    if args.only_decode and args.only_prefill:
        raise ValueError("Choose at most one of --only-decode and --only-prefill")
    if args.max_seq_len < 2 or args.max_model_len < 2 or args.block_size < 1:
        raise ValueError("Sequence lengths and block size must be positive")
    if args.warmup < 0 or args.repetitions < 2:
        raise ValueError("warmup >= 0 and repetitions >= 2 are required")
    if args.worker:
        combinations = cases(args, tp=int(os.environ["WORLD_SIZE"]))
        if not combinations:
            raise ValueError("No attention cases match this TP size")
        worker(args, combinations)
        return
    for tp in args.tensor_parallel_sizes:
        if not cases(args, tp=tp):
            raise ValueError(f"No attention cases match TP={tp}")
    forwarded = ["--model", args.model, "--tensor-parallel-sizes",
                 *map(str, args.tensor_parallel_sizes),
                 "--max-model-len", str(args.max_model_len),
                 "--max-seq-len", str(args.max_seq_len), "--min-batch-size",
                 str(args.min_batch_size), "--max-batch-size", str(args.max_batch_size),
                 "--block-size", str(args.block_size), "--max-kv-tokens",
                 str(args.max_kv_tokens), "--warmup", str(args.warmup),
                 "--repetitions", str(args.repetitions)]
    for option in ("batch_sizes", "prefill_chunk_sizes", "kv_cache_sizes"):
        value = getattr(args, option)
        if value:
            forwarded += ["--" + option.replace("_", "-"), *map(str, value)]
    if args.only_decode:
        forwarded.append("--only-decode")
    if args.only_prefill:
        forwarded.append("--only-prefill")
    if args.cases_from_csv:
        forwarded += ["--cases-from-csv", args.cases_from_csv]
    run_all_tp(str(Path(__file__).resolve()), forwarded,
               args.tensor_parallel_sizes, args.output)


if __name__ == "__main__":
    main()
