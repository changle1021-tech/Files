#!/usr/bin/env python3
"""Profile vLLM 0.5.1 Llama layer operators into Vidur's mlp.csv schema.

The two row-parallel projections are timed with reduce_results=False because
Vidur adds tensor-parallel all-reduce separately. vLLM fuses residual addition
into RMSNorm, so the two norm targets include that addition and the separate
Vidur ``add`` target is zero. No model weights are loaded; dummy weights
exercise the same FP16 kernels and sharding.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from vllm_051_profile_common import (check_vllm_version, close_tp, gpu_times,
                                     init_tp, put_stats, run_all_tp, write_csv)


DEFAULT_OUTPUT = "/root/changle/Files/vidur_vllm051_profiles/mlp.csv"


def vidur_token_grid(max_tokens: int) -> List[int]:
    grid = ([1, 2, 4] + list(range(8, 1024, 8))
            + list(range(1024, 2049, 16))
            + list(range(2048, 4097, 32))
            + list(range(4096, 8193, 64))
            + list(range(8192, 16385, 128))
            + list(range(16384, 32769, 256))
            + list(range(32768, 65537, 512))
            + list(range(65536, 131073, 1024)))
    # Vidur keeps repeated boundary points (1024/2048/4096...). Preserve
    # them so an equivalent run has the same rows and repeated measurements.
    return sorted((value for value in grid if value <= max_tokens), reverse=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/mnt/data02/000000/model/Llama-2-7b-hf")
    p.add_argument("--tensor-parallel-sizes", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--num-tokens", type=int, nargs="+",
                   help="Override Vidur's token grid (useful for a smoke test)")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repetitions", type=int, default=20)
    p.add_argument("--vidur-vocab-size", type=int, default=32768,
                   help="Vidur Llama-2-7B filter key; not a measured operator")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--part-output", help=argparse.SUPPRESS)
    return p


def worker(args: argparse.Namespace, tokens: List[int]) -> None:
    from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding, get_masked_input_and_mask)
    from vllm.model_executor.models.llama import LlamaDecoderLayer

    rank, _, tp = init_tp()
    try:
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
        if config.model_type != "llama" or config.hidden_act != "silu":
            raise ValueError("This profiler currently supports FP16 Llama/SILU models")
        torch.set_default_dtype(torch.float16)
        layer = LlamaDecoderLayer(config).cuda().eval()
        initialize_dummy_weights(layer)
        embedding = VocabParallelEmbedding(args.vidur_vocab_size,
                                           config.hidden_size).cuda().eval()
        initialize_dummy_weights(embedding)
        # Vidur models the two all-reduces independently.
        layer.self_attn.o_proj.reduce_results = False
        layer.mlp.down_proj.reduce_results = False
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        rows = []
        for count in tokens:
            input_ids = torch.randint(args.vidur_vocab_size, (count,),
                                      device="cuda", dtype=torch.long)
            positions = torch.arange(count, device="cuda", dtype=torch.long)
            x = torch.randn((count, hidden), device="cuda", dtype=torch.float16)
            q = torch.randn((count, layer.self_attn.q_size), device="cuda",
                            dtype=torch.float16)
            k = torch.randn((count, layer.self_attn.kv_size), device="cuda",
                            dtype=torch.float16)
            attn_partial = torch.randn((count, hidden // tp), device="cuda",
                                       dtype=torch.float16)
            mlp_partial = torch.randn((count, intermediate // tp), device="cuda",
                                      dtype=torch.float16)
            gate_up = torch.randn((count, 2 * intermediate // tp), device="cuda",
                                  dtype=torch.float16)

            def embed_local():
                # vLLM's embedding forward also all-reduces. Vidur profiles
                # that communication independently, as Sarathi did here.
                if tp > 1:
                    indices = embedding.shard_indices
                    masked_ids, mask = get_masked_input_and_mask(
                        input_ids, indices.org_vocab_start_index,
                        indices.org_vocab_end_index,
                        indices.num_org_vocab_padding,
                        indices.added_vocab_start_index,
                        indices.added_vocab_end_index)
                else:
                    masked_ids = input_ids
                output = F.embedding(masked_ids.long(), embedding.weight)
                if tp > 1:
                    output.masked_fill_(mask.unsqueeze(-1), 0)
                return output

            operations = {
                "emb": embed_local,
                "attn_pre_proj": lambda: layer.self_attn.qkv_proj(x),
                "attn_rope": lambda: layer.self_attn.rotary_emb(positions, q, k),
                "attn_post_proj": lambda: layer.self_attn.o_proj(attn_partial),
                "mlp_up_proj": lambda: layer.mlp.gate_up_proj(x),
                "mlp_down_proj": lambda: layer.mlp.down_proj(mlp_partial),
                "mlp_act": lambda: layer.mlp.act_fn(gate_up),
                # Except for the first layer, vLLM's LlamaDecoderLayer passes
                # the residual to both norms and uses the fused add+RMSNorm op.
                "input_layernorm": lambda: layer.input_layernorm(x, x),
                "post_attention_layernorm": lambda: layer.post_attention_layernorm(x, x),
            }
            row = {
                "n_head": config.num_attention_heads,
                "n_kv_head": getattr(config, "num_key_value_heads", config.num_attention_heads),
                "n_embd": hidden,
                "n_expanded_embd": intermediate,
                "vocab_size": args.vidur_vocab_size,
                "use_gated_mlp": True,
                "num_tokens": count,
                "num_tensor_parallel_workers": tp,
            }
            for name, operation in operations.items():
                put_stats(row, name, gpu_times(operation, args.warmup,
                                               args.repetitions))
            # The residual adds are already included in the fused norm kernels.
            # Vidur still requires this column as an independent model target.
            put_stats(row, "add", [0.0] * args.repetitions)
            rows.append(row)
            if rank == 0:
                print(f"TP={tp} num_tokens={count}", flush=True)
        if rank == 0:
            write_csv(args.part_output, rows)
    finally:
        close_tp()


def main() -> None:
    args = parser().parse_args()
    check_vllm_version()
    if args.max_tokens < 1 or args.warmup < 0 or args.repetitions < 2:
        raise ValueError("max-tokens >= 1, warmup >= 0 and repetitions >= 2 are required")
    tokens = list(args.num_tokens) if args.num_tokens else vidur_token_grid(args.max_tokens)
    if any(value < 1 or value > args.max_tokens for value in tokens):
        raise ValueError("num-tokens values must be within [1, max-tokens]")
    if args.worker:
        worker(args, tokens)
        return
    forwarded = ["--model", args.model, "--max-tokens", str(args.max_tokens),
                 "--warmup", str(args.warmup), "--repetitions", str(args.repetitions),
                 "--vidur-vocab-size", str(args.vidur_vocab_size)]
    if args.num_tokens:
        forwarded += ["--num-tokens", *map(str, args.num_tokens)]
    run_all_tp(str(Path(__file__).resolve()), forwarded,
               args.tensor_parallel_sizes, args.output)


if __name__ == "__main__":
    main()
