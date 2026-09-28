#!/usr/bin/env python3
"""Shared helpers for standalone vLLM 0.5.1 operator profilers."""

from __future__ import annotations

import csv
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import torch


STATS = ("min", "max", "mean", "median", "std")


def check_vllm_version() -> None:
    import vllm

    if vllm.__version__ != "0.5.1":
        raise RuntimeError(f"These profilers require vLLM 0.5.1, found {vllm.__version__}")


def init_tp() -> tuple[int, int, int]:
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment(world_size=world_size,
                                 rank=rank,
                                 local_rank=local_rank)
    initialize_model_parallel(tensor_model_parallel_size=world_size,
                              pipeline_model_parallel_size=1)
    return rank, local_rank, world_size


def close_tp() -> None:
    from vllm.distributed import destroy_model_parallel

    destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def gpu_times(fn: Callable[[], object], warmup: int, repetitions: int) -> List[float]:
    """Sum CUDA kernel durations per call, as Vidur's Kineto tracer does."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            for _ in range(repetitions):
                with torch.profiler.record_function("vidur_vllm051_operation"):
                    fn()
        torch.cuda.synchronize()
    # Kineto exposes both a CPU annotation and its CUDA mirror. FlashAttention
    # decode's C++ launcher is not correlated to the CPU annotation, while the
    # CUDA mirror still carries the kernel duration for the marked call.
    events = [event for event in profiler.events()
              if event.name == "vidur_vllm051_operation"
              and event.device_type == torch.autograd.DeviceType.CUDA]
    timings = [event.cuda_time_total * 1e-3 for event in events]
    if len(timings) != repetitions:
        raise RuntimeError(f"Expected {repetitions} CUDA measurements, got "
                           f"{len(timings)}")
    if any(not math.isfinite(value) or value <= 0 for value in timings):
        raise RuntimeError(f"Kineto did not capture CUDA kernels for every "
                           f"call: {timings}")
    values = torch.tensor(timings,
                          device="cuda", dtype=torch.float64)
    if torch.distributed.get_world_size() > 1:
        # A TP step finishes when its slowest rank has finished.
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
    return values.cpu().tolist()


def put_stats(row: dict, name: str, values: Sequence[float]) -> None:
    if not values or any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"Invalid timings for {name}: {values}")
    prefix = f"time_stats.{name}."
    row[prefix + "min"] = min(values)
    row[prefix + "max"] = max(values)
    row[prefix + "mean"] = statistics.fmean(values)
    row[prefix + "median"] = statistics.median(values)
    row[prefix + "std"] = statistics.pstdev(values)


def write_csv(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        raise RuntimeError("No profiling rows were produced")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary = target.with_name(target.name + ".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)


def run_all_tp(script: str, args: Sequence[str], sizes: Iterable[int],
               output: str) -> None:
    sizes = list(dict.fromkeys(sizes))
    if not sizes or any(tp < 1 for tp in sizes):
        raise ValueError("Tensor-parallel sizes must be positive")
    available = torch.cuda.device_count()
    if max(sizes) > available:
        raise ValueError(f"Requested TP={max(sizes)} but only {available} GPUs are visible")
    rows: List[dict] = []
    with tempfile.TemporaryDirectory(prefix="vllm051_operator_profile_") as temp:
        for tp in sizes:
            part = str(Path(temp) / f"tp{tp}.csv")
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       "--nnodes=1", f"--nproc_per_node={tp}", script, "--worker",
                       "--part-output", part, *args]
            print(f"Profiling TP={tp}", flush=True)
            subprocess.run(command, check=True)
            with open(part, newline="") as stream:
                rows.extend(csv.DictReader(stream))
    write_csv(output, rows)
    print(f"Wrote {len(rows)} rows to {output}", flush=True)
