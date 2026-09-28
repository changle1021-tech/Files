#!/usr/bin/env python3
"""Collect Vidur-compatible CPU-overhead profiles from vLLM 0.5.1.

This profiler intentionally targets the internal APIs of vLLM 0.5.1.  It
records non-overlapping timing regions on the engine driver and the local
driver worker, then writes the column names consumed by Vidur's existing CPU
overhead predictor.

Run one process per model/tensor-parallel configuration.  A single engine is
reused for all requested batch sizes so max_num_seqs remains a property of the
target serving configuration instead of changing with every measurement.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import platform
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Optional

import numpy as np
import torch
import vllm
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine


LOGGER = logging.getLogger("vllm_051_cpu_overhead")
EXPECTED_VLLM_VERSION = "0.5.1"
MS_PER_NS = 1e-6


class TimingStore:
    """In-memory per-iteration timings, all expressed in milliseconds."""

    def __init__(self) -> None:
        self.values: DefaultDict[str, List[float]] = defaultdict(list)

    def append_ns(self, name: str, elapsed_ns: int) -> None:
        self.values[name].append(elapsed_ns * MS_PER_NS)

    def append_ms(self, name: str, elapsed_ms: float) -> None:
        self.values[name].append(elapsed_ms)

    def mark(self, name: str) -> int:
        return len(self.values[name])

    def sum_since(self, name: str, index: int) -> float:
        return float(sum(self.values[name][index:]))

    def reset(self) -> None:
        self.values.clear()


def _wrap_timed_method(
    obj: Any,
    method_name: str,
    store: TimingStore,
    metric_name: str,
    after_call: Optional[Callable[[Any], None]] = None,
    synchronize_after: bool = False,
) -> None:
    """Replace one bound method on an instance with a timed wrapper."""

    original = getattr(obj, method_name)

    def timed(*args: Any, **kwargs: Any) -> Any:
        start_ns = time.perf_counter_ns()
        try:
            result = original(*args, **kwargs)
        finally:
            # Sarathi's CpuTimer synchronizes at the end of every measured
            # region. Matching that boundary is essential: without it, CUDA
            # model work is asynchronous and its wait is charged to sampler.
            if synchronize_after:
                torch.cuda.synchronize()
            store.append_ns(metric_name, time.perf_counter_ns() - start_ns)
        if after_call is not None:
            after_call(result)
        return result

    setattr(obj, method_name, timed)


def _unwrap_driver_worker(model_executor: Any) -> Any:
    """Return the actual local vLLM Worker for GPU, MP, and Ray executors."""

    worker = getattr(model_executor, "driver_worker", None)
    if worker is None:
        raise RuntimeError(
            f"Unsupported executor without driver_worker: {type(model_executor)!r}"
        )

    # RayGPUExecutor stores a local RayWorkerWrapper, whose .worker is the
    # actual Worker. GPUExecutor and MultiprocessingGPUExecutor store Worker
    # directly.
    nested_worker = getattr(worker, "worker", None)
    if nested_worker is not None:
        worker = nested_worker

    if not hasattr(worker, "model_runner"):
        raise RuntimeError(
            "Could not locate model_runner on vLLM's local driver worker; "
            "this script only supports the vLLM 0.5.1 GPU/MP/Ray executors."
        )
    return worker


def install_timing_hooks(engine: LLMEngine, store: TimingStore) -> None:
    """Install vLLM 0.5.1 timing hooks after engine initialization."""

    if len(engine.scheduler) != 1:
        raise RuntimeError("Only pipeline_parallel_size=1 is supported")

    def record_scheduled_batch(result: Any) -> None:
        _, scheduler_outputs = result
        store.append_ms(
            "observed_batch_size",
            float(len(scheduler_outputs.scheduled_seq_groups)),
        )

    _wrap_timed_method(
        engine.scheduler[0],
        "schedule",
        store,
        "schedule",
        after_call=record_scheduled_batch,
        synchronize_after=True,
    )
    _wrap_timed_method(
        engine,
        "_process_model_outputs",
        store,
        "process_model_outputs",
        synchronize_after=True,
    )

    worker = _unwrap_driver_worker(engine.model_executor)
    model_runner = worker.model_runner

    _wrap_timed_method(
        model_runner,
        "prepare_model_input",
        store,
        "prepare_inputs_e2e",
        synchronize_after=True,
    )

    original_sample = model_runner.model.sample

    def timed_sample(*args: Any, **kwargs: Any) -> Any:
        # vLLM invokes sample inside model_runner.execute_model. Synchronizing
        # before starting this nested timer assigns preceding async model work
        # to MODEL_EXECUTION_E2E, exactly as Sarathi's separate timer regions
        # do. The final sync assigns sampling kernels to SAMPLER_E2E.
        torch.cuda.synchronize()
        start_ns = time.perf_counter_ns()
        try:
            return original_sample(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            store.append_ns("sampler_e2e", time.perf_counter_ns() - start_ns)

    model_runner.model.sample = timed_sample

    original_model_execute = model_runner.execute_model

    def timed_model_execute(*args: Any, **kwargs: Any) -> Any:
        sampler_mark = store.mark("sampler_e2e")
        start_ns = time.perf_counter_ns()
        try:
            return original_model_execute(*args, **kwargs)
        finally:
            total_ms = (time.perf_counter_ns() - start_ns) * MS_PER_NS
            sampler_ms = store.sum_since("sampler_e2e", sampler_mark)
            store.append_ms("model_runner_e2e", total_ms)
            # This is used only to remove model execution from the driver-side
            # wall clock. Vidur does not train a model from this column.
            store.append_ms("model_execution_e2e", total_ms - sampler_ms)

    model_runner.execute_model = timed_model_execute

    original_executor_execute = engine.model_executor.execute_model

    def timed_executor_execute(*args: Any, **kwargs: Any) -> Any:
        prepare_mark = store.mark("prepare_inputs_e2e")
        runner_mark = store.mark("model_runner_e2e")
        start_ns = time.perf_counter_ns()
        try:
            return original_executor_execute(*args, **kwargs)
        finally:
            executor_ms = (time.perf_counter_ns() - start_ns) * MS_PER_NS
            local_worker_ms = (
                store.sum_since("prepare_inputs_e2e", prepare_mark)
                + store.sum_since("model_runner_e2e", runner_mark)
            )
            store.append_ms("executor_e2e", executor_ms)
            # Keep Vidur's historical column name. For the MP backend this is
            # executor/control-plane residual rather than literal Ray RPC time.
            store.append_ms("ray_comm_time", executor_ms - local_worker_ms)

    engine.model_executor.execute_model = timed_executor_execute

    original_step = engine.step

    def timed_step(*args: Any, **kwargs: Any) -> Any:
        start_ns = time.perf_counter_ns()
        try:
            return original_step(*args, **kwargs)
        finally:
            store.append_ns("step_e2e", time.perf_counter_ns() - start_ns)

    engine.step = timed_step


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise RuntimeError("Cannot calculate a mean from an empty timing series")
    return float(statistics.fmean(values))


def _median(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise RuntimeError("Cannot calculate a median from an empty timing series")
    return float(statistics.median(values))


class VllmCpuOverheadProfiler:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.store = TimingStore()
        self.request_id = 0
        self.rng = np.random.default_rng(args.seed)

        backend = None if args.executor_backend == "auto" else args.executor_backend
        engine_args = EngineArgs(
            model=args.model,
            tokenizer=args.tokenizer or args.model,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            tokenizer_revision=args.tokenizer_revision,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=1,
            distributed_executor_backend=backend,
            dtype=args.dtype,
            load_format=args.load_format,
            seed=args.seed,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            disable_log_stats=True,
        )
        LOGGER.info("Creating vLLM engine with %s", engine_args)
        self.engine = LLMEngine.from_engine_args(engine_args)
        install_timing_hooks(self.engine, self.store)

        hf_config = self.engine.model_config.hf_config
        self.vocab_size = int(getattr(hf_config, "vocab_size", 10000))
        if self.vocab_size <= 0:
            raise RuntimeError(f"Invalid vocabulary size: {self.vocab_size}")

    def _add_requests(self, batch_size: int, decode_tokens: int) -> None:
        sampling_params = SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=decode_tokens,
        )
        token_high = min(self.vocab_size, 10000)
        for _ in range(batch_size):
            prompt_token_ids = self.rng.integers(
                low=0,
                high=token_high,
                size=self.args.prompt_tokens,
            ).tolist()
            self.engine.add_request(
                request_id=str(self.request_id),
                inputs={"prompt_token_ids": prompt_token_ids},
                params=sampling_params,
            )
            self.request_id += 1

    def _drain_engine(self) -> int:
        steps = 0
        while self.engine.has_unfinished_requests():
            self.engine.step()
            steps += 1
        return steps

    def profile_batch(self, batch_size: int) -> Dict[str, Any]:
        LOGGER.info("Warming batch_size=%d", batch_size)
        self._add_requests(batch_size, self.args.warmup_decode_tokens)
        self._drain_engine()
        self.store.reset()

        total_steps = 0
        start_ns = time.perf_counter_ns()
        for repeat in range(self.args.repetitions):
            LOGGER.info(
                "Profiling batch_size=%d repeat=%d/%d",
                batch_size,
                repeat + 1,
                self.args.repetitions,
            )
            self._add_requests(batch_size, self.args.decode_tokens)
            total_steps += self._drain_engine()
        benchmark_wall_ms = (time.perf_counter_ns() - start_ns) * MS_PER_NS

        required = (
            "schedule",
            "sampler_e2e",
            "prepare_inputs_e2e",
            "model_execution_e2e",
            "process_model_outputs",
            "ray_comm_time",
        )
        for metric in required:
            if not self.store.values[metric]:
                raise RuntimeError(f"No samples recorded for {metric}")

        residuals = self.store.values["ray_comm_time"]
        negative_residuals = sum(value < -0.01 for value in residuals)
        if negative_residuals:
            raise RuntimeError(
                f"Found {negative_residuals} executor residuals below -0.01 ms; "
                "timing regions overlap or the vLLM internals do not match 0.5.1"
            )

        observed_batches = self.store.values["observed_batch_size"]
        nonzero_observed = [int(value) for value in observed_batches if value > 0]

        row: Dict[str, Any] = {
            "model_name": self.args.model_name_for_vidur or self.args.model,
            "batch_size": batch_size,
            "tensor_parallel_degree": self.args.tensor_parallel_size,
            "vllm_version": vllm.__version__,
            "executor_backend": type(self.engine.model_executor).__name__,
            "requested_executor_backend": self.args.executor_backend,
            "enforce_eager": self.args.enforce_eager,
            "prompt_tokens": self.args.prompt_tokens,
            "decode_tokens": self.args.decode_tokens,
            "repetitions": self.args.repetitions,
            "num_steps": total_steps,
            "benchmark_wall_ms": benchmark_wall_ms,
            "observed_batch_size_min": min(nonzero_observed),
            "observed_batch_size_max": max(nonzero_observed),
        }
        for metric in required:
            values = self.store.values[metric]
            row[f"{metric}_mean"] = _mean(values)
            row[f"{metric}_median"] = _median(values)

        # Vidur consumes the mean transport residual under this legacy name.
        row["ray_comm_time_mean"] = _mean(residuals)
        row["step_e2e_mean"] = _mean(self.store.values["step_e2e"])
        row["step_e2e_median"] = _median(self.store.values["step_e2e"])

        LOGGER.info(
            "batch=%d steps=%d schedule=%.4fms prepare=%.4fms "
            "sample=%.4fms process=%.4fms transport=%.4fms",
            batch_size,
            total_steps,
            row["schedule_median"],
            row["prepare_inputs_e2e_median"],
            row["sampler_e2e_median"],
            row["process_model_outputs_median"],
            row["ray_comm_time_mean"],
        )
        self.store.reset()
        return row

    def close(self) -> None:
        executor = getattr(self.engine, "model_executor", None)
        if executor is not None:
            executor.shutdown()
            # Prevent LLMEngine.__del__ from shutting down the executor twice.
            self.engine.model_executor = None
        del self.engine
        gc.collect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile vLLM 0.5.1 CPU overhead for Vidur"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-name-for-vidur",
        default=None,
        help="Exact model_name stored in CSV; defaults to --model",
    )
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--executor-backend",
        choices=("auto", "mp", "ray"),
        default="ray",
        help=(
            "Distributed executor used by vLLM (default: ray, so single-node "
            "and future multi-node profiles use the same RPC path)"
        ),
    )
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--load-format", default="dummy")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmup-decode-tokens", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(range(1, 9)),
    )
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument(
        "--output",
        default="/vllm-workspace/vidur_cpu_overhead/cpu_overheads.csv",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows, for example when collecting another TP degree",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if vllm.__version__ != EXPECTED_VLLM_VERSION:
        parser.error(
            f"this profiler requires vLLM {EXPECTED_VLLM_VERSION}, "
            f"found {vllm.__version__}"
        )
    if not args.batch_sizes or min(args.batch_sizes) < 1:
        parser.error("--batch-sizes must contain positive integers")
    args.batch_sizes = sorted(set(args.batch_sizes))
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.decode_tokens < 1 or args.warmup_decode_tokens < 1:
        parser.error("decode token counts must be positive")
    if args.prompt_tokens + args.decode_tokens > args.max_model_len:
        parser.error("prompt_tokens + decode_tokens exceeds max_model_len")

    if args.max_num_seqs is None:
        args.max_num_seqs = max(args.batch_sizes)
    if args.max_num_seqs < max(args.batch_sizes):
        parser.error("max_num_seqs must cover the largest requested batch")
    if args.max_num_batched_tokens is None:
        args.max_num_batched_tokens = max(
            args.max_model_len,
            max(args.batch_sizes) * args.prompt_tokens,
        )
    return args


def _write_rows(
    output_path: Path,
    rows: List[Dict[str, Any]],
    append: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not output_path.exists() or output_path.stat().st_size == 0
    mode = "a" if append else "w"

    if append and output_path.exists() and output_path.stat().st_size:
        with output_path.open(newline="") as existing_file:
            existing_header = next(csv.reader(existing_file))
        if existing_header != list(rows[0].keys()):
            raise RuntimeError(
                f"Cannot append: CSV schema differs in {output_path}"
            )

    with output_path.open(mode, newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _write_metadata(output_path: Path, args: argparse.Namespace) -> None:
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "arguments": vars(args),
        "timing_clock": "time.perf_counter_ns",
        "units": "milliseconds",
        "cuda_timing_boundary": (
            "torch.cuda.synchronize at the end of each Sarathi-compatible "
            "region, plus before vLLM's nested sampler"
        ),
        "ray_comm_time_semantics": (
            "executor wall time minus local driver prepare_model_input and "
            "model_runner.execute_model wall time"
        ),
    }
    metadata_path = output_path.with_name(
        f"{output_path.stem}.tp{args.tensor_parallel_size}.metadata.json"
    )
    with metadata_path.open("w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    profiler = VllmCpuOverheadProfiler(args)
    try:
        rows = [profiler.profile_batch(size) for size in args.batch_sizes]
    finally:
        profiler.close()

    output_path = Path(args.output)
    _write_rows(output_path, rows, args.append)
    _write_metadata(output_path, args)
    LOGGER.info("Wrote %d rows to %s", len(rows), output_path)


if __name__ == "__main__":
    main()
