#!/usr/bin/env python3
"""Probe actual vLLM 0.5.1 Ray/CUDA-graph decode without editing vLLM files.

Run one TP per process. plain -> event-only -> GPU trace -> plain control
measures instrumentation bias. No per-step CUDA synchronization is added.
Outputs use fixed names; subsequent runs replace this TP's prior artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from functools import wraps
from importlib.metadata import version
from pathlib import Path

import torch
from vllm.executor.ray_utils import RayWorkerWrapper
from vllm.worker.worker import Worker

from vllm_051_decode_trace import analyze, summarize


class Probe:
    def __init__(self, worker):
        self.worker = worker
        self.mode = "plain"
        self.current = None
        self.calls = []
        self.profiler = None
        self.prefix = None

    def region(self, label, fn, *args, **kwargs):
        if self.current is None:
            return fn(*args, **kwargs)
        pair = None
        if self.mode == "events":
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            pair[0].record()
        start = time.perf_counter_ns()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if pair:
                pair[1].record()
                self.current.setdefault("events", {})[label] = pair
            if self.mode == "trace":
                torch.cuda._sleep(1)
            self.current[label + "_host_ms"] = elapsed

    def execute(self, fn, model_input, *args, **kwargs):
        metadata = model_input.attn_metadata
        decode = metadata.num_prefills == 0
        if self.mode == "plain" or not decode:
            return fn(model_input, *args, **kwargs)
        if self.mode == "trace" and self.profiler is None:
            self.profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
            self.profiler.__enter__()
        self.current = {"decode_step": len(self.calls) + 1,
                        "cuda_graph": bool(metadata.decode_metadata.use_cuda_graph),
                        "block_table_shape": list(metadata.decode_metadata.block_tables.shape)}
        if not self.current["cuda_graph"]:
            raise RuntimeError("Expected CUDA-graph decode; refusing to compare an eager path")
        if self.mode == "trace":
            torch.cuda._sleep(1)
        start = time.perf_counter_ns()
        try:
            return fn(model_input, *args, **kwargs)
        finally:
            self.current["runner_host_ms"] = (time.perf_counter_ns() - start) / 1e6
            self.calls.append(self.current)
            self.current = None

    def finish(self):
        # Outside the workload only: reading elapsed GPU events requires completion.
        torch.cuda.synchronize()
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            path = self.prefix + f"_rank{self.worker.rank}.trace.json"
            self.profiler.export_chrome_trace(path)
            self.profiler = None
        for row in self.calls:
            for label, pair in row.pop("events", {}).items():
                row[label + "_gpu_ms"] = pair[0].elapsed_time(pair[1])
        return {"rank": self.worker.rank, "calls": self.calls,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "local_rank": self.worker.local_rank}


def wrap_region(obj, name, label, probe):
    original = getattr(obj, name)

    @wraps(original)
    def wrapped(*args, **kwargs):
        return probe.region(label, original, *args, **kwargs)

    setattr(obj, name, wrapped)


class DiagnosticWorker(Worker):
    def diagnostic_begin(self, mode, prefix):
        if not hasattr(self, "probe"):
            self.probe = Probe(self)
            runner = self.model_runner
            for graphs in runner.graph_runners:
                for graph in graphs.values():
                    wrap_region(graph, "forward", "forward", self.probe)
            wrap_region(runner.model, "compute_logits", "logits", self.probe)
            if self.is_driver_worker:
                wrap_region(runner.model, "sample", "sampling", self.probe)
            original = runner.execute_model
            runner.execute_model = lambda model_input, *a, **k: self.probe.execute(
                original, model_input, *a, **k)
        self.probe.mode = mode
        self.probe.prefix = prefix
        self.probe.calls = []
        return {"rank": self.rank, "device": str(self.device)}

    def diagnostic_finish(self):
        return self.probe.finish()


class DiagnosticRayWrapper(RayWorkerWrapper):
    def __init__(self, *args, **kwargs):
        kwargs["worker_module_name"] = "vllm_051_decode_probe"
        kwargs["worker_class_name"] = "DiagnosticWorker"
        super().__init__(*args, **kwargs)


def run_request(engine, args, request_id):
    from vllm import SamplingParams
    prompt = [1] + [42] * (args.prompt_tokens - 1)
    engine.add_request(str(request_id), {"prompt_token_ids": prompt},
                       SamplingParams(max_tokens=args.decode_tokens, temperature=0.0, ignore_eos=True))
    times = []
    while engine.has_unfinished_requests():
        start = time.perf_counter_ns()
        engine.step()
        times.append((time.perf_counter_ns() - start) / 1e6)
    if len(times) != args.decode_tokens:
        raise RuntimeError(f"Expected {args.decode_tokens} steps, got {len(times)}")
    return {"prefill_step_ms": times[0], "decode_step_ms": times[1:],
            "decode_summary_ms": summarize(times[1:]), "e2e_ms": sum(times)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, choices=[2, 4], required=True)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--warmup-requests", type=int, default=5)
    ap.add_argument("--baseline-requests", type=int, default=3)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--custom-all-reduce", action="store_true")
    ap.add_argument("--output-dir", default="/root/changle/Files/vllm_decode_breakdown")
    args = ap.parse_args()
    if version("vllm").split("+")[0] != "0.5.1":
        raise RuntimeError("This probe targets vLLM 0.5.1")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(outdir / f"tp{args.tp}")
    import ray
    import vllm.executor.ray_gpu_executor as executor_module
    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.llm_engine import LLMEngine
    executor_module.RayWorkerWrapper = DiagnosticRayWrapper
    ray.init(num_cpus=4, num_gpus=args.tp, object_store_memory=512 * 1024 * 1024,
             include_dashboard=False, _temp_dir=f"/tmp/vidur_gap_{outdir.name}_tp{args.tp}")
    engine_args = EngineArgs(model=args.model, dtype="float16", load_format="auto",
                             tensor_parallel_size=args.tp, distributed_executor_backend="ray",
                             max_model_len=4096, max_num_seqs=args.max_num_seqs,
                             max_num_batched_tokens=4096, max_seq_len_to_capture=4096,
                             block_size=16, gpu_memory_utilization=args.gpu_memory_utilization,
                             seed=36, disable_log_stats=True,
                             disable_custom_all_reduce=not args.custom_all_reduce)
    engine = LLMEngine.from_engine_args(engine_args)
    counter = 0
    for _ in range(args.warmup_requests):
        run_request(engine, args, counter)
        counter += 1
    report = {"config": vars(args), "versions": {k: version(k)
              for k in ("vllm", "torch", "ray", "vllm-flash-attn")}, "phases": {}}
    try:
        for mode, count, label in [("plain", args.baseline_requests, "baseline_before"),
                                   ("events", args.baseline_requests, "events"),
                                   ("trace", 1, "trace"),
                                   ("plain", args.baseline_requests, "baseline_after")]:
            requests, workers = [], []
            for _ in range(count):
                engine.model_executor._run_workers("diagnostic_begin", mode, prefix)
                requests.append(run_request(engine, args, counter))
                counter += 1
                workers.append(engine.model_executor._run_workers("diagnostic_finish"))
            report["phases"][label] = {"requests": requests, "workers": workers}
            values = [t for r in requests for t in r["decode_step_ms"]]
            report["phases"][label]["decode_summary_ms"] = summarize(values)
            Path(prefix + ".json").write_text(json.dumps(report, indent=2))
            print(label, json.dumps(summarize(values)), flush=True)
        n = args.decode_tokens - 1
        report["trace_analysis"] = [analyze(prefix + f"_rank{r}.trace.json", r, n)
                                    for r in range(args.tp)]
        baseline = report["phases"]["baseline_before"]["decode_summary_ms"]["mean"]
        for label in ("events", "trace", "baseline_after"):
            value = report["phases"][label]["decode_summary_ms"]["mean"]
            report["phases"][label]["delta_vs_baseline_ms"] = value - baseline
        Path(prefix + ".json").write_text(json.dumps(report, indent=2))
        print("RESULT", prefix + ".json", flush=True)
        print("RANK0_MODEL", json.dumps(report["trace_analysis"][0]["model_wall_ms"]), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
