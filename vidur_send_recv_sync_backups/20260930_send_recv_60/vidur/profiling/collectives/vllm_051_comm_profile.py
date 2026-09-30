#!/usr/bin/env python3
"""Vidur-compatible collectives profiling through vLLM 0.5.1.

The adjacent Vidur sources remain the authority for CLI defaults, the size
grid, layouts, buffer shapes, repetitions, Kineto aggregation and CSV schema.
Only communication setup/calls and CUDA graph capture are adapted. Importing
selected definitions avoids requiring Sarathi in the vLLM container.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import gc
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional


def source_definitions(path, names, dependencies=None):
    """Execute only the named definitions, retaining their original bodies."""
    path = Path(path)
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            node.decorator_list = []
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if targets & set(names):
                selected.append(node)
                found.update(targets)
    if set(names) - found:
        raise RuntimeError(f"Missing reference definitions in {path}: {set(names) - found}")
    namespace = {"__name__": __name__, **(dependencies or {})}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def reference_paths(root):
    profiling = Path(root) / "vidur/profiling"
    return {
        "main": profiling / "collectives/main.py",
        "runner": profiling / "collectives/benchmark_runner.py",
        "wrapper": profiling / "collectives/collectives_wrapper.py",
        "impl": profiling / "collectives/collectives_impl.py",
        "input": profiling / "collectives/collectives_input.py",
        "utils": profiling / "utils/__init__.py",
        "singleton": profiling / "utils/singleton.py",
        "store": profiling / "common/timer_stats_store.py",
        "timer": profiling / "common/cuda_timer.py",
    }


def reference_parser(root):
    """Read the original add_argument calls instead of copying their defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    tree = ast.parse(reference_paths(root)["main"].read_text())
    parse = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "parse_args")
    for node in ast.walk(parse):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "add_argument":
            evaluate = lambda expr: eval(compile(ast.Expression(expr), "<Vidur CLI>", "eval"),
                                         {"int": int, "str": str})
            parser.add_argument(*(evaluate(arg) for arg in node.args),
                                **{kw.arg: evaluate(kw.value) for kw in node.keywords})
    parser.add_argument("--mode", choices=["prefill", "decode", "both"], default="decode",
                        help="prefill=eager, decode=CUDA graph (default); both profiles both "
                             "but exports decode timings in the single CSV; custom all-reduce is off")
    parser.add_argument("--reference-root", default=str(root),
                        help="Vidur checkout supplying all reference parameters")
    parser.add_argument("--address", default=None, help="Optional existing Ray cluster address")
    parser.add_argument("--plan-only", action="store_true",
                        help="Show the exact parameter contract without starting Ray or GPUs")
    return parser


def load_grid(root):
    paths = reference_paths(root)
    inputs = source_definitions(paths["input"], ["CollectivesInput"], {"randint": random.randint})
    grid = source_definitions(paths["utils"],
                             ["get_collectives_sizes_to_profile", "get_collectives_inputs"],
                             {"product": product, "List": List,
                              "CollectivesInput": inputs.CollectivesInput})
    wrapper = source_definitions(paths["wrapper"],
                                 ["ACTIVE_STEPS", "GRAPH_DISABLED_STEPS", "DISABLE_GRAPH", "WARMUP_STEPS"])
    impl = source_definitions(paths["impl"], ["WARMUP_STEPS", "GRAPH_STEPS"])
    return grid, wrapper, impl


def contract(root, num_profile_rounds=None):
    _, wrapper, impl = load_grid(root)
    return {
        "dtype": "torch.float16", "size_input_unit": "elements", "size_csv_unit": "bytes",
        "active_steps": (wrapper.ACTIVE_STEPS if num_profile_rounds is None
                         else num_profile_rounds),
        "eager_calls_per_round": wrapper.GRAPH_DISABLED_STEPS + 1,
        "eager_explicit_warmup": 0,
        "graph_warmup_calls": impl.WARMUP_STEPS,
        "graph_calls_per_replay": impl.GRAPH_STEPS,
        "graph_replays_per_round": 1,
        "wrapper_warmup_constant_unused": wrapper.WARMUP_STEPS,
        "reference_disable_graph": wrapper.DISABLE_GRAPH,
        "timer": "kineto", "event_filter": "name.startswith('nccl')",
        "round_aggregation": "median NCCL event duration", "statistics_rank": 0,
        "custom_all_reduce": False,
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in reference_paths(root).items()},
    }


def load_runtime(root, mode):
    import numpy as np
    import ray
    import torch
    from torch.profiler import record_function
    from vllm.distributed import parallel_state as ps

    paths = reference_paths(root)
    grid, constants, graph_constants = load_grid(root)
    singleton = source_definitions(paths["singleton"], ["Singleton"])
    profile_method = source_definitions(paths["utils"], ["ProfileMethod"],
                                        {"enum": __import__("enum")})
    store = source_definitions(paths["store"], ["TimerStatsStore"],
                               {"np": np, "Singleton": singleton.Singleton,
                                "ProfileMethod": profile_method.ProfileMethod})
    timer = source_definitions(paths["timer"], ["CudaTimer"],
                               {"torch": torch, "time": time, "record_function": record_function,
                                "TimerStatsStore": store.TimerStatsStore,
                                "ProfileMethod": profile_method.ProfileMethod})
    reference_impl = source_definitions(paths["impl"],
                                        ["GraphedCollective", "WARMUP_STEPS", "GRAPH_STEPS"],
                                        {"torch": torch, "Callable": Callable})

    class VllmCollective(reference_impl.GraphedCollective):
        # All buffer allocation, operation selection and launch logic remain
        # inherited from Vidur. Use its preallocated buffers for every op.
        def _run_all_reduce(self):
            ps.get_tp_group().all_reduce(self._buffer)

        def _run_all_gather(self):
            # vLLM 0.5.1 uses torch collectives on its device_group for gather.
            torch.distributed.all_gather_into_tensor(
                self._gather_tensor, self._buffer, group=ps.get_tp_group().device_group)

        def _run_broadcast(self):
            ps.get_tp_group().broadcast(self._buffer, src=0)

        def _run_send_recv(self):
            group = ps.get_tp_group()
            if group.rank_in_group == 0:
                group.send(self._buffer, dst=1)
            elif group.pynccl_comm is not None and not group.pynccl_comm.disabled:
                # GroupCoordinator.recv allocates a new tensor. Use the same
                # PyNccl receive primitive with Vidur's existing buffer.
                group.pynccl_comm.recv(self._buffer, 0)
            else:
                torch.distributed.recv(self._buffer, src=group.ranks[0], group=group.device_group)

        def _run_reduce_scatter(self):
            torch.distributed.reduce_scatter_tensor(
                self._buffer, self._reduce_buffer, group=ps.get_tp_group().device_group)

        def _run_all_to_all(self):
            torch.distributed.all_to_all_single(
                self._alltoall_output_buffer, self._alltoall_input_buffer,
                group=ps.get_tp_group().device_group)

        def _build_graph(self):
            group = ps.get_tp_group()
            if self._collective_fn.__name__ in ("_run_all_reduce", "_run_send_recv"):
                if group.pynccl_comm is None or not group.pynccl_comm.available:
                    raise RuntimeError("Decode requires an available vLLM PyNccl communicator")
            with group.graph_capture() as context:
                for _ in range(graph_constants.WARMUP_STEPS):
                    self._collective_fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                mempool = torch.cuda.graph_pool_handle()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
                    with torch.cuda.graph(graph, pool=mempool, stream=context.stream):
                        for _ in range(graph_constants.GRAPH_STEPS):
                            self._collective_fn()
                torch.cuda.synchronize()
            return graph

    class AuditedTimer(timer.CudaTimer):
        def handle_trace(self, trace):
            events = [event for event in trace.events()
                      if event.name.startswith(self.filter_str)]
            if not events:
                raise RuntimeError("Kineto captured no NCCL events; refusing to write invalid timings")
            self.event_counts = getattr(self, "event_counts", []) + [len(events)]
            super().handle_trace(trace)

    wrapper = source_definitions(paths["wrapper"],
                                 ["CollectiveWrapper", "WARMUP_STEPS", "ACTIVE_STEPS",
                                  "GRAPH_DISABLED_STEPS", "DISABLE_GRAPH"],
                                 {"np": np, "torch": torch, "GraphedCollective": VllmCollective,
                                  "CudaTimer": AuditedTimer, "TimerStatsStore": store.TimerStatsStore})
    # This is the sole mode change: both branches use Vidur's original loops.
    wrapper.CollectiveWrapper.profile.__globals__["DISABLE_GRAPH"] = mode == "prefill"

    class AuditedWrapper(wrapper.CollectiveWrapper):
        def profile(self):
            result = super().profile()
            for value in result["time_stats"][self._collective].values():
                if not math.isfinite(float(value)):
                    raise RuntimeError(f"Non-finite timing: {result}")
            return {**result, "_audit": {
                "mode": mode, "round_nccl_event_counts": self._cuda_timer.event_counts,
                "statistics_samples": self._num_profile_rounds,
            }}

    runner = source_definitions(paths["runner"], ["BenchmarkRunner"],
                                {"gc": gc, "os": os, "ray": ray, "torch": torch, "Optional": Optional,
                                 "logger": logging.getLogger("vllm051_comm"),
                                 "CollectivesInput": object, "CollectiveWrapper": AuditedWrapper})

    class VllmRunner(runner.BenchmarkRunner):
        def run_collective(self, values):
            inputs = SimpleNamespace(**values)
            changed = (inputs.num_workers != self._last_num_workers or
                       inputs.num_workers_per_node != self._last_num_workers_per_node)
            if changed and torch.distributed.is_initialized():
                ps.destroy_model_parallel()
                ps.destroy_distributed_environment()
            result = super().run_collective(inputs)
            if result is not None:
                group = ps.get_tp_group()
                if group.ca_comm is not None:
                    raise RuntimeError("Custom all-reduce unexpectedly enabled")
                result["_audit"].update(environment_info())
                if group.pynccl_comm is not None and group.pynccl_comm.available:
                    result["_audit"]["pynccl_version"] = group.pynccl_comm.nccl.ncclGetVersion()
                result["_audit"]["implementation"] = (
                    "vllm_pynccl_cuda_graph" if mode == "decode" and
                    inputs.collective in ("all_reduce", "send_recv") else "vllm_torch_nccl_group")
            return result

        def _init_communication(self, comm_id, rank, num_workers, devices_per_node):
            # Retain Vidur's original init method, rank placement and env vars.
            super()._init_communication(comm_id, rank, num_workers, devices_per_node)
            torch.cuda.set_device(0)  # Original runners expose exactly one GPU.
            ps.set_custom_all_reduce(False)
            ps.init_distributed_environment(world_size=num_workers, rank=rank, local_rank=0,
                                             distributed_init_method=f"tcp://{self._head_ip}:{comm_id}")
            ps.initialize_model_parallel(tensor_model_parallel_size=num_workers,
                                         pipeline_model_parallel_size=1)
            if ps.get_tp_group().ca_comm is not None:
                raise RuntimeError("Could not disable custom all-reduce")

    return VllmRunner


def environment_info():
    import numpy as np
    import pandas as pd
    import ray
    import torch
    import vllm

    libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                        if "libnccl.so" in line})
    return {
        "python": sys.version.split()[0], "vllm": vllm.__version__, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "torch_nccl": torch.cuda.nccl.version(),
        "ray": ray.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "nccl_libraries": libraries, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nccl_env": {key: value for key, value in os.environ.items()
                     if key.startswith("NCCL_") or key == "VLLM_NCCL_SO_PATH"},
    }


class BenchmarkActor:
    """Load source definitions inside the worker, avoiding Ray serialization
    of dynamically executed classes or extension-module capsules."""

    def __init__(self, root, mode, gpu_id, gpus_per_node, head_ip,
                 num_profile_rounds=None):
        self._runner = load_runtime(root, mode)(
            gpu_id, gpus_per_node, head_ip, num_profile_rounds=num_profile_rounds)

    def run_collective(self, values):
        return self._runner.run_collective(values)


def write_results(output, collective, results, metadata):
    import pandas as pd

    if not results:
        raise RuntimeError("No rows collected; check available GPUs and layout combinations")
    audits = [row["_audit"] for row in results]
    frame = pd.DataFrame([{key: value for key, value in row.items() if key != "_audit"}
                          for row in results])
    frame = (pd.json_normalize(frame["time_stats"]).add_prefix("time_stats.")
             .join(frame.drop(columns=["time_stats"])))
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{collective}.csv"
    frame.to_csv(str(path) + ".partial")  # Preserve Vidur's index column, too.
    Path(str(path) + ".partial").replace(path)
    metadata["row_audits"] = audits
    metadata["rows"] = len(results)
    (output / f"{collective}.metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {len(results)} Vidur-compatible rows: {path}", flush=True)


def write_single_mode_results(output, collective, mode_results, metadata):
    """Export one calibration table; prefer decode when both modes were profiled.

    Vidur has no phase feature in its communication model. Mixing two timings
    for the same layout and size would train an ambiguous target. Selecting the
    complete decode sequence also preserves repeated points in Vidur's grid.
    """
    csv_mode = "decode" if "decode" in mode_results else "prefill"
    write_results(output, collective, mode_results[csv_mode], {
        **metadata,
        "csv_mode": csv_mode,
        "csv_selection_policy": "decode_preferred",
        "profiled_modes": list(mode_results),
        "mode_row_counts": {mode: len(rows) for mode, rows in mode_results.items()},
        "mode_row_audits": {mode: [row["_audit"] for row in rows]
                            for mode, rows in mode_results.items()},
    })


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--reference-root", default=str(Path(__file__).resolve().parents[3]))
    reference_root = Path(bootstrap.parse_known_args()[0].reference_root)
    parser = reference_parser(reference_root)
    args = parser.parse_args()
    if args.num_profile_rounds is not None and args.num_profile_rounds < 1:
        parser.error("--num_profile_rounds must be a positive integer")
    grid, _, _ = load_grid(reference_root)
    parameters = contract(reference_root, args.num_profile_rounds)
    parameters["cli"] = vars(args)
    sizes = grid.get_collectives_sizes_to_profile(args.max_collective_size)
    if not sizes:
        raise ValueError("No sizes in Vidur's grid under --max_collective_size")
    if args.plan_only:
        print(json.dumps({**parameters, "size_grid_length": len(sizes),
                          "size_grid_first_elements": sizes[0], "size_grid_last_elements": sizes[-1],
                          "size_grid_sha256": hashlib.sha256(json.dumps(sizes).encode()).hexdigest()}, indent=2))
        return

    import ray
    import vllm
    from tqdm import tqdm

    if vllm.__version__ != "0.5.1":
        raise RuntimeError(f"Requires vLLM 0.5.1, found {vllm.__version__}")
    ray.init(address=args.address)
    try:
        total_gpus = int(ray.cluster_resources()["GPU"])
        node_ips = [node["NodeName"] for node in ray.nodes()]
        if total_gpus <= 0 or not node_ips:
            raise RuntimeError("No GPUs available in Ray")
        gpus_per_node = total_gpus // len(node_ips)
        parameters["topology"] = {"total_gpus": total_gpus, "num_nodes": len(node_ips),
                                   "gpus_per_node": gpus_per_node, "node_ips": node_ips}
        try:
            parameters["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"],
                text=True).strip()
            parameters["nvidia_smi_topology"] = subprocess.check_output(
                ["nvidia-smi", "topo", "-m"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
        inputs = grid.get_collectives_inputs(len(node_ips), args.num_workers_per_node_combinations,
                                             args.max_collective_size, args.collective, total_gpus)
        unavailable = sorted({item.num_workers_per_node for item in inputs
                              if item.num_workers_per_node > gpus_per_node})
        # Keep the reference CLI defaults, including 8, on a four-GPU machine.
        # These layouts cannot be executed; Vidur's runner otherwise asserts.
        parameters["unavailable_devices_per_node"] = unavailable
        if unavailable:
            print(f"Skipping physically unavailable layouts {unavailable}; "
                  f"this Ray cluster exposes {gpus_per_node} GPUs/node", flush=True)
        inputs = [item for item in inputs if item.num_workers_per_node <= gpus_per_node]
        run_dir = (Path(args.output_dir) / "collective" /
                   datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        modes = ["prefill", "decode"] if args.mode == "both" else [args.mode]
        mode_results = {}
        for mode in modes:
            runner_class = ray.remote(num_gpus=1)(BenchmarkActor)
            runners = [runner_class.options(resources={f"node:{node_ips[gpu_id // gpus_per_node]}": 0.01})
                       .remote(str(reference_root), mode, gpu_id, gpus_per_node, node_ips[0],
                               num_profile_rounds=args.num_profile_rounds)
                       for gpu_id in range(total_gpus)]
            results = []
            try:
                for inputs_item in tqdm(inputs, desc=f"{mode}/{args.collective}"):
                    futures = [runner.run_collective.remote(vars(inputs_item)) for runner in runners]
                    received = ray.get(futures)
                    if received[0] is not None:
                        results.append(received[0])  # Exactly Vidur's rank-0 policy.
                mode_results[mode] = results
            finally:
                for runner in runners:
                    ray.kill(runner)
        write_single_mode_results(run_dir, args.collective, mode_results,
                                  {**parameters, "mode": args.mode})
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
