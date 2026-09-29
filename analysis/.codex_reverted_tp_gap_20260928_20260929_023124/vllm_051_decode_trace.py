"""Analyze GPU-only Kineto traces delimited by cheap CUDA sleep markers.

Times are per-rank critical-path spans, not sums over ranks. Collective
kernel durations include waiting, and must not be called wire transfer time.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path


def union_us(intervals):
    total = 0.0
    end = None
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if end is None or lo > end:
            total += hi - lo
            end = hi
        elif hi > end:
            total += hi - end
            end = hi
    return total


def category(name):
    n = name.lower()
    if "allreduce" in n or "cross_device_reduce" in n:
        return "allreduce_including_rank_wait"
    if "broadcast" in n:
        return "metadata_broadcast"
    if "nccl" in n:
        return "other_collective"
    if "flash_fwd" in n or "flash_attn" in n:
        return "attention"
    if "rotary" in n:
        return "rope"
    if "reshape_and_cache" in n:
        return "kv_cache_save"
    if any(s in n for s in ("gemm", "gemv", "xmma", "norm_kernel", "act_and_mul")):
        return "compute"
    return "other_gpu"


def stage(events, lo, hi):
    groups = {}
    counts = {}
    kernels = {}
    for e in events:
        start = e["ts"]
        end = start + e.get("dur", 0)
        if end <= lo or start >= hi:
            continue
        interval = (max(lo, start), min(hi, end))
        c = category(e["name"])
        groups.setdefault(c, []).append(interval)
        counts[c] = counts.get(c, 0) + 1
        kernels[e["name"]] = kernels.get(e["name"], 0) + interval[1] - interval[0]
    busy = union_us([i for values in groups.values() for i in values])
    grouped = {k: union_us(v) / 1000 for k, v in groups.items()}
    wall = max(0, hi - lo) / 1000
    return {
        "wall_ms": wall,
        "kernel_union_ms": busy / 1000,
        "no_kernel_gap_ms": max(0, wall - busy / 1000),
        "cross_category_overlap_ms": max(0, sum(grouped.values()) - busy / 1000),
        "groups_ms": grouped,
        "kernel_counts": counts,
        "kernels_us": kernels,
    }


def summarize(values):
    if not values:
        return None
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "n": len(values)}


def is_marker(event):
    name = event.get("name", "").lower()
    return "spin_kernel(" in name or "sleep" in name


def analyze(path, rank, expected_steps, trim=8):
    data = json.loads(Path(path).read_text())
    gpu = [e for e in data["traceEvents"]
           if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    markers = sorted((e for e in gpu if is_marker(e)),
                     key=lambda e: e["ts"])
    width = 4 if rank == 0 else 3
    if len(markers) != expected_steps * width:
        raise RuntimeError(f"rank {rank}: expected {expected_steps * width} markers, "
                           f"got {len(markers)}; cannot safely assign stages")
    gpu = [e for e in gpu if not is_marker(e)]
    steps = []
    for index in range(expected_steps):
        m = markers[index * width:(index + 1) * width]
        row = {"decode_step": index + 1}
        for i, label in enumerate(("forward", "logits", "sampling")[:width - 1]):
            row[label] = stage(gpu, m[i]["ts"] + m[i]["dur"], m[i + 1]["ts"])
        if index + 1 < expected_steps:
            next_start = markers[(index + 1) * width]["ts"]
            row["between_steps"] = stage(gpu, m[-1]["ts"] + m[-1]["dur"], next_start)
            row["complete_cycle_ms"] = (next_start - m[0]["ts"]) / 1000
        row["model_wall_ms"] = row["forward"]["wall_ms"] + row["logits"]["wall_ms"]
        steps.append(row)
    selected = steps[trim:len(steps) - trim if trim else None]
    if not selected:
        raise RuntimeError("profiling window empty after trimming")
    out = {"rank": rank, "trace_file": str(path), "trim_each_end": trim,
           "selected_decode_steps": [selected[0]["decode_step"], selected[-1]["decode_step"]],
           "model_wall_ms": summarize([s["model_wall_ms"] for s in selected])}
    out["complete_cycle_ms"] = summarize([s["complete_cycle_ms"] for s in selected
                                          if "complete_cycle_ms" in s])
    for label in list(("forward", "logits", "sampling")[:width - 1]) + ["between_steps"]:
        items = [s[label] for s in selected if label in s]
        names = sorted({k for item in items for k in item["groups_ms"]})
        out[label] = {key: summarize([s[key] for s in items])
                      for key in ("wall_ms", "kernel_union_ms", "no_kernel_gap_ms",
                                  "cross_category_overlap_ms")}
        out[label]["groups_ms"] = {k: summarize([s["groups_ms"].get(k, 0) for s in items])
                                   for k in names}
        out[label]["counts_per_step"] = {k: statistics.fmean(
            [s["kernel_counts"].get(k, 0) for s in items]) for k in names}
        kernels = {}
        for item in items:
            for k, v in item["kernels_us"].items():
                kernels[k] = kernels.get(k, 0) + v / len(items) / 1000
        out[label]["kernels_ms_per_step"] = dict(sorted(kernels.items(), key=lambda x: -x[1]))
    return out
