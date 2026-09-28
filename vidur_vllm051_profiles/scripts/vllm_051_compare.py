#!/usr/bin/env python3
"""Compare matching Vidur/Sarathi and vLLM 0.5.1 profiling CSV rows."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path


DEFAULT_DIR = Path("/root/changle/Files/vidur_vllm051_profiles")
DEFAULT_OLD_DIR = Path(
    "/root/changle/SimAI/vidur-alibabacloud/data/profiling/compute/"
    "h100/meta-llama/Llama-2-7b-hf"
)
FIELDS = ("kind", "tp", "case", "repeat_index", "metric", "status", "old_ms", "new_ms",
          "delta_ms", "delta_pct")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def key(row: dict[str, str], kind: str) -> tuple[str, ...]:
    shared = (row["num_tensor_parallel_workers"],)
    if kind == "mlp":
        return shared + (row["num_tokens"],)
    return shared + tuple(row[name].lower() for name in
                          ("batch_size", "prefill_chunk_size",
                           "kv_cache_size", "is_prefill"))


def compare(kind: str, old_path: Path, new_path: Path) -> list[dict[str, object]]:
    old_rows = defaultdict(deque)
    for row in read_rows(old_path):
        old_rows[key(row, kind)].append(row)
    repeat_index = defaultdict(int)
    result = []
    for new in read_rows(new_path):
        row_key = key(new, kind)
        repeat_index[row_key] += 1
        if not old_rows[row_key]:
            print(f"No original {kind} row for {row_key}")
            continue
        old = old_rows[row_key].popleft()
        for column in sorted(set(old) | set(new)):
            if not (column.startswith("time_stats.") and column.endswith(".median")):
                continue
            old_text, new_text = old.get(column, ""), new.get(column, "")
            if not old_text and not new_text:
                continue
            old_ms = float(old_text) if old_text else None
            new_ms = float(new_text) if new_text else None
            matched = old_ms is not None and new_ms is not None
            delta = new_ms - old_ms if matched else None
            result.append({
                "kind": kind,
                "tp": row_key[0],
                "case": "/".join(row_key[1:]),
                "repeat_index": repeat_index[row_key],
                "metric": column.removeprefix("time_stats.").removesuffix(".median"),
                "status": ("matched" if matched else
                           "new_only" if new_ms is not None else "old_only"),
                "old_ms": f"{old_ms:.6f}" if old_ms is not None else "",
                "new_ms": f"{new_ms:.6f}" if new_ms is not None else "",
                "delta_ms": f"{delta:.6f}" if delta is not None else "",
                "delta_pct": f"{delta / old_ms * 100:.2f}"
                if matched and old_ms else "",
            })
    remaining = sum(len(rows) for rows in old_rows.values())
    if remaining:
        print(f"{remaining} original {kind} rows have no new counterpart")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-mlp", type=Path, default=DEFAULT_OLD_DIR / "mlp.csv")
    parser.add_argument("--old-attention", type=Path,
                        default=DEFAULT_OLD_DIR / "attention.csv")
    parser.add_argument("--new-mlp", type=Path, default=DEFAULT_DIR / "mlp.csv")
    parser.add_argument("--new-attention", type=Path,
                        default=DEFAULT_DIR / "attention.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "comparison.csv")
    args = parser.parse_args()
    rows = compare("mlp", args.old_mlp, args.new_mlp)
    rows.extend(compare("attention", args.old_attention, args.new_attention))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print(f"Compared {len(rows)} metrics: {args.output}")


if __name__ == "__main__":
    main()
