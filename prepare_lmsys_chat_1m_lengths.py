#!/usr/bin/env python3
"""Build a complete request-length pool from locally downloaded LMSYS Parquet.

Each conversation contributes its first adjacent user/assistant pair. Text is
used only for token counting; the output CSV contains no conversation text.
All eligible conversations are saved; sampling happens in the benchmark client.
"""

import argparse
import csv
import json
import math
import tempfile
from collections import Counter
from pathlib import Path


def first_pair(conversation):
    if not isinstance(conversation, list):
        return None
    for left, right in zip(conversation, conversation[1:]):
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        if left.get("role") != "user" or right.get("role") != "assistant":
            continue
        prompt, answer = left.get("content"), right.get("content")
        if isinstance(prompt, str) and prompt.strip() and isinstance(answer, str) and answer.strip():
            return prompt, answer
    return None


def find_parquet_files(path):
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if path.is_dir():
        files = sorted(path.glob("data/*.parquet")) or sorted(path.glob("*.parquet"))
        if files:
            return files
    raise ValueError(f"No Parquet files found at {path}; specify the dataset directory or a Parquet file")


def summarize(histogram):
    count = sum(histogram.values())
    result = {"count": count, "min": min(histogram), "max": max(histogram)}
    thresholds = {label: math.ceil(count * fraction) for label, fraction in
                  (("p50", 0.5), ("p90", 0.9), ("p99", 0.99))}
    cumulative = 0
    for length in sorted(histogram):
        cumulative += histogram[length]
        for label, threshold in thresholds.items():
            if label not in result and cumulative >= threshold:
                result[label] = length
    for threshold in (512, 1024, 2048, 3072):
        result[f"ge_{threshold}"] = sum(n for length, n in histogram.items() if length >= threshold)
    return result


def prepare(args, tokenizer, parquet):
    files = find_parquet_files(args.dataset_path)
    total = sum(parquet.ParquetFile(path).metadata.num_rows for path in files)
    print(f"Local Parquet files: {len(files)}; conversations: {total:,}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    accepted = examined = invalid = too_long = 0
    prefill_histogram, decode_histogram = Counter(), Counter()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=args.output.parent, prefix=args.output.name + ".",
                                         suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            writer = csv.DictWriter(target, fieldnames=["request_id", "num_prefill_tokens", "num_decode_tokens"])
            writer.writeheader()
            next_report = 10000
            for path in files:
                print(f"Reading {path.name}", flush=True)
                for batch in parquet.ParquetFile(path).iter_batches(batch_size=args.batch_size, columns=["conversation"]):
                    conversations = batch.column(0).to_pylist()
                    pairs = []
                    for conversation in conversations:
                        examined += 1
                        pair = first_pair(conversation)
                        if pair is None:
                            invalid += 1
                        else:
                            pairs.append(pair)
                    if pairs:
                        prompts, answers = zip(*pairs)
                        common = dict(return_length=True, return_attention_mask=False,
                                      return_token_type_ids=False, truncation=False, padding=False)
                        prefill_lengths = tokenizer(list(prompts), add_special_tokens=True, **common)["length"]
                        decode_lengths = tokenizer(list(answers), add_special_tokens=False, **common)["length"]
                        if len(prefill_lengths) != len(pairs) or len(decode_lengths) != len(pairs):
                            raise RuntimeError("Tokenizer returned an unexpected batch size")
                        for prefill, decode in zip(prefill_lengths, decode_lengths):
                            prefill, decode = int(prefill), int(decode)
                            if prefill < 1 or decode < 1:
                                invalid += 1
                                continue
                            if prefill + decode > args.max_model_len:
                                too_long += 1
                                continue
                            writer.writerow({"request_id": accepted, "num_prefill_tokens": prefill,
                                             "num_decode_tokens": decode})
                            accepted += 1
                            prefill_histogram[prefill] += 1
                            decode_histogram[decode] += 1
                    if examined >= next_report or examined == total:
                        target.flush()
                        print(f"Examined={examined:,}/{total:,}; saved={accepted:,}; invalid={invalid:,}; above_context_limit={too_long:,}", flush=True)
                        next_report = (examined // 10000 + 1) * 10000
        if examined != total:
            raise RuntimeError(f"Read {examined} conversations; Parquet metadata reports {total}")
        if accepted == 0:
            raise RuntimeError("No eligible conversations found; output file was not replaced")
        temporary.replace(args.output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"Saved all {accepted:,} eligible requests to {args.output.resolve()}", flush=True)
    print("Prefill lengths: " + json.dumps(summarize(prefill_histogram), sort_keys=True), flush=True)
    print("Decode lengths: " + json.dumps(summarize(decode_histogram), sort_keys=True), flush=True)
    print("No request-count limit or source-model filter was applied.", flush=True)
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True, help="Local dataset directory, data directory, or Parquet file")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Local tokenizer directory used by vLLM")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_model_len < 2 or args.batch_size < 1:
        parser.error("max-model-len must be >= 2 and batch-size must be positive")
    if not args.tokenizer.is_dir():
        parser.error("--tokenizer must be an existing local directory")
    try:
        import pyarrow.parquet as parquet
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, local_files_only=True)
        # Counting does not truncate at the tokenizer's advertised context size.
        tokenizer.model_max_length = 10**30
        prepare(args, tokenizer, parquet)
    except Exception as exc:
        parser.exit(2, f"Could not prepare the local dataset: {exc}\n")


if __name__ == "__main__":
    main()
