#!/usr/bin/env python3

import argparse
import asyncio
import csv
import json
import math
import random
import time
from pathlib import Path

import aiohttp
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM Poisson benchmark client with optional request lengths"
    )

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--endpoint", type=str, default="/v1/completions")
    parser.add_argument("--model", type=str, default="Llama-2-7b-hf")

    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--qps", type=float, default=2.8)
    parser.add_argument("--prefill-tokens", type=int, default=3072)
    parser.add_argument("--decode-tokens", type=int, default=50)
    parser.add_argument(
        "--request-lengths-file",
        type=Path,
        help=(
            "CSV with request_id,num_prefill_tokens,num_decode_tokens; "
            "uniformly sample --num-requests rows without replacement from the complete CSV"
        ),
    )
    parser.add_argument("--seed", type=int, default=36)

    parser.add_argument("--bos-token-id", type=int, default=1)
    parser.add_argument("--fill-token-id", type=int, default=42)

    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--timeout", type=float, default=3600.0)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/changle/Files",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="vllm_benchmark_result.json",
        help="Fixed-name JSON result; overwritten on every run",
    )

    parser.add_argument(
        "--trace-output-file",
        type=str,
        default="/root/changle/Files/vllm_request_arrival_trace.csv",
        help=(
            "Vidur trace_replay CSV containing measured client submission "
            "times; overwritten on every run"
        ),
    )

    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="Number of warmup requests to send before benchmark",
    )

    args = parser.parse_args()

    if args.num_requests < 1:
        parser.error("--num-requests must be positive")
    if args.qps <= 0:
        parser.error("--qps must be positive")
    if args.prefill_tokens < 1 or args.decode_tokens < 1:
        parser.error("token counts must be positive")

    return args


def load_request_lengths(args):
    args.length_pool_size = None
    args.sampled_dataset_request_ids = None
    if args.request_lengths_file is None:
        return [
            (args.prefill_tokens, args.decode_tokens)
            for _ in range(args.num_requests)
        ]

    # A separate RNG keeps arrival times unchanged when the pool size changes.
    rng = random.Random(args.seed)
    reservoir = []
    count = 0
    with args.request_lengths_file.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"num_prefill_tokens", "num_decode_tokens"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Length CSV needs num_prefill_tokens,num_decode_tokens columns"
            )
        for index, row in enumerate(reader):
            if row.get("request_id") not in (None, "", str(index)):
                raise ValueError(f"CSV row {index} has a mismatched request_id")
            try:
                prefill = int(row["num_prefill_tokens"])
                decode = int(row["num_decode_tokens"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"CSV row {index} has invalid token counts") from exc
            if prefill < 1 or decode < 1:
                raise ValueError(f"CSV row {index} has nonpositive token counts")
            entry = (prefill, decode, index)
            count += 1
            if len(reservoir) < args.num_requests:
                reservoir.append(entry)
            else:
                replacement = rng.randrange(count)
                if replacement < args.num_requests:
                    reservoir[replacement] = entry
    if count < args.num_requests:
        raise ValueError(
            f"Length CSV has {count} rows; --num-requests={args.num_requests}"
        )
    rng.shuffle(reservoir)
    args.length_pool_size = count
    args.sampled_dataset_request_ids = [entry[2] for entry in reservoir]
    return [(entry[0], entry[1]) for entry in reservoir]


def make_prompt(args, prefill_tokens=None):
    if prefill_tokens is None:
        prefill_tokens = args.prefill_tokens
    if prefill_tokens == 1:
        return [args.bos_token_id]

    return (
        [args.bos_token_id]
        + [args.fill_token_id] * (prefill_tokens - 1)
    )


def generate_vidur_arrival_intervals(num_requests, qps, seed):
    if qps <= 0:
        raise ValueError("qps must be greater than 0")

    rng = random.Random(seed)

    max_interval = (1.0 / qps) * 3.0

    def next_interval():
        interval = -math.log(1.0 - rng.random()) / qps
        return min(interval, max_interval)

    # Vidur consumes this sample before its first request. The HTTP benchmark
    # normalizes request 0 to time zero, so retain the value as metadata and
    # use the remaining random sequence for the inter-request intervals.
    vidur_initial_delay = next_interval() if num_requests > 0 else 0.0
    intervals = []

    for _ in range(max(num_requests - 1, 0)):
        intervals.append(next_interval())

    return vidur_initial_delay, intervals


def intervals_to_arrival_times(intervals):
    arrival_times = [0.0]

    for interval in intervals:
        arrival_times.append(arrival_times[-1] + interval)

    return arrival_times


async def send_request(session, args, prompt, request_id, decode_tokens=None):
    url = f"http://{args.host}:{args.port}{args.endpoint}"
    if decode_tokens is None:
        decode_tokens = args.decode_tokens

    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": decode_tokens,
        "ignore_eos": args.ignore_eos,
        "temperature": args.temperature,
        "stream": True,
    }

    start_time = time.perf_counter()
    first_token_time = None
    finish_time = None
    finish_reason = None
    received_done = False

    try:
        async with session.post(url, json=payload) as response:

            if response.status != 200:
                text = await response.text()

                print(
                    f"[Request {request_id}] "
                    f"HTTP {response.status}: {text}"
                )

                return None

            while True:
                line = await response.content.readline()

                if not line:
                    break

                line = line.decode(
                    "utf-8",
                    errors="replace"
                ).strip()

                if not line.startswith("data: "):
                    continue

                data = line[6:]

                if data == "[DONE]":
                    finish_time = time.perf_counter()
                    received_done = True
                    break

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = obj.get("choices", [])

                if not choices:
                    continue

                choice = choices[0]
                text = choice.get("text", "")
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]

                if text and first_token_time is None:
                    first_token_time = time.perf_counter()

    except Exception as e:
        print(
            f"[Request {request_id}] ERROR: {repr(e)}"
        )

        return None

    if finish_time is None:
        finish_time = time.perf_counter()

    if not received_done:
        print(f"[Request {request_id}] stream ended without [DONE]")
        return None

    e2e = finish_time - start_time

    if first_token_time is None:
        ttft = None
        tpot = None

    else:
        ttft = first_token_time - start_time

        # vLLM reports finish_reason='length' when max_tokens is reached.
        # Without that confirmation, requested and actual token counts may differ.
        if finish_reason == "length" and decode_tokens > 1:
            tpot = (
                e2e - ttft
            ) / (decode_tokens - 1)
        elif finish_reason == "length":
            tpot = 0.0
        else:
            tpot = None

    result = {
        "request_id": request_id,
        "prefill_tokens": len(prompt),
        "decode_tokens": decode_tokens,
        "finish_reason": finish_reason,
        "ttft_ms": None if ttft is None else ttft * 1000,
        "tpot_ms": None if tpot is None else tpot * 1000,
        "e2e_ms": e2e * 1000,
    }

    ttft_str = (
        "N/A"
        if result["ttft_ms"] is None
        else f"{result['ttft_ms']:.3f}"
    )

    tpot_str = (
        "N/A"
        if result["tpot_ms"] is None
        else f"{result['tpot_ms']:.3f}"
    )

    print(
        f"[Request {request_id}] "
        f"TTFT={ttft_str} ms, "
        f"TPOT={tpot_str} ms, "
        f"E2E={result['e2e_ms']:.3f} ms"
    )

    return result


async def send_request_at(
    session,
    args,
    prompt,
    decode_tokens,
    request_id,
    benchmark_start,
    target_arrived_at,
):
    """Submit one request at an absolute deadline and record launch jitter."""

    loop = asyncio.get_running_loop()
    deadline = benchmark_start + target_arrived_at
    delay = deadline - loop.time()

    if delay > 0:
        await asyncio.sleep(delay)

    actual_submit_at = loop.time() - benchmark_start
    arrival_record = {
        "request_id": request_id,
        "target_arrived_at": target_arrived_at,
        "actual_submit_at": actual_submit_at,
        "submit_jitter_ms": (actual_submit_at - target_arrived_at) * 1000,
    }

    result = await send_request(
        session,
        args,
        prompt,
        request_id,
        decode_tokens,
    )

    if result is not None:
        result.update(arrival_record)

    return result, arrival_record


def calc_summary(values):
    values = [
        value
        for value in values
        if value is not None and math.isfinite(value)
    ]

    if not values:
        return None

    arr = np.asarray(values, dtype=np.float64)

    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def write_json(output_path, output):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2)

    temporary_path.replace(output_path)


def write_vidur_trace(trace_path, arrival_records, request_lengths):
    """Write measured client submission times in Vidur trace_replay format."""

    if not arrival_records:
        raise RuntimeError("No request arrival records were captured")

    arrival_records = sorted(arrival_records, key=lambda row: row["request_id"])
    first_submit_at = arrival_records[0]["actual_submit_at"]
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = trace_path.with_suffix(trace_path.suffix + ".tmp")
    fieldnames = [
        "request_id",
        "arrived_at",
        "num_prefill_tokens",
        "num_decode_tokens",
        "target_arrived_at",
        "actual_submit_at",
        "submit_jitter_ms",
    ]

    with temporary_path.open("w", newline="", encoding="utf-8") as trace_file:
        writer = csv.DictWriter(trace_file, fieldnames=fieldnames)
        writer.writeheader()

        for record in arrival_records:
            writer.writerow(
                {
                    "request_id": record["request_id"],
                    # Normalize the observed trace so Vidur request 0 arrives
                    # at simulation time zero.
                    "arrived_at": record["actual_submit_at"] - first_submit_at,
                    "num_prefill_tokens": request_lengths[record["request_id"]][0],
                    "num_decode_tokens": request_lengths[record["request_id"]][1],
                    "target_arrived_at": record["target_arrived_at"],
                    "actual_submit_at": record["actual_submit_at"],
                    "submit_jitter_ms": record["submit_jitter_ms"],
                }
            )

    temporary_path.replace(trace_path)


async def main():
    args = parse_args()
    request_lengths = load_request_lengths(args)
    prompts = {
        length: make_prompt(args, length)
        for length, _ in request_lengths
    }

    vidur_initial_delay, intervals = generate_vidur_arrival_intervals(
        args.num_requests,
        args.qps,
        args.seed,
    )
    planned_arrival_times = intervals_to_arrival_times(intervals)

    timeout = aiohttp.ClientTimeout(
        total=args.timeout
    )

    print("========== Benchmark Config ==========")
    print(f"Host             : {args.host}")
    print(f"Port             : {args.port}")
    print(f"Model            : {args.model}")
    print(f"Requests         : {args.num_requests}")
    print(f"QPS              : {args.qps}")
    if args.request_lengths_file is None:
        print(f"Prefill tokens   : {args.prefill_tokens}")
        print(f"Decode tokens    : {args.decode_tokens}")
    else:
        print(f"Request lengths  : {args.request_lengths_file}")
        print(f"Length pool      : {args.length_pool_size} rows; uniform sampling without replacement")
    print(f"Seed             : {args.seed}")
    print(f"Warmup requests  : {args.warmup_requests}")
    print()

    async with aiohttp.ClientSession(
        timeout=timeout
    ) as session:

        if args.warmup_requests > 0:
            print(
                f"========== Warmup "
                f"({args.warmup_requests} requests) =========="
            )

            for i in range(args.warmup_requests):
                print(f"[Warmup {i}] sending...")
                result = await send_request(
                    session,
                    args,
                    prompts[request_lengths[0][0]],
                    -1 - i,
                    request_lengths[0][1],
                )

                status = (
                    "done"
                    if result is not None
                    else "FAILED"
                )

                print(f"[Warmup {i}] {status}")

            print("Warmup complete.\n")

        loop = asyncio.get_running_loop()
        benchmark_start = loop.time()
        total_start = time.perf_counter()

        # Every request sleeps until its own absolute deadline. A late wakeup
        # therefore affects only that request instead of shifting all later
        # arrivals as a chain of relative asyncio.sleep() calls would.
        tasks = [
            asyncio.create_task(
                send_request_at(
                    session,
                    args,
                    prompts[request_lengths[request_id][0]],
                    request_lengths[request_id][1],
                    request_id,
                    benchmark_start,
                    planned_arrival_times[request_id],
                )
            )
            for request_id in range(args.num_requests)
        ]

        task_outputs = await asyncio.gather(*tasks)

    total_end = time.perf_counter()
    total_runtime_ms = (total_end - total_start) * 1000

    results = [
        result
        for result, _ in task_outputs
        if result is not None
    ]
    arrival_records = [record for _, record in task_outputs]

    ttft = calc_summary(
        [r["ttft_ms"] for r in results]
    )

    tpot = calc_summary(
        [r["tpot_ms"] for r in results]
    )

    e2e = calc_summary(
        [r["e2e_ms"] for r in results]
    )

    print()
    print("========== Results ==========")
    print(f"Successful requests : {len(results)}")
    print(f"Total runtime       : {total_runtime_ms:.3f} ms")

    for name, stats in (
        ("TTFT", ttft),
        ("TPOT", tpot),
        ("E2E", e2e),
    ):
        if stats is None:
            print(f"{name}: N/A")
            continue

        print(
            f"{name}: "
            f"mean={stats['mean']:.3f} ms, "
            f"P50={stats['p50']:.3f}, "
            f"P90={stats['p90']:.3f}, "
            f"P95={stats['p95']:.3f}"
        )

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / args.output_file
    trace_path = Path(args.trace_output_file)

    output = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "vidur_initial_arrival_delay_seconds": vidur_initial_delay,
        "arrival_intervals_seconds": intervals,
        "planned_arrival_times_seconds": planned_arrival_times,
        "arrival_records": arrival_records,
        "vidur_trace_file": str(trace_path),
        "summary": {
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "e2e_ms": e2e,
            "total_runtime_ms": total_runtime_ms,
        },
        "requests": results,
    }

    write_vidur_trace(trace_path, arrival_records, request_lengths)
    write_json(output_path, output)

    print()
    print(f"Saved benchmark result to: {output_path}")
    print(f"Saved Vidur arrival trace to: {trace_path}")


if __name__ == "__main__":
    asyncio.run(main())
