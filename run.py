import argparse
import csv
import re
import subprocess
import sys


def extract(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        default="sglang",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "--model",
        default="NousResearch/Llama-2-7b-hf",
    )

    parser.add_argument(
        "--dataset-name",
        default="random",
    )

    parser.add_argument(
        "--num-prompts",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--request-rate",
        type=float,
        default=24,
    )

    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--output-len",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=1,
    )

    parser.add_argument(
        "--input-lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 3072],
    )

    parser.add_argument(
        "--csv-file",
        default="bench_summary.csv",
    )

    return parser.parse_args()


args = parse_args()


with open(args.csv_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "input_len",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_e2e_ms",
        "total_duration_s",
    ])

    for input_len in args.input_lengths:

        cmd = [
            sys.executable,
            "-m",
            "sglang.bench_serving",

            "--backend",
            args.backend,

            "--host",
            args.host,

            "--port",
            str(args.port),

            "--model",
            args.model,

            "--dataset-name",
            args.dataset_name,

            "--num-prompts",
            str(args.num_prompts),

            "--request-rate",
            str(args.request_rate),

            "--max-concurrency",
            str(args.max_concurrency),

            "--random-output-len",
            str(args.output_len),

            "--random-range-ratio",
            str(args.random_range_ratio),

            "--random-input-len",
            str(input_len),
        ]

        print("\n========================================")
        print(f"Input length: {input_len}")
        print("========================================")

        print(" ".join(cmd))
        print()

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output = result.stdout

        # 显示 SGLang 原始结果
        print(output)

        ttft = extract(
            r"Mean TTFT \(ms\):\s+([\d.]+)",
            output,
        )

        tpot = extract(
            r"Mean TPOT \(ms\):\s+([\d.]+)",
            output,
        )

        e2e = extract(
            r"Mean E2E Latency \(ms\):\s+([\d.]+)",
            output,
        )

        duration = extract(
            r"Benchmark duration \(s\):\s+([\d.]+)",
            output,
        )

        writer.writerow([
            input_len,
            ttft,
            tpot,
            e2e,
            duration,
        ])

        f.flush()


print(f"\n结果已保存到 {args.csv_file}")