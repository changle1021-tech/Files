import csv
import re
import subprocess
import sys

INPUT_LENGTHS = [128, 256, 512, 1024, 2048, 3072]

MODEL = "NousResearch/Llama-2-7b-hf"
OUTPUT_LEN = 50
NUM_PROMPTS = 128
MAX_CONCURRENCY = 8

CSV_FILE = "bench_summary.csv"


def extract(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "input_len",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_e2e_ms",
        "total_duration_s",
    ])

    for input_len in INPUT_LENGTHS:

        cmd = [
            sys.executable,
            "-m",
            "sglang.bench_serving",
            "--backend", "sglang",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--model", MODEL,
            "--dataset-name", "random",
            "--num-prompts", str(NUM_PROMPTS),
            "--max-concurrency", str(MAX_CONCURRENCY),
            "--random-output-len", str(OUTPUT_LEN),
            "--random-range-ratio", "1",
            "--random-input-len", str(input_len),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output = result.stdout
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

print(f"结果已保存到 {CSV_FILE}")