import argparse
import json
import statistics
import time

import requests
from transformers import AutoTokenizer


URL = "http://127.0.0.1:8002/v1/completions"
MODEL = "/mnt/data02/000000/model/internlm-20b"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt-tokens",
        type=int,
        required=True,
        help="输入 token 数",
    )

    parser.add_argument(
        "--output-tokens",
        type=int,
        default=50,
        help="输出 token 数，默认 50",
    )

    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="测试请求数，默认 10",
    )

    parser.add_argument(
        "--url",
        type=str,
        default=URL,
        help="vLLM API 地址",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="模型名称/路径",
    )

    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="不执行 warmup",
    )

    return parser.parse_args()


args = parse_args()


tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=True,
)


def build_prompt(target_tokens):
    """
    构造固定 token 数的输入。
    """

    base_text = (
        "The development of artificial intelligence has significantly "
        "changed the way computers process and understand information. "
    )

    text = ""

    while True:
        text += base_text

        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if len(ids) >= target_tokens:
            ids = ids[:target_tokens]

            # 直接 decode 截断后的 token
            prompt = tokenizer.decode(
                ids,
                skip_special_tokens=True,
            )

            return prompt


PROMPT = build_prompt(args.prompt_tokens)

actual_prompt_tokens = len(
    tokenizer.encode(
        PROMPT,
        add_special_tokens=False,
    )
)


def send_request(request_id):
    payload = {
        "model": args.model,
        "prompt": PROMPT,
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
    }

    start_time = time.perf_counter()

    first_token_time = None
    end_time = None

    output_text = ""

    with requests.post(
        args.url,
        headers={"Content-Type": "application/json"},
        json=payload,
        stream=True,
        timeout=300,
    ) as response:

        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):

            if not line:
                continue

            if not line.startswith("data:"):
                continue

            data = line[len("data:"):].strip()

            if data == "[DONE]":
                end_time = time.perf_counter()
                break

            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = obj.get("choices", [])

            if not choices:
                continue

            text = choices[0].get("text", "")

            if text:
                now = time.perf_counter()

                if first_token_time is None:
                    first_token_time = now

                output_text += text

    if end_time is None:
        end_time = time.perf_counter()

    if first_token_time is None:
        raise RuntimeError(
            f"Request {request_id}: no output received"
        )

    output_token_ids = tokenizer.encode(
        output_text,
        add_special_tokens=False,
    )

    num_output_tokens = len(output_token_ids)

    ttft = first_token_time - start_time
    e2e = end_time - start_time

    if num_output_tokens > 1:
        tpot = (
            e2e - ttft
        ) / (num_output_tokens - 1)
    else:
        tpot = 0.0

    return {
        "request_id": request_id,
        "ttft": ttft,
        "tpot": tpot,
        "e2e": e2e,
        "output_tokens": num_output_tokens,
    }


def main():
    results = []

    print("=" * 90)
    print(f"Server           : {args.url}")
    print(f"Model            : {args.model}")
    print(f"Requests         : {args.num_requests}")
    print(f"Prompt target    : {args.prompt_tokens}")
    print(f"Prompt actual    : {actual_prompt_tokens}")
    print(f"Output tokens    : {args.output_tokens}")
    print("=" * 90)

    if not args.no_warmup:
        print("Warmup...")
        send_request(0)
        print("Warmup finished.\n")

    for i in range(args.num_requests):

        r = send_request(i + 1)
        results.append(r)

        print(
            f"Request {i + 1:2d} | "
            f"TTFT {r['ttft'] * 1000:9.3f} ms | "
            f"TPOT {r['tpot'] * 1000:9.3f} ms/token | "
            f"E2E {r['e2e'] * 1000:9.3f} ms | "
            f"Output {r['output_tokens']:3d}"
        )

    mean_ttft = statistics.mean(
        r["ttft"] for r in results
    ) * 1000

    mean_tpot = statistics.mean(
        r["tpot"] for r in results
    ) * 1000

    mean_e2e = statistics.mean(
        r["e2e"] for r in results
    ) * 1000

    print()
    print("=" * 90)
    print("Average")
    print("=" * 90)

    print(f"Mean TTFT : {mean_ttft:.3f} ms")
    print(f"Mean TPOT : {mean_tpot:.3f} ms/token")
    print(f"Mean E2E  : {mean_e2e:.3f} ms")

    print("=" * 90)


if __name__ == "__main__":
    main()