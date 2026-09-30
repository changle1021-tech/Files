#!/usr/bin/env python3
"""Download every conversation for one LMSYS source model via server filtering.

The complete matching subset is saved as UTF-8 JSONL. No tokenization, length
filter, or request-count limit is applied. Only the Python standard library is
required. The dataset's Hugging Face access conditions must be accepted first.
"""

import argparse
import getpass
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "lmsys/lmsys-chat-1m"
API = "https://datasets-server.huggingface.co"


def request_json(endpoint, params, token):
    url = API + endpoint + "?" + urllib.parse.urlencode(params)
    headers = {"Accept": "application/json", "User-Agent": "lmsys-subset-downloader/1"}
    if token:
        headers["Authorization"] = "Bearer " + token
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < 4:
                print(f"HTTP {exc.code}; retrying...", file=sys.stderr, flush=True)
                time.sleep(min(2 ** (attempt + 1), 30))
                continue
            if exc.code in (401, 403):
                raise RuntimeError(
                    "Access denied. Accept the dataset conditions on Hugging Face "
                    "and supply a token with access to gated datasets."
                ) from None
            raise RuntimeError(
                f"Hugging Face {endpoint} returned HTTP {exc.code}. "
                "Server filtering may be unavailable; no full-dataset download was attempted."
            ) from None
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < 4:
                print("Network error; retrying...", file=sys.stderr, flush=True)
                time.sleep(min(2 ** (attempt + 1), 30))
                continue
            raise RuntimeError("Could not reach the Hugging Face filtering API.") from None


def download_subset(output, source_model, config, token):
    status = request_json("/is-valid", {"dataset": DATASET}, token)
    if status.get("filter") is not True:
        raise RuntimeError("Server filtering is unavailable for this dataset.")
    params = {
        "dataset": DATASET, "config": config, "split": "train",
        "where": '"model"=\'' + source_model.replace("'", "''") + "'",
        "length": 100,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    total = None
    count = 0
    last_index = -1
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=output.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            while total is None or count < total:
                page = request_json("/filter", dict(params, offset=count), token)
                if page.get("partial") is not False:
                    raise RuntimeError("The server index is incomplete; refusing a partial subset.")
                page_total = page.get("num_rows_total")
                if not isinstance(page_total, int) or isinstance(page_total, bool) or page_total < 0:
                    raise RuntimeError("The server returned an invalid matching-row count.")
                if total is None:
                    total = page_total
                    print(f"Source model: {source_model}; matching conversations: {total:,}", flush=True)
                    if total == 0:
                        raise RuntimeError("No conversations match this source model.")
                elif total != page_total:
                    raise RuntimeError("The matching-row count changed during download; retry.")
                rows = page.get("rows")
                if not isinstance(rows, list) or not rows or len(rows) > min(100, total - count):
                    raise RuntimeError("The server returned an incomplete or invalid page.")
                for item in rows:
                    if not isinstance(item, dict) or item.get("truncated_cells") != []:
                        raise RuntimeError("The API returned truncated conversation data.")
                    row = item.get("row")
                    index = item.get("row_idx")
                    if not isinstance(index, int) or index <= last_index:
                        raise RuntimeError("The API returned duplicated or unordered rows.")
                    if not isinstance(row, dict) or row.get("model") != source_model:
                        raise RuntimeError("The API returned a different source model.")
                    if not isinstance(row.get("conversation"), list):
                        raise RuntimeError("The API returned invalid conversation data.")
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                    last_index = index
                handle.flush()
                if count == len(rows):
                    estimated_mb = temporary.stat().st_size / count * total / 1_000_000
                    print(f"Estimated JSONL size from the first page: {estimated_mb:.1f} MB (rough estimate)", flush=True)
                if count % 1000 == 0 or count == total:
                    print(f"Downloaded {count:,}/{total:,} conversations", flush=True)
        temporary.replace(output)
        print(f"Saved {count:,} conversations to {output.resolve()}", flush=True)
        print(f"File size: {output.stat().st_size:,} bytes ({output.stat().st_size / 1_000_000:.2f} MB)", flush=True)
        return count
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", default="llama-2-7b-chat")
    parser.add_argument("--config", default="default")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-token", nargs="?", const="__PROMPT_FOR_TOKEN__",
                        help="With no value, prompt privately for the Hugging Face token")
    args = parser.parse_args()
    if args.hf_token == "__PROMPT_FOR_TOKEN__":
        if not sys.stdin.isatty():
            parser.error("--hf-token without a value requires an interactive terminal")
        token = getpass.getpass("Hugging Face token: ").strip()
        if not token:
            parser.error("Hugging Face token cannot be empty")
    else:
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            try:
                from huggingface_hub import get_token
                token = get_token()
            except ImportError:
                pass
    try:
        download_subset(args.output, args.source_model, args.config, token)
    except KeyboardInterrupt:
        parser.exit(130, "Download interrupted; destination file was not replaced.\n")
    except Exception as exc:
        message = str(exc)
        if token:
            message = message.replace(token, "[REDACTED]")
        parser.exit(2, f"Download failed: {message}\n")


if __name__ == "__main__":
    main()
