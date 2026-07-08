"""
Download the full WikiText-2 test set and write it as raw text.

Usage::

    python scripts/fetch_eval_corpus.py [--output eval/data/wikitext2_test.txt]

Downloads the public WikiText-2 dataset from Hugging Face's ``datasets``
mirror URL using only the stdlib (``urllib.request``) — no ``datasets``,
``requests``, or other pip packages required.
"""

from __future__ import annotations

import argparse
import os
import urllib.request

WIKITEXT2_TEST_URL = (
    "https://raw.githubusercontent.com/"
    "huggingface/datasets/refs/heads/main/data/wikitext/wikitext-2/wiki.test.raw"
)

DEFAULT_OUTPUT = "eval/data/wikitext2_test.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the full WikiText-2 test corpus as raw text"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    out_path: str = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"Downloading WikiText-2 test set from:\n  {WIKITEXT2_TEST_URL}")
    print(f"Saving to: {out_path}")

    urllib.request.urlretrieve(WIKITEXT2_TEST_URL, out_path)

    with open(out_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Downloaded {len(lines)} lines ({sum(len(l) for l in lines)} bytes)")


if __name__ == "__main__":
    main()
