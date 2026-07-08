"""
WikiText-2 perplexity eval CLI.

Usage::

    python -m eval.run_eval \\
        --model models/gemma-4-E2B/model.safetensors \\
        --compressed output/compressed.ssf \\
        [--corpus eval/data/wikitext2_sample.tokens.json] \\
        [--tokenizer tokenizer.json] \\
        [--seq-len 2048] [--stride 512] [--threshold 0.95] \\
        [--output eval/artifacts/result.json]

Exits with code ``0`` if the recovery gate passes, ``1`` if it fails.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from eval.artifact import write_artifact
from eval.constants import DEFAULT_SEQ_LEN, DEFAULT_STRIDE, RECOVERY_GATE_THRESHOLD
from eval.corpus import resolve_corpus
from eval.grader import grade
from eval.model_path import resolve_model_path
from spectralstream.utils.tokenizer_engine import AutoTokenizer, build_default_tokenizer

logger = logging.getLogger("spectralstream.eval")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate compression quality via WikiText-2 perplexity",
    )
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help="Path to the base (uncompressed) model file",
    )
    parser.add_argument(
        "--compressed",
        required=False,
        default=None,
        help="Path to the compressed .ssf model file",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Path to corpus file (.json for pre-tokenized, .txt for raw text)",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Path to tokenizer.json or .gguf for model-native tokenization",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f"Sliding-window sequence length (default {DEFAULT_SEQ_LEN})",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"Slide stride (default {DEFAULT_STRIDE})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=RECOVERY_GATE_THRESHOLD,
        help=f"Recovery gate threshold (default {RECOVERY_GATE_THRESHOLD})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the eval artifact JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the eval grader.

    Returns
    -------
    int
        0 if gate passed, 1 if gate failed, 2 on error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Silently configure logging if not already set up
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    try:
        # Resolve model paths
        logger.info("Resolving model paths...")
        base_path = resolve_model_path(args.model)
        compressed_path = resolve_model_path(args.compressed)
        logger.info("  Base model: %s", base_path)
        logger.info("  Compressed: %s", compressed_path)

        # Resolve corpus
        logger.info("Loading corpus...")

        # Load tokenizer: model-native if --tokenizer supplied, byte-level default otherwise
        if args.tokenizer is None:
            tokenizer = build_default_tokenizer()
            tokenizer_name = "default"
        elif args.tokenizer.endswith(".gguf"):
            try:
                tokenizer = AutoTokenizer.from_gguf(args.tokenizer)
                tokenizer_name = f"auto_from_gguf:{os.path.basename(args.tokenizer)}"
            except Exception:
                logger.warning(
                    "Failed to load GGUF tokenizer from %s, falling back to default",
                    args.tokenizer,
                )
                tokenizer = build_default_tokenizer()
                tokenizer_name = "default"
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
                tokenizer_name = f"auto_from_pretrained:{os.path.basename(args.tokenizer)}"
            except Exception:
                logger.warning(
                    "Failed to load tokenizer from %s, falling back to default",
                    args.tokenizer,
                )
                tokenizer = build_default_tokenizer()
                tokenizer_name = "default"

        logger.info("Tokenizer: %s", tokenizer_name)
        test_tokens = resolve_corpus(args.corpus, tokenizer=tokenizer)
        logger.info("  %d tokens loaded", len(test_tokens))

        # Method name from compressed filename
        method_name = "ssf"
        if args.compressed:
            import os as _os

            method_name = _os.path.splitext(_os.path.basename(args.compressed))[0]

        # Run eval
        logger.info("Running eval (base → compressed)...")
        artifact = grade(
            base_path=base_path,
            compressed_path=compressed_path,
            test_tokens=test_tokens,
            seq_len=args.seq_len,
            stride=args.stride,
            threshold=args.threshold,
            tokenizer_name=tokenizer_name,
            method_name=method_name,
        )

        # Write artifact
        out_path = write_artifact(artifact, output_path=args.output)
        logger.info("Artifact written to %s", out_path)

        # Summary
        logger.info(
            "Results: base_ppl=%.4f  compressed_ppl=%.4f  "
            "recovery_ratio=%.4f  gate_passed=%s",
            artifact["base_ppl"],
            artifact["compressed_ppl"],
            artifact["recovery_ratio"],
            artifact["gate_passed"],
        )

        return 0 if artifact["gate_passed"] else 1

    except (FileNotFoundError, ValueError) as e:
        logger.error("Error: %s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
