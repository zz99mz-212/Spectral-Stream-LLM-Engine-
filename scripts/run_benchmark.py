#!/usr/bin/env python3
"""Run comprehensive benchmarks on compressed models via the InferencePipeline."""

from __future__ import annotations

import json
import sys
import os
from typing import Dict, Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectralstream.inference.pipeline import InferencePipeline, InferenceConfig


def run_inference_benchmark(
    model_path: str,
    prompt_lengths=None,
    num_runs: int = 3,
    kv_method: str = "none",
    kv_cache_gb: float = 4.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the full inference benchmark suite."""
    if prompt_lengths is None:
        prompt_lengths = [128, 512, 2048]

    config = InferenceConfig(
        model_path=model_path,
        cache_size_gb=2.0,
        kv_cache_size_gb=kv_cache_gb,
        kv_cache_method=kv_method,
        verbose=verbose,
    )

    print(f"Loading model: {model_path}")
    pipeline = InferencePipeline(model_path, config)
    print(f"  Config: {pipeline.model_config.__class__.__name__}")
    print(f"  Layers: {len(pipeline.layers)}")
    print(f"  Tensors: {len(pipeline.tensor_names)}")

    try:
        print(f"\nBenchmarking throughput ({num_runs}x runs)...")
        results = pipeline.benchmark(prompt_lengths=prompt_lengths, num_runs=num_runs)

        print("\n── Throughput ──")
        for seq_key, seq_data in results.get("throughput", {}).items():
            print(f"  {seq_key}:")
            for k, v in seq_data.items():
                print(f"    {k}: {v}")

        print("\n── Memory ──")
        for k, v in results.get("memory", {}).items():
            print(f"  {k}: {v}")

        print("\n── Compression ──")
        for k, v in results.get("compression", {}).items():
            print(f"  {k}: {v}")

        # Quick perplexity check
        fake_vocab_size = min(pipeline.model_config.VOCAB_SIZE, 32000)
        rng = np.random.RandomState(42)
        test_tokens = rng.randint(1, fake_vocab_size, size=1024).tolist()
        print(f"\nMeasuring perplexity on 1024 tokens...")
        ppl = pipeline.measure_perplexity(test_tokens, stride=512)
        print(f"  Perplexity: {ppl:.2f}")
        results["perplexity"] = round(ppl, 2)

    finally:
        pipeline.close()

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run comprehensive inference benchmarks on compressed models"
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default="models/gemma-4-E2B/model.safetensors",
        help="Path to model (.ssf or .safetensors)",
    )
    parser.add_argument(
        "--prompt-lengths",
        type=str,
        default="128,512,2048",
        help="Comma-separated list of prompt sequence lengths",
    )
    parser.add_argument(
        "--num-runs", type=int, default=3, help="Number of benchmark runs"
    )
    parser.add_argument(
        "--kv-method",
        type=str,
        default="none",
        help="KV cache compression method (none, fwht_int8, dct_sparse, etc.)",
    )
    parser.add_argument(
        "--kv-cache-gb", type=float, default=4.0, help="KV cache size limit in GB"
    )
    parser.add_argument(
        "--output", type=str, default="", help="Save results to JSON file"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    prompt_lengths = [int(x) for x in args.prompt_lengths.split(",")]

    results = run_inference_benchmark(
        model_path=args.model_path,
        prompt_lengths=prompt_lengths,
        num_runs=args.num_runs,
        kv_method=args.kv_method,
        kv_cache_gb=args.kv_cache_gb,
        verbose=args.verbose,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
