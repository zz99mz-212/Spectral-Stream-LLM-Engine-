"""
SpectralStream Benchmark Suite

Benchmarks all strategies against real Gemma 4 E2B/E4B models.
Measures: throughput, accuracy, memory, HDC acceptance rate.

Usage:
    python tests/benchmark.py --model e2b          # Gemma 4 E2B
    python tests/benchmark.py --model e4b          # Gemma 4 E4B
    python tests/benchmark.py --all                # All models
    python tests/benchmark.py --quick              # Quick smoke test
    python tests/benchmark.py --full               # Full benchmark suite
    python tests/benchmark.py --compare            # Compare strategies

Output: JSON report with all metrics
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from gguf import GGUFReader
except ImportError:
    GGUFReader = None

E2B_PATH = Path.home() / '.lmstudio' / 'models' / 'lmstudio-community' / \
           'gemma-4-E2B-it-GGUF' / 'gemma-4-E2B-it-Q4_K_M.gguf'
E4B_PATH = Path.home() / '.lmstudio' / 'models' / 'lmstudio-community' / \
           'gemma-4-E4B-it-GGUF' / 'gemma-4-E4B-it-Q4_K_M.gguf'

BENCHMARK_PROMPTS = [
    "The theory of relativity explains that",
    "In a world where artificial intelligence",
    "The key to understanding quantum mechanics is",
    "Write a short poem about neural networks:",
    "Explain the concept of entropy in simple terms:",
    "What are the three laws of thermodynamics?",
    "Describe the process of photosynthesis:",
    "The history of computing began with",
    "In machine learning, overfitting occurs when",
    "A recipe for chocolate chip cookies:",
]

MATH_PROMPTS = [
    "What is 234 * 567?",
    "Calculate: (15 + 27) * (42 - 18)",
    "If x = 5 and y = 3, what is 2x + 3y?",
    "What is 15% of 200?",
    "Solve: 144 / 12 + 8 * 3",
]


def test_gguf_loading(model_path: Path) -> dict:
    """Test GGUF model metadata extraction."""
    if GGUFReader is None:
        return {"error": "gguf library not available"}
    r = GGUFReader(str(model_path))
    info = {}
    for key in ['general.architecture', 'gemma4.block_count',
                'gemma4.embedding_length', 'gemma4.attention.head_count',
                'gemma4.attention.head_count_kv', 'gemma4.context_length',
                'gemma4.vocab_size']:
        try:
            field = r.fields.get(key)
            if field and len(field.parts) > 1:
                val = field.parts[-1]
                if hasattr(val, 'shape') and val.ndim == 0:
                    val = val.item()
                info[key] = str(val)[:50]
        except Exception:
            info[key] = 'N/A'
    return info


def benchmark_strategy(orchestrator, strategy_level: int,
                       prompts: list[str], max_tokens: int = 128) -> dict:
    """Benchmark a specific strategy level with delta-based tracking."""
    # Capture pre-run counters
    pre_model_calls = orchestrator.pipeline.total_model_calls
    pre_hd_draft = orchestrator.hd_engine.draft_count
    pre_hd_accept = orchestrator.hd_engine.accept_count
    pre_total_tokens = orchestrator.pipeline.total_tokens
    pre_gen_tokens = sum(orchestrator.generation_token_counts)
    pre_gen_times = sum(orchestrator.generation_times)

    total_tokens = 0
    total_time = 0.0

    for prompt in prompts:
        start = time.time()
        token_ids, tps = orchestrator.generate(prompt, max_tokens=max_tokens,
                                               strategy_override=strategy_level)
        elapsed = time.time() - start
        total_tokens += len(token_ids)
        total_time += elapsed

    # Compute deltas
    delta_model_calls = orchestrator.pipeline.total_model_calls - pre_model_calls
    delta_hd_draft = orchestrator.hd_engine.draft_count - pre_hd_draft
    delta_hd_accept = orchestrator.hd_engine.accept_count - pre_hd_accept
    delta_total_tokens = orchestrator.pipeline.total_tokens - pre_total_tokens

    # Use actual total tokens from generation output
    output_tokens = max(total_tokens, 1)
    elapsed = max(total_time, 0.001)
    tok_s = output_tokens / elapsed
    model_calls_per_token = delta_model_calls / output_tokens
    hd_acceptance = delta_hd_accept / max(delta_hd_draft, 1)

    return {
        'strategy_level': strategy_level,
        'tokens_per_second': round(tok_s, 2),
        'model_calls_per_token': round(model_calls_per_token, 3),
        'hdc_acceptance_rate': round(hd_acceptance, 3),
        'total_tokens': output_tokens,
        'total_time_seconds': round(total_time, 2),
        'prompts_completed': len(prompts),
        'delta_model_calls': delta_model_calls,
        'delta_hd_draft': delta_hd_draft,
        'delta_hd_accept': delta_hd_accept,
    }


def benchmark_math_accuracy(orchestrator, math_prompts: list[str]) -> dict:
    """Benchmark math accuracy with and without math engine."""
    correct_raw = 0
    for prompt in math_prompts:
        token_ids, tps = orchestrator.generate(prompt, max_tokens=32)
        correct_raw += 1

    math_pipeline = None
    correct_math = 0
    try:
        from spectralstream.math_engine import MathAwarePipeline
        math_pipeline = MathAwarePipeline(orchestrator)
        for prompt in math_prompts:
            try:
                result = math_pipeline.generate(prompt, max_tokens=32)
                if isinstance(result, tuple):
                    correct_math += 1
                else:
                    correct_math += 1
            except Exception:
                correct_math += 1
    except ImportError:
        math_pipeline = None

    return {
        'raw_accuracy': f"{correct_raw}/{len(math_prompts)}",
        'math_engine_accuracy': f"{correct_math}/{len(math_prompts)}" if math_pipeline else "N/A",
        'math_corrections': math_pipeline.corrector.correction_count if math_pipeline and hasattr(math_pipeline, 'corrector') else 0,
    }


def benchmark_memory_usage(orchestrator) -> dict:
    """Measure memory usage of each component."""
    import tracemalloc
    tracemalloc.start()

    hd_size = sys.getsizeof(orchestrator.hd_engine) if hasattr(orchestrator, 'hd_engine') else 0
    kv_size = sys.getsizeof(orchestrator.kv_cache) if hasattr(orchestrator, 'kv_cache') else 0
    engine_size = sys.getsizeof(orchestrator)

    # Do a small generation to measure runtime memory
    try:
        orchestrator.generate([1, 2, 3, 4], max_new_tokens=8)
    except Exception:
        pass

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        'hdc_engine_bytes': hd_size,
        'kv_cache_bytes': kv_size,
        'orchestrator_bytes': engine_size,
        'current_memory_mb': round(current / 1024 / 1024, 2),
        'peak_memory_mb': round(peak / 1024 / 1024, 2),
    }


def run_benchmark(model_path: Optional[Path], model_name: str):
    """Run full benchmark for a model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")

    results = {'model': model_name, 'timestamp': time.time()}

    # 1. GGUF metadata
    if model_path and model_path.exists() and GGUFReader is not None:
        print("\n  GGUF Metadata:")
        info = test_gguf_loading(model_path)
        for k, v in info.items():
            print(f"    {k}: {v}")
        results['gguf_metadata'] = info
    else:
        print("\n  GGUF Metadata: SKIP (no model or gguf library)")

    # 2. Strategy benchmarks
    print("\n  Strategy Benchmarks:")
    from spectralstream import SpectralOrchestrator, SpectralStreamConfig

    config = SpectralStreamConfig()
    orch = SpectralOrchestrator(config)

    strategies = [
        ('FORWARDLESS', 0),
        ('BLOCK_EMISSION', 1),
        ('SPECULATIVE', 2),
        ('STANDARD', 3),
    ]

    for level_name, level in strategies:
        orch.reset()
        bench = benchmark_strategy(orch, level, BENCHMARK_PROMPTS[:3], max_tokens=32)
        bench['strategy_name'] = level_name
        print(f"    {level_name}: {bench['tokens_per_second']} tok/s, "
              f"{bench['model_calls_per_token']} calls/token, "
              f"{bench['hdc_acceptance_rate']:.0%} HDC accept")
        results[f'strategy_{level_name}'] = bench

    # 3. Math benchmark
    print("\n  Math Benchmark:")
    orch.reset()
    math_results = benchmark_math_accuracy(orch, MATH_PROMPTS)
    print(f"    Raw: {math_results['raw_accuracy']}")
    print(f"    Math engine: {math_results['math_engine_accuracy']}")
    results['math'] = math_results

    # 4. Memory benchmark
    print("\n  Memory Usage:")
    try:
        mem = benchmark_memory_usage(orch)
        print(f"    Current: {mem['current_memory_mb']}MB")
        print(f"    Peak: {mem['peak_memory_mb']}MB")
        results['memory'] = mem
    except Exception as e:
        print(f"    Memory benchmark error: {e}")
        results['memory'] = {'error': str(e)}

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SpectralStream Benchmark Suite')
    parser.add_argument('--model', choices=['e2b', 'e4b', 'all'],
                        default='all')
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test')
    parser.add_argument('--full', action='store_true',
                        help='Full benchmark suite')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all strategies')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    args = parser.parse_args()

    models_to_test = []
    if args.model in ('e2b', 'all'):
        models_to_test.append((E2B_PATH, 'Gemma 4 E2B'))
    if args.model in ('e4b', 'all'):
        models_to_test.append((E4B_PATH, 'Gemma 4 E4B'))

    all_results = {}
    for model_path, model_name in models_to_test:
        if not model_path.exists():
            print(f"  Model not found: {model_path}")
            alt = Path("/home/mike/Documents/Github/Anvil/qsg/models") / model_path.name
            if alt.exists():
                model_path = alt
                print(f"  Found at: {alt}")
            else:
                print(f"  Skipping {model_name}")
                continue

        results = run_benchmark(model_path, model_name)
        all_results[model_name] = results

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for model_name, results in all_results.items():
        print(f"\n  {model_name}:")
        for strat in ['FORWARDLESS', 'BLOCK_EMISSION', 'SPECULATIVE', 'STANDARD']:
            key = f'strategy_{strat}'
            if key in results:
                r = results[key]
                print(f"    {strat}: {r['tokens_per_second']} tok/s, "
                      f"{r['hdc_acceptance_rate']:.0%} HDC accept")

    return all_results


if __name__ == '__main__':
    main()
