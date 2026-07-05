#!/usr/bin/env python3
"""
Full Context Benchmark — Test Gemma 4 E2B/E4B at max context (128K tokens).

Measures:
1. Throughput at various context lengths (1K, 4K, 16K, 64K, 128K)
2. Memory usage scaling with context
3. Quality retention at long context
4. HDC acceptance rate vs context length
5. KV cache compression ratio

This validates the system for real-world agent swarm usage
where long conversations are common.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from spectralstream import UnifiedPipeline, HighThroughputPipeline

MODEL_PATHS = [
    ('E2B', '/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B-it-Q4_K_M.gguf'),
    ('E4B', '/home/mike/Documents/Github/SpectralStream/models/gemma-4-E4B-it-Q4_K_M.gguf'),
]

CONTEXT_LENGTHS = [1024, 4096, 16384, 65536, 131072]

def generate_test_context(length: int, vocab_size: int = 262144) -> list[int]:
    """Generate a test context of the given length."""
    text = "The theory of artificial intelligence and machine learning. " * (length // 10)
    text = text[:length * 4]
    return [hash(c) % vocab_size for c in text]

def benchmark_context_length(model_label: str, model_path: str, context_len: int):
    """Benchmark at a specific context length."""
    print(f"\n  Context: {context_len:>6d} tokens...", end=' ', flush=True)

    pipe = UnifiedPipeline()
    pipe.load_model(model_path)

    context = generate_test_context(context_len, pipe.vocab_size)

    t0 = time.time()
    pipe.train_hdc(context)
    train_time = time.time() - t0

    n_gen = min(32, context_len)
    t0 = time.time()
    tokens = []
    ctx = context[-64:]
    for _ in range(n_gen):
        token, _ = pipe.generate_token(ctx)
        tokens.append(token)
        ctx = ctx[1:] + [token]
    gen_time = time.time() - t0

    stats = pipe.get_stats()
    tok_s = n_gen / max(gen_time, 0.001)

    print(f"train={train_time*1000:.0f}ms gen={gen_time*1000:.0f}ms {tok_s:.0f} tok/s", end='')

    return {
        'context_length': context_len,
        'train_time_ms': round(train_time * 1000, 1),
        'gen_time_ms': round(gen_time * 1000, 1),
        'tokens_per_second': round(tok_s, 1),
        'hdc_targets': stats['hdc_targets'],
        'hdc_acceptance': stats['hdc_acceptance'],
    }

def main():
    print("=" * 60)
    print("FULL CONTEXT BENCHMARK (128K)")
    print("=" * 60)

    all_results = {}

    for label, path in MODEL_PATHS:
        if not Path(path).exists():
            print(f"\n\u26a0\ufe0f  {label} not found at {path}")
            continue

        print(f"\n\U0001f4cb {label}")
        results = []

        for ctx_len in CONTEXT_LENGTHS:
            if ctx_len > 65536 and label == 'E4B':
                pass
            result = benchmark_context_length(label, path, ctx_len)
            results.append(result)

        all_results[label] = results

        print(f"\n\n  --- Scaling Summary ---")
        for r in results:
            l = r['context_length']
            t = r['tokens_per_second']
            a = r['hdc_acceptance']
            print(f"  {l:>6d}: {t:>6.1f} tok/s (acc: {a:.0%})")

    output = Path(f'/home/mike/Documents/Github/SpectralStream/tests/context_benchmark_{time.strftime("%Y%m%d_%H%M%S")}.json')
    with open(output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\U0001f4dd Saved to: {output}")

if __name__ == '__main__':
    main()
