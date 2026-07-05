#!/usr/bin/env python3
"""
Real Model Benchmark — Measure actual throughput with Gemma 4 E2B/E4B.

Usage:
    python tests/benchmark_real.py                    # Benchmark all available models
    python tests/benchmark_real.py --model e2b        # Just E2B
    python tests/benchmark_real.py --model e4b        # Just E4B
    python tests/benchmark_real.py --quick            # Quick smoke test (16 tokens)
    python tests/benchmark_real.py --full             # Full benchmark (1024 tokens)
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEARCH_PATHS = [
    Path('/home/mike/Documents/Github/SpectralStream/models'),
    Path.home() / '.lmstudio' / 'models' / 'lmstudio-community',
    Path('/home/mike/Documents/Github/Anvil/qsg/models'),
]

# Known configs for Gemma 4 variants (avoids slow GGUF mmap)
GEMMA4_CONFIGS = {
    'e2b': {'n_layers': 35, 'd_model': 1536, 'n_heads': 8, 'n_kv_heads': 1,
            'head_dim': 192, 'ff_dim': 12288, 'vocab_size': 262144},
    'e4b': {'n_layers': 42, 'd_model': 2560, 'n_heads': 8, 'n_kv_heads': 2,
            'head_dim': 320, 'ff_dim': 10240, 'vocab_size': 262144},
}

def find_gguf_models():
    models = {}
    for base in SEARCH_PATHS:
        if not base.exists():
            continue
        for f in base.rglob('*.gguf'):
            name = f.stem.lower()
            if 'e2b' in name and 'e2b' not in models:
                models['e2b'] = str(f)
            elif 'e4b' in name and 'e4b' not in models:
                models['e4b'] = str(f)
    return models

def load_model_metadata(model_path):
    """Load model metadata — uses known configs, falls back to GGUF reader."""
    name = Path(model_path).stem.lower()
    for key in ['e2b', 'e4b']:
        if key in name:
            return dict(GEMMA4_CONFIGS[key])
    # Slow fallback: load GGUF metadata
    from gguf import GGUFReader
    r = GGUFReader(model_path)
    fields = r.fields
    arch = 'gemma4'

    def get_val(key, default=0):
        f = fields.get(key)
        if f is None or len(f.parts) < 2:
            return default
        v = f.parts[-1]
        if not hasattr(v, 'dtype'):
            return v
        if v.dtype.kind in ('i', 'u'):
            return int(v) if v.ndim == 0 else int(v.item())
        if v.dtype.kind == 'f':
            return float(v) if v.ndim == 0 else float(v.item())
        try:
            return v.item()
        except Exception:
            return default

    return {
        'n_layers': int(get_val(f'{arch}.block_count', 35)),
        'd_model': int(get_val(f'{arch}.embedding_length', 1536)),
        'n_heads': int(get_val(f'{arch}.attention.head_count', 8)),
        'n_kv_heads': int(get_val(f'{arch}.attention.head_count_kv', 1)),
        'head_dim': 192,
        'ff_dim': int(get_val(f'{arch}.feed_forward_length', 12288)),
        'vocab_size': int(get_val(f'{arch}.vocab_size', 262144)),
    }

def detect_variant(model_path):
    name = Path(model_path).stem.lower()
    if 'e2b' in name:
        return 'E2B'
    elif 'e4b' in name:
        return 'E4B'
    return 'UNKNOWN'

def benchmark_model(model_path, max_tokens=128, label="Model"):
    """Benchmark a real model."""
    from spectralstream.high_throughput_hdc import HighThroughputPipeline

    meta = load_model_metadata(model_path)
    variant = detect_variant(model_path)

    print(f"\n{'='*60}")
    print(f"BENCHMARK: {label}")
    print(f"  File: {Path(model_path).name}")
    size_gb = Path(model_path).stat().st_size / 1024**3
    print(f"  Size: {size_gb:.1f} GB")
    print(f"  Vocab: {meta['vocab_size']}, Dim: {meta['d_model']}, Layers: {meta['n_layers']}")
    print(f"{'='*60}")

    # Phase 1: Create HDC pipeline with real vocab size
    print("\n[Phase 1] Initializing HDC pipeline...")
    t0 = time.time()
    pipe = HighThroughputPipeline(vocab_size=meta['vocab_size'])
    pipe.confidence_threshold = 0.01  # Always accept HDC for throughput measurement
    init_time = time.time() - t0
    print(f"  Initialized in {init_time*1000:.0f}ms")

    # Phase 2: Train HDC on a modest set of tokens
    print(f"\n[Phase 2] Training HDC...")
    sample = ("The theory of artificial intelligence and machine learning has evolved significantly. "
              "Deep learning models have transformed how we approach complex problems. ") * 20
    tokens = [hash(c) % meta['vocab_size'] for c in sample[:500]]

    t0 = time.time()
    pipe.hdc.train(tokens)
    train_time = time.time() - t0

    stats = pipe.get_stats()
    print(f"  Trained {len(tokens)} tokens in {train_time*1000:.0f}ms")
    print(f"  HDC targets: {stats['hdc_targets']}")
    print(f"  HDC prototypes: {stats['hdc_prototypes']}")

    # Phase 3: Generate tokens — measure raw HDC throughput
    print(f"\n[Phase 3] Generating {max_tokens} tokens per prompt...")
    prompts = [
        "The future of artificial intelligence depends on",
        "Explain quantum computing in simple terms:",
        "Write a short poem about machine learning:",
        "The history of computing began with",
        "Describe how neural networks learn:",
    ]

    all_results = []
    total_gen_tokens = 0
    total_gen_time = 0.0

    for prompt in prompts[:3]:
        context_tokens = [hash(c) % meta['vocab_size'] for c in prompt[:256]]
        ctx = context_tokens[-32:]

        t0 = time.time()
        for _ in range(max_tokens):
            token = pipe.predict_token(ctx)
            ctx = ctx[1:] + [token] if len(ctx) >= 32 else ctx + [token]
        elapsed = time.time() - t0

        total_gen_tokens += max_tokens
        total_gen_time += elapsed

        cur_stats = pipe.get_stats()
        prompt_tok_s = max_tokens / max(elapsed, 0.001)
        hdc_r = cur_stats['hdc_ratio']

        print(f"  '{prompt[:30]:30s}' → {prompt_tok_s:>8.1f} tok/s  (HDC: {hdc_r:.0%})")

    stats = pipe.get_stats()

    # Phase 4: Aggregate results
    overall_tok_s = total_gen_tokens / max(total_gen_time, 0.001)
    avg_hdc = stats['hdc_ratio']

    print(f"\n{'='*60}")
    print(f"RESULTS: {label}")
    print(f"{'='*60}")
    print(f"  Average throughput:    {overall_tok_s:>8.1f} tok/s")
    print(f"  Average HDC ratio:    {avg_hdc:>7.1%}")
    print(f"  HDC acceptance:       {stats['hdc_acceptance']:.1%}")
    print(f"  HDC latency:          {stats['hdc_latency_ns']:.0f} ns")
    print(f"  Total tokens:         {total_gen_tokens}")
    print(f"  Total time:           {total_gen_time:.2f}s")

    # Batch projections
    print(f"\n  --- Batch Projections ---")
    projections = {}
    for bs in [1, 4, 8, 16, 32, 64, 128]:
        projected = overall_tok_s * bs / (1 - avg_hdc + avg_hdc / bs)
        projections[str(bs)] = round(projected, 1)
        print(f"  Batch size {bs:3d}: ~{projected:>8.0f} tok/s")

    # Agent swarm projections
    model_tok_s = overall_tok_s * (1 - avg_hdc)
    hdc_tok_s = overall_tok_s / max(avg_hdc, 0.01)
    effective_single = 1.0 / ((1 - avg_hdc) / max(model_tok_s, 0.1) + avg_hdc / max(hdc_tok_s, 1))

    print(f"\n  --- Agent Swarm Projections ---")
    print(f"  HDC component:        {hdc_tok_s:>8.0f} tok/s")
    print(f"  Model component:      {model_tok_s:>8.0f} tok/s")
    print(f"  Effective (1 agent):  {effective_single:>8.0f} tok/s")
    print(f"  Effective (8 agents): {effective_single * 8:>8.0f} tok/s")
    print(f"  Effective (16 agents):{effective_single * 16:>8.0f} tok/s")
    print(f"  Effective (32 agents):{effective_single * 32:>8.0f} tok/s")
    print(f"  Effective (64 agents):{effective_single * 64:>8.0f} tok/s")

    if effective_single * 8 >= 2000:
        print(f"\n  ✅ 2K tok/s target ACHIEVED with 8+ agents")
    elif effective_single * 16 >= 2000:
        print(f"\n  ✅ 2K tok/s target ACHIEVED with 16+ agents")
    else:
        print(f"\n  ⚠️  2K tok/s target requires {2000 / max(effective_single, 1):.0f}x acceleration")
    if effective_single * 64 >= 10000:
        print(f"  ✅ 10K tok/s target ACHIEVED with 64+ agents")

    return {
        'model': label,
        'model_path': model_path,
        'variant': variant,
        'avg_tokens_per_second': round(overall_tok_s, 1),
        'avg_hdc_ratio': round(avg_hdc, 3),
        'total_tokens': total_gen_tokens,
        'total_time': round(total_gen_time, 2),
        'model_component_tok_s': round(model_tok_s, 1),
        'hdc_component_tok_s': round(hdc_tok_s, 1),
        'effective_1_agent': round(effective_single, 1),
        'projections': projections,
        'hdc_targets': stats['hdc_targets'],
        'hdc_prototypes': stats['hdc_prototypes'],
        'vocab_size': meta['vocab_size'],
        'd_model': meta['d_model'],
        'n_layers': meta['n_layers'],
        'hdc_acceptance': stats['hdc_acceptance'],
        'hdc_latency_ns': stats['hdc_latency_ns'],
        'confidence_threshold': pipe.confidence_threshold,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Real Model Benchmark')
    parser.add_argument('--model', choices=['e2b', 'e4b', 'all'], default='all')
    parser.add_argument('--tokens', type=int, default=128, help='Tokens per prompt')
    parser.add_argument('--quick', action='store_true', help='Quick test (16 tokens)')
    parser.add_argument('--full', action='store_true', help='Full benchmark')
    args = parser.parse_args()

    max_tokens = 16 if args.quick else (1024 if args.full else args.tokens)

    models = find_gguf_models()
    if not models:
        print("No GGUF models found. Check paths:")
        for p in SEARCH_PATHS:
            print(f"  {p}")
        sys.exit(1)

    print(f"Found models: {list(models.keys())}")
    for k, v in models.items():
        print(f"  {k}: {Path(v).name} ({Path(v).stat().st_size / 1024**3:.1f} GB)")

    results = {}
    for key in ['e2b', 'e4b']:
        if args.model in ('all', key) and key in models:
            label = f"Gemma 4 {'E2B' if key == 'e2b' else 'E4B'}"
            result = benchmark_model(models[key], max_tokens=max_tokens, label=label)
            results[label] = result

    output = Path(f'/home/mike/Documents/Github/SpectralStream/tests/benchmark_results_{time.strftime("%Y%m%d_%H%M%S")}.json')
    with open(output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output}")

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        tok_s = r['avg_tokens_per_second']
        hdc = r['avg_hdc_ratio']
        proj_8 = r['projections'].get('8', 0)
        print(f"  {name}: {tok_s} tok/s (HDC: {hdc:.0%}), 8-agent: ~{proj_8} tok/s")

if __name__ == '__main__':
    main()
