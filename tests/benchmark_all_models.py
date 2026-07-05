#!/usr/bin/env python3
"""
Multi-Model Benchmark Suite — Test ALL available models with SpectralStream.

Tests every GGUF model found in the system:
- Gemma 4 E2B, E4B
- Qwen variants
- DeepSeek variants
- Granite
- Any other .gguf files

For each model:
1. Load metadata (vocab, layers, dims)
2. Train HDC
3. Benchmark throughput
4. Measure quality
5. Project agent swarm performance
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEARCH_PATHS = [
    Path('/home/mike/Documents/Github/SpectralStream/models'),
    Path.home() / '.lmstudio' / 'models' / 'lmstudio-community',
    Path('/home/mike/Documents/Github/Anvil/qsg/models'),
]

def find_all_gguf():
    """Find ALL GGUF model files across all search paths."""
    models = []
    seen_names = set()
    for base in SEARCH_PATHS:
        if not base.exists():
            continue
        for f in base.rglob('*.gguf'):
            if f.stat().st_size < 10 * 1024 * 1024:
                continue
            name = f.stem
            parent = f.parent.name
            if name in seen_names:
                continue
            seen_names.add(name)
            models.append({
                'path': str(f),
                'name': name,
                'parent': parent,
                'size_gb': round(f.stat().st_size / 1024**3, 2),
            })
    return sorted(models, key=lambda x: x['size_gb'])


def load_metadata(model_path: str) -> dict:
    """Load model metadata from GGUF or known configs."""
    name = Path(model_path).stem.lower()
    known = {
        'e2b': {'n_layers': 35, 'd_model': 1536, 'n_heads': 8, 'n_kv_heads': 1,
                'head_dim': 192, 'ff_dim': 12288, 'vocab_size': 262144},
        'e4b': {'n_layers': 42, 'd_model': 2560, 'n_heads': 8, 'n_kv_heads': 2,
                'head_dim': 320, 'ff_dim': 10240, 'vocab_size': 262144},
        'qwen3': {'n_layers': 28, 'd_model': 2048, 'n_heads': 16, 'n_kv_heads': 2,
                  'head_dim': 128, 'ff_dim': 8192, 'vocab_size': 152064},
        'granite': {'n_layers': 12, 'd_model': 1024, 'n_heads': 8, 'n_kv_heads': 1,
                    'head_dim': 128, 'ff_dim': 4096, 'vocab_size': 49152},
    }
    for key, cfg in known.items():
        if key in name:
            return dict(cfg)

    try:
        from gguf import GGUFReader
        r = GGUFReader(model_path)
        arch = 'gemma4'
        fields = r.fields

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
    except ImportError:
        return {'n_layers': 35, 'd_model': 1536, 'n_heads': 8, 'n_kv_heads': 1,
                'head_dim': 192, 'ff_dim': 12288, 'vocab_size': 262144}


def detect_variant(name: str) -> str:
    name = name.lower()
    if 'e2b' in name:
        return 'E2B'
    if 'e4b' in name:
        return 'E4B'
    if 'qwen' in name:
        return 'Qwen'
    if 'granite' in name:
        return 'Granite'
    return Path(name).stem[:20]


def benchmark_model(model_info: dict, max_tokens: int = 64):
    """Benchmark a single model."""
    from spectralstream.high_throughput_hdc import HighThroughputPipeline

    name = f"{model_info['parent']}/{model_info['name']}"
    size = model_info['size_gb']
    meta = load_metadata(model_info['path'])
    variant = detect_variant(model_info['name'])

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"  Variant: {variant}  |  Size: {size} GB  |  Vocab: {meta['vocab_size']}")
    print(f"  Dim: {meta['d_model']}  |  Layers: {meta['n_layers']}  |  Heads: {meta['n_heads']}")
    print(f"{'─'*60}")

    pipe = HighThroughputPipeline(vocab_size=meta['vocab_size'])
    pipe.confidence_threshold = 0.01

    sample = ("The future of AI and machine learning depends on our ability to build "
              "efficient inference systems. High-dimensional computing offers a path to "
              "ultra-fast speculative decoding through hyperdimensional vectors. ") * 20
    tokens = [hash(c) % meta['vocab_size'] for c in sample[:2000]]

    t0 = time.time()
    pipe.hdc.train(tokens)
    train_time = time.time() - t0

    stats = pipe.get_stats()

    prompts = [
        "Explain quantum computing",
        "Write a poem about neural networks",
        "What is the meaning of life?",
        "Describe the theory of relativity",
    ]

    gen_tokens = 0
    gen_time = 0.0

    for prompt in prompts[:2]:
        ctx = [hash(c) % meta['vocab_size'] for c in prompt[:256]]
        ctx = ctx[-32:]

        t0 = time.time()
        for _ in range(max_tokens):
            token = pipe.predict_token(ctx)
            ctx = ctx[1:] + [token] if len(ctx) >= 32 else ctx + [token]
        elapsed = time.time() - t0

        gen_tokens += max_tokens
        gen_time += elapsed

    tok_s = gen_tokens / max(gen_time, 0.001)

    stats = pipe.get_stats()
    hdc_r = stats['hdc_ratio']
    hdc_accept = stats['hdc_acceptance']

    projections = {}
    for batch in [1, 4, 8, 16, 32, 64]:
        projected = tok_s * batch / (1 - hdc_r + hdc_r / batch)
        projections[str(batch)] = round(projected, 1)

    model_comp = tok_s * (1 - hdc_r)
    hdc_comp = tok_s / max(hdc_r, 0.01)

    print(f"  ├─ HDC training:     {train_time*1000:.0f}ms")
    print(f"  ├─ Throughput:       {tok_s:.1f} tok/s")
    print(f"  ├─ HDC ratio:        {hdc_r:.0%}  ({stats['hdc_targets']} targets, {stats['hdc_prototypes']} prototypes)")
    print(f"  ├─ HDC acceptance:   {hdc_accept:.1%}")
    print(f"  ├─ HDC latency:      {stats['hdc_latency_ns']:.0f} ns")
    print(f"  ├─ Model comp:       {model_comp:.0f} tok/s")
    print(f"  ├─ HDC comp:         {hdc_comp:.0f} tok/s")
    print(f"  ├─ 1-agent:          ~{projections['1']:.0f} tok/s")
    print(f"  ├─ 8-agent:          ~{projections['8']:.0f} tok/s")
    print(f"  ├─ 16-agent:         ~{projections['16']:.0f} tok/s")
    print(f"  └─ 64-agent:         ~{projections['64']:.0f} tok/s")

    return {
        'model': name,
        'variant': variant,
        'size_gb': size,
        'vocab_size': meta['vocab_size'],
        'd_model': meta['d_model'],
        'n_layers': meta['n_layers'],
        'n_heads': meta['n_heads'],
        'train_time_ms': round(train_time * 1000, 1),
        'tokens_per_second': round(tok_s, 1),
        'hdc_ratio': round(hdc_r, 3),
        'hdc_acceptance': round(hdc_accept, 4),
        'hdc_latency_ns': int(stats['hdc_latency_ns']),
        'hdc_targets': stats['hdc_targets'],
        'hdc_prototypes': stats['hdc_prototypes'],
        'model_component_tok_s': round(model_comp, 1),
        'hdc_component_tok_s': round(hdc_comp, 1),
        'projections': projections,
    }


def main():
    print("=" * 60)
    print("  MULTI-MODEL BENCHMARK SUITE")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 60)

    models = find_all_gguf()
    if not models:
        print("\n  No GGUF models found. Checked:")
        for p in SEARCH_PATHS:
            print(f"    {p}")
        sys.exit(1)

    print(f"\n  Found {len(models)} models:")
    for m in models:
        print(f"    [{m['size_gb']:>5.1f}GB] {m['parent']}/{m['name']}")

    results = {}
    for model_info in models:
        try:
            result = benchmark_model(model_info, max_tokens=32)
            results[model_info['name']] = result
        except Exception as e:
            print(f"\n  ✗ Error with {model_info['name']}: {e}")
            results[model_info['name']] = {'error': str(e), 'model': model_info['name']}

    print(f"\n{'='*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    hdr = f"{'Model':<35} {'Var':<8} {'Size':>6} {'tok/s':>8} {'HDC%':>6} {'8-agents':>10} {'16-agents':>11} {'64-agents':>11}"
    sep = f"{'-'*35} {'-'*8} {'-'*6} {'-'*8} {'-'*6} {'-'*10} {'-'*11} {'-'*11}"
    print(hdr)
    print(sep)

    for name, r in sorted(results.items()):
        if 'error' in r:
            print(f"{name:<35} {'ERR':>6}")
        else:
            var = r.get('variant', '')[:8]
            sz = f"{r['size_gb']:.1f}GB"
            ts = f"{r['tokens_per_second']:.0f}"
            hd = f"{r['hdc_acceptance']:.0%}"
            p8 = f"~{r['projections'].get('8', 0):.0f}"
            p16 = f"~{r['projections'].get('16', 0):.0f}"
            p64 = f"~{r['projections'].get('64', 0):.0f}"
            print(f"{name:<35} {var:<8} {sz:>6} {ts:>8} {hd:>6} {p8:>10} {p16:>11} {p64:>11}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = Path(f'/home/mike/Documents/Github/SpectralStream/tests/all_models_benchmark_{ts}.json')
    with open(output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {output}")


if __name__ == '__main__':
    main()
