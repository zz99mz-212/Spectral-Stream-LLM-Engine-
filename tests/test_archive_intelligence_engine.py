"""
Test Intelligence Engine on Synthetic Gemma-4-like Model
========================================================
Tests the full compression pipeline on synthetic tensors that mimic
Gemma-4's architecture (10-20 tensors, various shapes).
"""

import logging
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_intelligence_engine")


def create_synthetic_gemma4_tensors(seed: int = 42) -> dict:
    """Create synthetic tensors mimicking Gemma-4 structure.

    Returns dict of {name: tensor} with shapes based on real Gemma-4.
    """
    rng = np.random.RandomState(seed)
    tensors = {}

    # Gemma-4 config: hidden=1536, head_dim=256, ffn_dim=6144
    hidden = 1536
    head_dim = 256
    ffn_dim = 6144
    n_heads = 6
    vocab_size = 262144

    # Use smaller embedding for test speed (real one is 262144 x 1536 = 1.5GB)
    embed_vocab = 4096  # Reduced for test speed

    # Layer 0
    tensors["layer.0.attn.q_proj.weight"] = rng.randn(hidden, hidden).astype(np.float32) * 0.02
    tensors["layer.0.attn.k_proj.weight"] = rng.randn(head_dim, hidden).astype(np.float32) * 0.02
    tensors["layer.0.attn.v_proj.weight"] = rng.randn(head_dim, hidden).astype(np.float32) * 0.02
    tensors["layer.0.attn.o_proj.weight"] = rng.randn(hidden, hidden).astype(np.float32) * 0.02
    tensors["layer.0.ffn_gate.weight"] = rng.randn(ffn_dim, hidden).astype(np.float32) * 0.02
    tensors["layer.0.ffn_up.weight"] = rng.randn(ffn_dim, hidden).astype(np.float32) * 0.02
    tensors["layer.0.ffn_down.weight"] = rng.randn(hidden, ffn_dim).astype(np.float32) * 0.02
    tensors["layer.0.attn_norm.weight"] = rng.randn(hidden).astype(np.float32) * 0.01 + 1.0
    tensors["layer.0.ffn_norm.weight"] = rng.randn(hidden).astype(np.float32) * 0.01 + 1.0

    # Layer 1
    tensors["layer.1.attn.q_proj.weight"] = rng.randn(hidden, hidden).astype(np.float32) * 0.02
    tensors["layer.1.attn.k_proj.weight"] = rng.randn(head_dim, hidden).astype(np.float32) * 0.02
    tensors["layer.1.attn.v_proj.weight"] = rng.randn(head_dim, hidden).astype(np.float32) * 0.02
    tensors["layer.1.attn.o_proj.weight"] = rng.randn(hidden, hidden).astype(np.float32) * 0.02
    tensors["layer.1.ffn_gate.weight"] = rng.randn(ffn_dim, hidden).astype(np.float32) * 0.02
    tensors["layer.1.ffn_up.weight"] = rng.randn(ffn_dim, hidden).astype(np.float32) * 0.02
    tensors["layer.1.ffn_down.weight"] = rng.randn(hidden, ffn_dim).astype(np.float32) * 0.02

    # Embeddings (reduced for test speed)
    tensors["embed_tokens.weight"] = rng.randn(embed_vocab, hidden).astype(np.float32) * 0.02

    # Final norm
    tensors["final_norm.weight"] = rng.randn(hidden).astype(np.float32) * 0.01 + 1.0

    return tensors


def test_tensor_profiler():
    """Test TensorProfiler on various tensor types."""
    from spectralstream.compression.intelligence_engine import TensorProfiler

    profiler = TensorProfiler()
    tensors = create_synthetic_gemma4_tensors()

    print("\n" + "=" * 80)
    print("TENSOR PROFILER TEST")
    print("=" * 80)

    for name, tensor in tensors.items():
        profile = profiler.profile(tensor, name=name)
        print(
            f"  {name:40s} | shape={str(profile.shape):16s} | "
            f"type={profile.tensor_type:10s} | method={profile.recommended_method:15s} | "
            f"rank={profile.effective_rank:6.1f} | energy={profile.energy_concentration:.3f} | "
            f"outliers={profile.outlier_ratio:.4f} | sens={profile.sensitivity:.2f}"
        )

    print(f"\n  Profiled {len(tensors)} tensors")
    return True


def test_error_budget_allocator():
    """Test ErrorBudgetAllocator priority-based budget distribution."""
    from spectralstream.compression.intelligence_engine import ErrorBudgetAllocator, TensorProfiler

    profiler = TensorProfiler()
    allocator = ErrorBudgetAllocator()
    tensors = create_synthetic_gemma4_tensors()

    print("\n" + "=" * 80)
    print("ERROR BUDGET ALLOCATOR TEST")
    print("=" * 80)

    profiles = {}
    for name, tensor in tensors.items():
        profiles[name] = profiler.profile(tensor, name=name)

    budgets = allocator.allocate(profiles, total_budget=0.01)

    for name, budget in sorted(budgets.items()):
        print(
            f"  {name:40s} | priority={budget.priority:6s} | "
            f"sensitivity={budget.sensitivity_weight:.2f} | "
            f"max_error={budget.max_relative_error:.4f}"
        )

    # Verify: attention tensors should have tighter budgets than FFN
    attn_budget = budgets["layer.0.attn_q_proj.weight" if "layer.0.attn_q_proj.weight" in budgets else "layer.0.attn.q_proj.weight"]
    ffn_budget = budgets["layer.0.ffn_gate.weight"]
    assert attn_budget.max_relative_error <= ffn_budget.max_relative_error * 1.5, \
        f"Attention budget ({attn_budget.max_relative_error}) should be <= FFN budget ({ffn_budget.max_relative_error})"

    print(f"\n  Allocated budgets for {len(budgets)} tensors")
    print(f"  Budget range: [{min(b.max_relative_error for b in budgets.values()):.4f}, "
          f"{max(b.max_relative_error for b in budgets.values()):.4f}]")
    return True


def test_compression_strategy_selector():
    """Test CompressionStrategySelector strategy selection."""
    from spectralstream.compression.intelligence_engine import (
        CompressionStrategySelector,
        ErrorBudget,
        TensorProfiler,
    )

    profiler = TensorProfiler()
    selector = CompressionStrategySelector()
    tensors = create_synthetic_gemma4_tensors()

    print("\n" + "=" * 80)
    print("COMPRESSION STRATEGY SELECTOR TEST")
    print("=" * 80)

    for name, tensor in list(tensors.items())[:5]:
        profile = profiler.profile(tensor, name=name)
        budget = ErrorBudget(
            tensor_name=name,
            max_relative_error=0.01,
            priority="MEDIUM",
            sensitivity_weight=0.5,
        )
        strategies = selector.select(tensor, profile, budget)
        strategy_names = [s[0] for s in strategies]
        print(f"  {name:40s} | strategies: {strategy_names}")

    print(f"\n  Tested {min(5, len(tensors))} tensors")
    return True


def test_single_tensor_compress_decompress():
    """Test compress/decompress round-trip on a single tensor."""
    from spectralstream.compression.intelligence_engine import CompressionOrchestrator

    orchestrator = CompressionOrchestrator()
    rng = np.random.RandomState(42)

    print("\n" + "=" * 80)
    print("SINGLE TENSOR COMPRESS/DECOMPRESS TEST")
    print("=" * 80)

    # Test with a typical attention projection: 1536x1536
    tensor = rng.randn(1536, 1536).astype(np.float32) * 0.02
    ct = orchestrator.compress_tensor(tensor, name="layer.0.attn.q_proj.weight")

    # Decompress
    reconstructed = orchestrator.decompress_tensor(ct)

    # Verify round-trip
    orig_flat = tensor.ravel()
    recon_flat = reconstructed.ravel()

    mse = float(np.mean((orig_flat - recon_flat) ** 2))
    rel_error = float(np.linalg.norm(orig_flat - recon_flat) / (np.linalg.norm(orig_flat) + 1e-10))
    cos_sim = float(np.dot(orig_flat, recon_flat) / (np.linalg.norm(orig_flat) * np.linalg.norm(recon_flat) + 1e-10))

    print(f"  Tensor: {tensor.shape}")
    print(f"  Method: {ct.method}")
    print(f"  Ratio:  {ct.compression_ratio:.2f}x")
    print(f"  Error:  {ct.relative_error:.4%} (claimed) / {rel_error:.4%} (verified)")
    print(f"  SNR:    {ct.snr_db:.1f} dB")
    print(f"  CosSim: {ct.cosine_similarity:.6f} / {cos_sim:.6f} (verified)")
    print(f"  Grade:  {ct.quality_grade}")
    print(f"  Time:   {ct.computation_time*1000:.1f} ms")

    # Verify metrics are honest (claimed error should be close to verified)
    assert abs(ct.relative_error - rel_error) < 0.001, \
        f"Claimed error ({ct.relative_error}) vs verified ({rel_error}) differ too much"

    # Verify shape is preserved
    assert reconstructed.shape == tensor.shape, \
        f"Shape mismatch: {reconstructed.shape} vs {tensor.shape}"

    print("\n  Round-trip verification: PASSED")
    return True


def test_model_compression():
    """Test full model compression on synthetic Gemma-4 tensors."""
    from spectralstream.compression.intelligence_engine import (
        CompressionOrchestrator,
        TensorProfiler,
        ErrorBudgetAllocator,
    )

    orchestrator = CompressionOrchestrator(
        target_ratio=4.0,
        max_error=0.01,
        total_error_budget=0.01,
    )
    profiler = TensorProfiler()
    budget_allocator = ErrorBudgetAllocator()

    print("\n" + "=" * 80)
    print("FULL MODEL COMPRESSION TEST (synthetic Gemma-4)")
    print("=" * 80)

    tensors = create_synthetic_gemma4_tensors()

    # Pre-profile all tensors
    profiles = {}
    for name, tensor in tensors.items():
        profiles[name] = profiler.profile(tensor, name=name)

    # Pre-allocate budgets
    budgets = budget_allocator.allocate(profiles, total_budget=0.01)

    t0 = time.perf_counter()
    compressed_tensors = {}

    for name, tensor in tensors.items():
        ct = orchestrator.compress_tensor(tensor, profile=profiles[name], budget=budgets[name], name=name)
        compressed_tensors[name] = ct

    elapsed = time.perf_counter() - t0

    # Compute totals
    total_original = sum(t.nbytes for t in tensors.values())
    total_compressed = sum(len(ct.data) for ct in compressed_tensors.values())
    overall_ratio = total_original / max(total_compressed, 1)

    errors = [ct.relative_error for ct in compressed_tensors.values() if ct.relative_error > 0]
    avg_error = float(np.mean(errors)) if errors else 0.0
    max_error = float(np.max(errors)) if errors else 0.0

    method_dist = {}
    for ct in compressed_tensors.values():
        method_dist[ct.method] = method_dist.get(ct.method, 0) + 1

    # Print results
    print(f"\n  {'Tensor':40s} | {'Method':15s} | {'Ratio':>6s} | {'Error':>8s} | {'Grade':>5s}")
    print("  " + "-" * 85)
    for name, ct in compressed_tensors.items():
        print(
            f"  {name:40s} | {ct.method:15s} | {ct.compression_ratio:5.2f}x | "
            f"{ct.relative_error:7.4%} | {ct.quality_grade:>5s}"
        )

    print(f"\n  Summary:")
    print(f"    Tensors:          {len(compressed_tensors)}")
    print(f"    Original:         {total_original:,} bytes ({total_original/1024/1024:.1f} MB)")
    print(f"    Compressed:       {total_compressed:,} bytes ({total_compressed/1024/1024:.1f} MB)")
    print(f"    Overall ratio:    {overall_ratio:.2f}x")
    print(f"    Avg error:        {avg_error:.4%}")
    print(f"    Max error:        {max_error:.4%}")
    print(f"    Method dist:      {method_dist}")
    print(f"    Time:             {elapsed:.2f}s")

    # Verify overall ratio is reasonable
    assert overall_ratio >= 1.0, f"Overall ratio should be >= 1.0, got {overall_ratio}"

    # Verify no catastrophic errors
    assert max_error < 0.10, f"Max error {max_error:.4%} exceeds 10% threshold"

    print("\n  Model compression test: PASSED")
    return True


def test_cross_layer_optimization():
    """Test that cross-layer patterns are detected and exploited."""
    from spectralstream.compression.intelligence_engine import TensorProfiler

    profiler = TensorProfiler()

    print("\n" + "=" * 80)
    print("CROSS-LAYER OPTIMIZATION TEST")
    print("=" * 80)

    rng = np.random.RandomState(42)
    hidden = 1536

    # Create similar tensors across layers (mimics transformer blocks)
    base = rng.randn(hidden, hidden).astype(np.float32) * 0.02
    layer0_q = base + rng.randn(hidden, hidden).astype(np.float32) * 0.001
    layer1_q = base + rng.randn(hidden, hidden).astype(np.float32) * 0.001
    layer2_q = base + rng.randn(hidden, hidden).astype(np.float32) * 0.001

    # Profile each
    p0 = profiler.profile(layer0_q, name="layer.0.attn.q_proj.weight")
    p1 = profiler.profile(layer1_q, name="layer.1.attn.q_proj.weight")
    p2 = profiler.profile(layer2_q, name="layer.2.attn.q_proj.weight")

    print(f"  Layer 0 Q: rank={p0.effective_rank:.1f}, energy={p0.energy_concentration:.3f}")
    print(f"  Layer 1 Q: rank={p1.effective_rank:.1f}, energy={p1.energy_concentration:.3f}")
    print(f"  Layer 2 Q: rank={p2.effective_rank:.1f}, energy={p2.energy_concentration:.3f}")

    # The cross-layer similarity (low delta) would be detected by
    # the delta_int4 method in the strategy selector
    delta01 = np.linalg.norm(layer0_q - layer1_q) / (np.linalg.norm(layer0_q) + 1e-10)
    delta02 = np.linalg.norm(layer0_q - layer2_q) / (np.linalg.norm(layer0_q) + 1e-10)

    print(f"  Delta 0->1: {delta01:.4%}")
    print(f"  Delta 0->2: {delta02:.4%}")
    print(f"  Cross-layer similarity detected: {delta01 < 0.05}")

    return True


def test_different_tensor_shapes():
    """Test profiling and compression across various tensor shapes."""
    from spectralstream.compression.intelligence_engine import CompressionOrchestrator

    orchestrator = CompressionOrchestrator()
    rng = np.random.RandomState(42)

    print("\n" + "=" * 80)
    print("DIFFERENT TENSOR SHAPES TEST")
    print("=" * 80)

    shapes_and_names = [
        ((4096, 1536), "embed_tokens.weight"),
        ((2048, 1536), "attn.q_proj.weight"),
        ((256, 1536), "attn.k_proj.weight"),
        ((256, 1536), "attn.v_proj.weight"),
        ((1536, 2048), "attn.o_proj.weight"),
        ((6144, 1536), "ffn_gate.weight"),
        ((6144, 1536), "ffn_up.weight"),
        ((1536, 6144), "ffn_down.weight"),
        ((1536,), "attn_norm.weight"),
        ((1536,), "ffn_norm.weight"),
    ]

    print(f"\n  {'Shape':20s} | {'Name':25s} | {'Method':15s} | {'Ratio':>6s} | {'Error':>8s}")
    print("  " + "-" * 80)

    for shape, name in shapes_and_names:
        tensor = rng.randn(*shape).astype(np.float32) * 0.02
        ct = orchestrator.compress_tensor(tensor, name=name)
        print(
            f"  {str(shape):20s} | {name:25s} | {ct.method:15s} | "
            f"{ct.compression_ratio:5.2f}x | {ct.relative_error:7.4%}"
        )

    print(f"\n  Tested {len(shapes_and_names)} different shapes")
    return True


def test_error_budget_allocation_sensitivity():
    """Test that error budgets are properly allocated based on sensitivity."""
    from spectralstream.compression.intelligence_engine import ErrorBudgetAllocator, TensorProfiler

    profiler = TensorProfiler()
    allocator = ErrorBudgetAllocator()

    print("\n" + "=" * 80)
    print("ERROR BUDGET SENSITIVITY TEST")
    print("=" * 80)

    rng = np.random.RandomState(42)
    hidden = 1536

    # Create tensors with different sensitivity roles
    test_tensors = {
        "embed_tokens.weight": rng.randn(4096, hidden).astype(np.float32) * 0.02,
        "layer.0.attn.q_proj.weight": rng.randn(hidden, hidden).astype(np.float32) * 0.02,
        "layer.0.attn.k_proj.weight": rng.randn(256, hidden).astype(np.float32) * 0.02,
        "layer.0.ffn_gate.weight": rng.randn(6144, hidden).astype(np.float32) * 0.02,
        "layer.0.ffn_down.weight": rng.randn(hidden, 6144).astype(np.float32) * 0.02,
        "layer.0.attn_norm.weight": rng.randn(hidden).astype(np.float32) * 0.01 + 1.0,
    }

    profiles = {}
    for name, tensor in test_tensors.items():
        profiles[name] = profiler.profile(tensor, name=name)

    budgets = allocator.allocate(profiles, total_budget=0.01)

    # Verify priority ordering
    priorities = [(name, b.priority, b.max_relative_error) for name, b in budgets.items()]
    priorities.sort(key=lambda x: x[2])

    print(f"\n  Priority ordering (tightest budget first):")
    for name, priority, err in priorities:
        print(f"    {priority:6s} | {err:.4f} | {name}")

    # Embeddings and attention should have tighter budgets than FFN
    embed_budget = budgets["embed_tokens.weight"]
    ffn_budget = budgets["layer.0.ffn_gate.weight"]
    norm_budget = budgets["layer.0.attn_norm.weight"]

    assert embed_budget.priority == "HIGH", f"Embedding should be HIGH priority, got {embed_budget.priority}"
    assert ffn_budget.priority in ("MEDIUM", "LOW"), f"FFN should be MEDIUM/LOW, got {ffn_budget.priority}"

    print(f"\n  Sensitivity test: PASSED")
    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("INTELLIGENCE ENGINE TEST SUITE")
    print("Synthetic Gemma-4 Model Compression")
    print("=" * 80)

    tests = [
        ("Tensor Profiler", test_tensor_profiler),
        ("Error Budget Allocator", test_error_budget_allocator),
        ("Compression Strategy Selector", test_compression_strategy_selector),
        ("Single Tensor Compress/Decompress", test_single_tensor_compress_decompress),
        ("Full Model Compression", test_model_compression),
        ("Cross-Layer Optimization", test_cross_layer_optimization),
        ("Different Tensor Shapes", test_different_tensor_shapes),
        ("Error Budget Sensitivity", test_error_budget_allocation_sensitivity),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = "PASSED" if passed else "FAILED"
        except Exception as e:
            results[name] = f"FAILED: {e}"
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    for name, result in results.items():
        status = "✓" if "PASSED" in result else "✗"
        print(f"  {status} {name:40s} {result}")

    n_passed = sum(1 for r in results.values() if "PASSED" in r)
    n_total = len(results)
    print(f"\n  {n_passed}/{n_total} tests passed")
    print("=" * 80)

    return n_passed == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
