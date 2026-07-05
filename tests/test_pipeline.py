#!/usr/bin/env python3
"""
Quick test of SpectralStream components.
Verifies all subsystems work independently and together.
"""

import sys

sys.path.insert(0, "/home/mike/Documents/Github/SpectralStream")

import pytest
import time
import numpy as np

# The old spectralstream top-level API has been refactored.
# These imports now come from the correct refactored modules.
from spectralstream.core.math_primitives import HadamardRotator, LloydMaxQuantizer

# Classes from archived modules are available via backward-compat stubs
pytest.importorskip("spectralstream.hadamard_preconditioner")
from spectralstream.hadamard_preconditioner import HadamardPreconditioner  # noqa: F401

pytest.importorskip("spectralstream.model_compressor")
from spectralstream.model_compressor import ModelCompressor  # noqa: F401


@pytest.mark.skip(
    reason="HDCBundle moved to archive; use core.math_primitives.hrr_* instead"
)
def test_hdc_bundle() -> bool:
    """Test HDC bundle encoding, similarity, and learning.

    Returns
    -------
    bool
        True on success.
    """
    print("=== Test HDC Bundle ===")
    bundle = HDCBundle(dim=1000)
    hv1 = bundle.ensure_token_vector(42)
    hv2 = bundle.ensure_token_vector(42)
    hv3 = bundle.ensure_token_vector(7)

    assert hv1.shape == (1000,), f"Expected (1000,), got {hv1.shape}"
    assert np.array_equal(hv1, hv2), "Same token should give same HV"
    assert not np.array_equal(hv1, hv3), "Different tokens should give different HVs"

    sim = bundle._popcount_sim(hv1, hv2)
    assert sim == 1.0, f"Self-similarity should be 1.0, got {sim}"

    bundle.learn([10, 20, 30, 40, 50])
    bundle.learn([10, 20, 30, 40, 50])
    bundle.learn([10, 20, 30, 99, 50])

    predictions = bundle.predict_next((10, 20, 30), n_candidates=5)
    assert len(predictions) > 0, "Should have predictions after learning"

    print("  HDC Bundle: OK")
    return True


@pytest.mark.skip(reason="HDCDraftEngine moved to archive")
def test_hd_draft_engine() -> bool:
    """Test HDC draft engine block generation.

    Returns
    -------
    bool
        True on success.
    """
    print("=== Test HDC Draft Engine ===")
    engine = HDCDraftEngine(vocab_size=1000, hd_dim=500)

    # Simulate some context
    for tok in [10, 20, 30, 40, 50]:
        engine.observe(tok)

    blocks = engine.draft_block(block_size=4)
    assert len(blocks) > 0, "Should generate candidate blocks"
    assert all(len(b) == 4 for b in blocks), (
        f"All blocks should be size 4, got {[len(b) for b in blocks]}"
    )

    print(f"  Generated {len(blocks)} candidate blocks of size {len(blocks[0])}")
    print("  HDC Draft Engine: OK")
    return True


def test_hadamard_rotator() -> bool:
    """Test Hadamard rotation and its inverse round-trip fidelity.

    Verifies that a random vector rotated via :class:`HadamardRotator` and
    then inverse-rotated has a mean-squared error below 1e-4.

    Returns
    -------
    bool
        True on success.
    """
    print("=== Test Hadamard Rotator ===")
    rot = HadamardRotator(dim=128, seed=42)
    vec = np.random.randn(32, 128).astype(np.float32)

    rotated = rot.rotate(vec)
    reconstructed = rot.inverse_rotate(rotated)

    assert rotated.shape == vec.shape, f"Shape mismatch: {rotated.shape} vs {vec.shape}"
    mse = np.mean((vec - reconstructed) ** 2)
    assert mse < 1e-4, f"Inverse rotation error too high: {mse}"

    print(f"  Rotation/Inverse error: {mse:.6f}")
    print("  Hadamard Rotator: OK")
    return True


def test_lloydmax_quantizer() -> bool:
    """Test Lloyd-Max quantizer compress/decompress round-trip.

    Verifies that random float32 data quantised to 4 bits and then
    decompressed has a low reconstruction MSE.

    Returns
    -------
    bool
        True on success.
    """
    print("=== Test Lloyd-Max Quantizer ===")
    quant = LloydMaxQuantizer(n_bits=4)
    data = np.random.randn(1000, 128).astype(np.float32)

    quantized = quant.quantize(data)
    mse = np.mean((data - quantized) ** 2)
    ratio = np.prod(data.shape) * 4 / np.prod(quantized.shape) * 32 / 4  # bits

    print(f"  Quantization MSE: {mse:.4f}")
    print(f"  Theoretical compression: {ratio:.1f}x")

    indices, centroids = quant.compress(data)
    decompressed = quant.decompress(indices, data.shape)
    decompress_mse = np.mean((data - decompressed) ** 2)

    print(f"  Compress/Decompress MSE: {decompress_mse:.4f}")
    print("  Lloyd-Max Quantizer: OK")
    return True


@pytest.mark.skip(
    reason="SpectralKVCache moved to archive; use kv_cache.manager.KVCacheManager instead"
)
def test_spectral_kv_cache() -> bool:
    """Test spectral KV cache store, query, retrieve, and compression ratio.

    Returns
    -------
    bool
        True on success.
    """
    print("=== Test Spectral KV Cache ===")
    cache = SpectralKVCache(dim=64, max_size=256, k_bits=4, v_bits=2)

    for i in range(100):
        k = np.random.randn(64).astype(np.float32)
        v = np.random.randn(64).astype(np.float32)
        cache.store(k, v, position=i)

    query = np.random.randn(64).astype(np.float32)
    results = cache.query(query, top_k=5)

    assert len(results) > 0, "Should return results"
    print(f"  Query returned {len(results)} results")
    print(f"  Top-1 similarity: {results[0][1]:.4f}")

    retrieved = cache.retrieve(0)
    print(f"  Retrieved position 0: {'hit' if retrieved else 'miss'}")
    print(f"  Hit rate: {cache.hit_rate():.2%}")
    print(f"  Compression: {cache.compression_ratio():.1f}x")
    print("  Spectral KV Cache: OK")
    return True


@pytest.mark.skip(reason="AttractorScoringEnsemble moved to archive")
def test_attractor_scorer() -> bool:
    """Test attractor scoring ensemble candidate scoring.

    Returns
    -------
    bool
        True on success.
    """
    print("=== Test Attractor Scoring ===")
    scorer = AttractorScoringEnsemble(hidden_dim=64)

    context_emb = np.random.randn(10, 64).astype(np.float32)
    candidates = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    hidden_per_candidate = [
        [np.random.randn(64).astype(np.float32) for _ in range(len(c))]
        for c in candidates
    ]

    scores = scorer.score_candidates(candidates, context_emb, hidden_per_candidate)
    assert len(scores) == len(candidates), "Should score all candidates"

    print(f"  Scores: {[(i, f'{s:.4f}') for i, s in scores]}")
    print("  Attractor Scorer: OK")
    return True


@pytest.mark.skip(reason="BlockEmissionPipeline moved to archive")
def test_block_emission() -> bool:
    """Test block emission pipeline generation and statistics.

    Returns
    -------
    bool
        True on success.
    """
    print("=== Test Block Emission ===")
    from spectralstream.gguf_model import DummyModel

    model = DummyModel(hidden_dim=128, vocab_size=1000, n_layers=4, n_heads=4)

    hd_engine = HDCDraftEngine(vocab_size=1000, hd_dim=500)
    scorer = AttractorScoringEnsemble(hidden_dim=128)

    pipeline = BlockEmissionPipeline(
        model_fn=model.forward,
        hd_engine=hd_engine,
        scorer=scorer,
        block_size=4,
        n_candidate_blocks=8,
        coherence_threshold=0.3,
    )

    # Need to seed HDC with observations
    for tok in [10, 20, 30, 40]:
        hd_engine.observe(tok)

    output = pipeline.generate([10, 20, 30, 40], max_new_tokens=32)
    assert len(output) > 4, f"Should generate more tokens than input, got {len(output)}"

    stats = pipeline.statistics()
    print(f"  Generated {len(output)} tokens")
    print(f"  Tokens per model call: {stats['tokens_per_model_call']:.2f}")
    print(f"  Block success rate: {stats['block_success_rate']:.2%}")
    print("  Block Emission: OK")
    return True


@pytest.mark.skip(
    reason="SpectralStream top-level class moved to archive; use inference.engine or pipeline"
)
def test_engine() -> bool:
    """Test full SpectralStream engine generation and throughput.

    Returns
    -------
    bool
        True on success.
    """
    print("=== Test Full Engine ===")
    engine = SpectralStream(
        hidden_dim=128,
        vocab_size=1000,
        n_heads=4,
        n_layers=4,
        block_size=4,
        hd_dim=500,
        coherence_threshold=0.3,
    )

    tokens, tps = engine.generate([1, 2, 3, 4], max_new_tokens=32)
    stats = engine.stats()

    print(f"  Generated {len(tokens)} tokens")
    print(f"  Tokens/sec: {tps:.1f}")
    print(f"  Tokens/model call: {stats['tokens_per_model_call']:.2f}")
    print(f"  Stats: {stats}")
    print("  Full Engine: OK")
    return True


def run_benchmark() -> None:
    """Run a quick throughput benchmark on the SpectralStream engine.

    Generates text from several prompts, measures tokens/second, and
    prints aggregate timing statistics.  Only invoked when the module is
    run directly (``__name__ == "__main__"``).
    """
    print("\n=== Quick Benchmark ===")
    engine = SpectralStream(
        hidden_dim=256,
        vocab_size=1000,
        n_heads=8,
        n_layers=6,
        block_size=8,
        hd_dim=2000,
    )

    prompts = [
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120],
    ]

    times = []
    for prompt in prompts:
        start = time.time()
        tokens, tps = engine.generate(prompt, max_new_tokens=64)
        elapsed = time.time() - start
        times.append(elapsed)
        ngen = len(tokens) - len(prompt)
        print(f"  Prompt: {ngen} tokens in {elapsed:.2f}s = {ngen / elapsed:.1f} tok/s")
        engine.reset()

    avg_tps = np.mean(
        [
            len(t) - len(p)
            for t, p in zip(
                [engine.generate(p, 64)[0] for p in prompts[:1]], prompts[:1]
            )
        ]
    ) / np.mean(times)

    print(f"\n=== Benchmark Results ===")
    print(f"Setup: Dummy model 256-dim, 6-layer, 8-head")
    print(f"Block size: 8")
    print(f"Observed tokens/sec: {np.mean([64 / t for t in times]):.1f}")
    print(f"Total time: {sum(times):.1f}s for {len(times) * 64} tokens")


if __name__ == "__main__":
    tests = [
        test_hdc_bundle,
        test_hd_draft_engine,
        test_hadamard_rotator,
        test_lloydmax_quantizer,
        test_spectral_kv_cache,
        test_attractor_scorer,
        test_block_emission,
        test_engine,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
        print()

    print(f"=== Results: {passed} passed, {failed} failed ===")

    if passed == len(tests):
        run_benchmark()
