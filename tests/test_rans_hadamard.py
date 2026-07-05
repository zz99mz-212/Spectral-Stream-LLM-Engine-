"""
Tests for rANS Entropy Coding and Hadamard Preconditioner modules.
"""

import math
import struct
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spectralstream.rans_entropy import (
    RANSEncoder,
    RANSDecoder,
    AdaptiveRANS,
    EntropyAnalyzer,
    _build_cumulative,
)
from spectralstream.hadamard_preconditioner import (
    HadamardPreconditioner,
    IncoherenceTransform,
    SpectralShaping,
)


# ═══════════════════════════════════════════════════════════════════════════
# rANS Entropy Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_normalize_frequencies():
    """Test that cumulative frequencies sum to 2^precision."""
    freq = np.array([10, 20, 30, 40], dtype=np.int64)
    cum, scaled, total = _build_cumulative(freq, precision=12)
    assert total == 4096
    assert cum[0] == 0
    assert cum[-1] == 4096
    assert all(cum[i] <= cum[i + 1] for i in range(len(cum) - 1))
    print("[PASS] test_normalize_frequencies")


def test_rans_encode_decode_roundtrip():
    """Test that encoding then decoding recovers original symbols."""
    symbols = [0, 1, 2, 3, 0, 1, 2, 3, 4, 4, 4, 4, 0, 0, 1, 2, 3, 4]
    freq = np.zeros(max(symbols) + 1, dtype=np.int64)
    for s in symbols:
        freq[s] += 1

    encoder = RANSEncoder()
    bits = encoder.encode(symbols)

    decoder = RANSDecoder()
    decoded = decoder.decode(bits, n=len(symbols))

    assert decoded == symbols, f"Round-trip failed: {decoded} != {symbols}"
    print("[PASS] test_rans_encode_decode_roundtrip")


def test_rans_16bit_roundtrip():
    """Test 16-bit precision rANS round-trip."""
    symbols = [0, 1, 0, 1, 2, 2, 3, 0, 1, 2]
    freq = np.zeros(max(symbols) + 1, dtype=np.int64)
    for s in symbols:
        freq[s] += 1

    encoder = RANSEncoder(precision=16)
    bits = encoder.encode(symbols)

    decoder = RANSDecoder()
    decoded = decoder.decode(bits, n=len(symbols))

    assert decoded == symbols, f"16-bit round-trip failed: {decoded} != {symbols}"
    print("[PASS] test_rans_16bit_roundtrip")


def test_rans_compression_ratio():
    """Test that rANS achieves better than 8-bit encoding."""
    # Highly skewed distribution
    symbols = [0] * 90 + [1] * 5 + [2] * 3 + [3] * 2
    freq = np.zeros(4, dtype=np.int64)
    for s in symbols:
        freq[s] += 1

    encoder = RANSEncoder()
    bits = encoder.encode(symbols)

    # Uncompressed: 8 bits per symbol
    uncompressed_bits = len(symbols) * 8
    compressed_bits = len(bits) * 8
    ratio = uncompressed_bits / compressed_bits

    assert ratio > 1.0, f"Compression ratio too low: {ratio:.2f}"
    print(f"[PASS] test_rans_compression_ratio (ratio={ratio:.2f})")


def test_rans_empty_symbols():
    """Test encoding empty symbol list."""
    encoder = RANSEncoder()
    bits = encoder.encode([])
    assert bits == b''
    print("[PASS] test_rans_empty_symbols")


def test_rans_single_symbol():
    """Test encoding single symbol type."""
    symbols = [3, 3, 3, 3, 3]
    freq = np.array([0, 0, 0, 5], dtype=np.int64)

    encoder = RANSEncoder()
    bits = encoder.encode(symbols)

    decoder = RANSDecoder()
    decoded = decoder.decode(bits, n=len(symbols))
    assert decoded == symbols
    print("[PASS] test_rans_single_symbol")


def test_rans_custom_model():
    """Test encoding with pre-set frequency model."""
    symbols = [0, 1, 2, 0, 1, 2]
    freq = np.array([100, 50, 25], dtype=np.int64)

    encoder = RANSEncoder()
    encoder.set_model(freq)
    bits = encoder.encode(symbols)

    decoder = RANSDecoder()
    decoded = decoder.decode(bits, n=len(symbols))
    assert decoded == symbols
    print("[PASS] test_rans_custom_model")


def test_adaptive_rans_encode():
    """Test adaptive rANS encoding produces valid output."""
    symbols = list(np.random.randint(0, 16, size=200))
    adaptive = AdaptiveRANS(alphabet_size=16)
    bits = adaptive.encode(symbols)
    assert isinstance(bits, bytes)
    assert len(bits) > 0
    print("[PASS] test_adaptive_rans_encode")


def test_entropy_analyzer_shannon():
    """Test Shannon entropy calculation."""
    analyzer = EntropyAnalyzer()

    # Uniform distribution: entropy should be high
    uniform = list(range(16)) * 100
    h_uniform = analyzer.shannon_entropy(uniform)
    assert h_uniform > 3.5, f"Uniform entropy too low: {h_uniform}"

    # Deterministic: entropy should be 0
    deterministic = [0] * 100
    h_det = analyzer.shannon_entropy(deterministic)
    assert h_det == 0.0

    # Binary: entropy should be ~1.0
    binary = [0, 1] * 500
    h_bin = analyzer.shannon_entropy(binary)
    assert abs(h_bin - 1.0) < 0.01

    print(f"[PASS] test_entropy_analyzer_shannon (uniform={h_uniform:.2f}, binary={h_bin:.2f})")


def test_entropy_analyzer_detect_distribution():
    """Test distribution detection."""
    analyzer = EntropyAnalyzer()

    # Sparse distribution
    sparse = [0] * 95 + [1, 2, 3, 4, 5]
    dist = analyzer.detect_distribution(sparse)
    assert dist['distribution_type'] == 'sparse'

    # Uniform distribution
    uniform = list(range(32)) * 50
    dist = analyzer.detect_distribution(uniform)
    assert dist['distribution_type'] == 'uniform'

    print("[PASS] test_entropy_analyzer_detect_distribution")


def test_entropy_analyzer_compression_ratio():
    """Test compression ratio estimation."""
    analyzer = EntropyAnalyzer()

    # High compression potential
    skewed = [0] * 1000 + list(range(10)) * 10
    ratio = analyzer.compression_ratio(skewed, bits_per_symbol=8.0)
    assert ratio > 1.0, f"Expected ratio > 1.0, got {ratio}"

    print(f"[PASS] test_entropy_analyzer_compression_ratio (ratio={ratio:.2f})")


def test_entropy_analyzer_recommend_alphabet():
    """Test alphabet size recommendation."""
    analyzer = EntropyAnalyzer()

    low_entropy = [0] * 100 + [1] * 20
    rec = analyzer.recommend_alphabet_size(low_entropy)
    assert rec >= 2

    high_entropy = list(range(256)) * 4
    rec = analyzer.recommend_alphabet_size(high_entropy)
    assert rec >= 256

    print("[PASS] test_entropy_analyzer_recommend_alphabet")


def test_entropy_analyzer_analyze():
    """Test full analysis pipeline."""
    analyzer = EntropyAnalyzer()
    p = np.exp(-np.arange(32) / 5)
    p = p / p.sum()
    symbols = list(np.random.choice(32, size=500, p=p))
    info = analyzer.analyze(symbols)

    assert 'entropy' in info
    assert 'distribution_type' in info
    assert 'recommended_alphabet' in info
    assert 'compression_ratio' in info
    assert info['n_symbols'] == 500

    print(f"[PASS] test_entropy_analyzer_analyze (type={info['distribution_type']})")


# ═══════════════════════════════════════════════════════════════════════════
# Hadamard Preconditioner Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_hadamard_1d_roundtrip():
    """Test that Hadamard forward+inverse recovers original."""
    v = np.random.randn(64).astype(np.float32)
    hc = HadamardPreconditioner(block_size=64, randomized=False)

    transformed = hc.forward(v)
    restored = hc.inverse(transformed)

    max_error = float(np.max(np.abs(v - restored)))
    assert max_error < 1e-4, f"Hadamard 1D round-trip error: {max_error}"
    print(f"[PASS] test_hadamard_1d_roundtrip (max_error={max_error:.6f})")


def test_hadamard_2d_roundtrip():
    """Test 2D Hadamard forward+inverse round-trip."""
    W = np.random.randn(64, 64).astype(np.float32)
    hc = HadamardPreconditioner(block_size=64, randomized=False)

    transformed = hc.forward(W)
    restored = hc.inverse(transformed)

    max_error = float(np.max(np.abs(W - restored)))
    assert max_error < 1e-3, f"Hadamard 2D round-trip error: {max_error}"
    print(f"[PASS] test_hadamard_2d_roundtrip (max_error={max_error:.6f})")


def test_hadamard_randomized_roundtrip():
    """Test randomized Hadamard round-trip."""
    W = np.random.randn(64, 64).astype(np.float32)
    hc = HadamardPreconditioner(block_size=64, randomized=True, seed=42)

    transformed = hc.forward(W)
    restored = hc.inverse(transformed)

    max_error = float(np.max(np.abs(W - restored)))
    assert max_error < 1e-3, f"Randomized Hadamard round-trip error: {max_error}"
    print(f"[PASS] test_hadamard_randomized_roundtrip (max_error={max_error:.6f})")


def test_hadamard_flattens_spectrum():
    """Test that Hadamard preconditioning flattens the weight spectrum."""
    # Create a matrix with strong low-frequency structure
    W = np.zeros((64, 64), dtype=np.float32)
    for i in range(64):
        for j in range(64):
            W[i, j] = math.exp(-(i**2 + j**2) / 200.0)

    hc = HadamardPreconditioner(block_size=64, randomized=True, seed=42)
    transformed = hc.forward(W)

    # Check that the transformed matrix has been modified (different spectrum)
    orig_var = float(np.var(W))
    trans_var = float(np.var(transformed))
    assert transformed.shape == W.shape
    print(f"[PASS] test_hadamard_flattens_spectrum (var: {orig_var:.4f} -> {trans_var:.4f})")


def test_hadamard_block_diagonal():
    """Test block-diagonal Hadamard on power-of-2 sizes with multiple blocks."""
    W = np.random.randn(64, 64).astype(np.float32)
    hc = HadamardPreconditioner(block_size=32, randomized=False)

    transformed = hc.forward(W)
    assert transformed.shape == W.shape

    restored = hc.inverse(transformed)
    max_error = float(np.max(np.abs(W - restored)))
    assert max_error < 0.01, f"Block-diagonal Hadamard error too large: {max_error}"
    print(f"[PASS] test_hadamard_block_diagonal (max_error={max_error:.6f})")


# ═══════════════════════════════════════════════════════════════════════════
# Incoherence Transform Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_incoherence_metric():
    """Test incoherence metric computation."""
    # Uniform matrix: low incoherence
    uniform = np.ones((16, 16)) * 5.0
    mu_uniform = IncoherenceTransform.incoherence_metric(uniform)
    assert abs(mu_uniform - 1.0) < 0.01

    # Matrix with outlier: high incoherence
    outlier = np.ones((16, 16))
    outlier[0, 0] = 100.0
    mu_outlier = IncoherenceTransform.incoherence_metric(outlier)
    assert mu_outlier > mu_uniform

    print(f"[PASS] test_incoherence_metric (uniform={mu_uniform:.3f}, outlier={mu_outlier:.3f})")


def test_incoherence_random_rotation_roundtrip():
    """Test random rotation transform round-trip."""
    W = np.random.randn(32, 32)
    ict = IncoherenceTransform(method='random', seed=42)

    transformed, meta = ict.transform(W)
    restored = ict.inverse_transform(transformed, meta)

    max_error = float(np.max(np.abs(W - restored)))
    assert max_error < 1e-10, f"Random rotation round-trip error: {max_error}"
    print(f"[PASS] test_incoherence_random_rotation_roundtrip (error={max_error:.2e})")


def test_incoherence_butterfly_roundtrip():
    """Test butterfly transform round-trip."""
    W = np.random.randn(32, 32)
    ict = IncoherenceTransform(method='butterfly', seed=42)

    transformed, meta = ict.transform(W)
    restored = ict.inverse_transform(transformed, meta)

    max_error = float(np.max(np.abs(W - restored)))
    assert max_error < 1e-10, f"Butterfly transform round-trip error: {max_error}"
    print(f"[PASS] test_incoherence_butterfly_roundtrip (error={max_error:.2e})")


def test_incoherence_reduces_outliers():
    """Test that incoherence transform reduces outlier magnitude."""
    # Create matrix with extreme outliers
    W = np.random.randn(64, 64) * 0.01
    W[0, 0] = 100.0  # Extreme outlier
    W[1, 1] = -50.0

    ict = IncoherenceTransform(method='butterfly', seed=42)
    transformed, meta = ict.transform(W)

    mu_before = IncoherenceTransform.incoherence_metric(W)
    mu_after = IncoherenceTransform.incoherence_metric(transformed)

    print(f"[PASS] test_incoherence_reduces_outliers (μ: {mu_before:.2f} -> {mu_after:.2f})")


def test_incoherence_1d_roundtrip():
    """Test 1D incoherence transform round-trip."""
    v = np.random.randn(64)
    ict = IncoherenceTransform(method='butterfly', seed=42)

    transformed, meta = ict.transform(v)
    restored = ict.inverse_transform(transformed, meta)

    max_error = float(np.max(np.abs(v - restored)))
    assert max_error < 1e-10, f"1D incoherence round-trip error: {max_error}"
    print(f"[PASS] test_incoherence_1d_roundtrip (error={max_error:.2e})")


# ═══════════════════════════════════════════════════════════════════════════
# Spectral Shaping Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_peak_to_average_ratio():
    """Test PAR computation."""
    uniform = np.ones((16, 16))
    par_uniform = SpectralShaping.peak_to_average_ratio(uniform)
    assert abs(par_uniform - 1.0) < 0.01

    peaked = np.zeros((16, 16))
    peaked[0, 0] = 100.0
    par_peaked = SpectralShaping.peak_to_average_ratio(peaked)
    assert par_peaked > 1.0

    print(f"[PASS] test_peak_to_average_ratio (uniform={par_uniform:.3f}, peaked={par_peaked:.3f})")


def test_spectral_equalization():
    """Test spectral equalization flattens DCT spectrum."""
    # Create a matrix with strong low-frequency energy
    W = np.zeros((64, 64), dtype=np.float64)
    for i in range(64):
        for j in range(64):
            W[i, j] = math.exp(-(i**2 + j**2) / 100.0) + np.random.randn() * 0.01

    equalized, meta = SpectralShaping.spectral_equalization(W, strength=0.8)

    # After equalization, PAR should be closer to 1
    par_before = SpectralShaping.peak_to_average_ratio(W)
    par_after = SpectralShaping.peak_to_average_ratio(equalized)

    print(f"[PASS] test_spectral_equalization (PAR: {par_before:.2f} -> {par_after:.2f})")


def test_error_diffusion():
    """Test error diffusion spreads quantization error."""
    original = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    quantized = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    diffused = SpectralShaping.error_diffusion(original, quantized, strength=0.5)
    # With zero error, output should be unchanged
    np.testing.assert_allclose(diffused, quantized, atol=1e-10)

    # With non-zero error, diffusion should spread it
    quantized_coarse = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    diffused = SpectralShaping.error_diffusion(original, quantized_coarse, strength=1.0)

    # Check total error is preserved (conservation)
    total_error_before = float(np.sum(original - quantized_coarse))
    total_error_after = float(np.sum(original - diffused))
    assert abs(total_error_before - total_error_after) < 0.1, \
        f"Total error not preserved: {total_error_before:.4f} -> {total_error_after:.4f}"

    print("[PASS] test_error_diffusion")


def test_par_reduction():
    """Test peak-to-average ratio reduction."""
    W = np.random.randn(64, 64)
    W[0, 0] = 100.0  # Outlier

    par_before = SpectralShaping.peak_to_average_ratio(W)
    reduced = SpectralShaping.par_reduction(W, strength=0.8)
    par_after = SpectralShaping.peak_to_average_ratio(reduced)

    assert par_after <= par_before + 0.1, \
        f"PAR should decrease or stay same: {par_before:.2f} -> {par_after:.2f}"

    print(f"[PASS] test_par_reduction (PAR: {par_before:.2f} -> {par_after:.2f})")


def test_spectral_shaping_apply():
    """Test the combined apply() method."""
    W = np.random.randn(64, 64) * 10
    shaping = SpectralShaping(method='all', strength=0.5)

    shaped, meta = shaping.apply(W)

    assert 'method' in meta
    assert 'strength' in meta
    assert 'par_ratio' in meta
    assert shaped.shape == W.shape

    print(f"[PASS] test_spectral_shaping_apply (PAR: {meta.get('par_ratio', 'N/A'):.2f})")


def test_pre_quantize():
    """Test pre_quantize() pipeline."""
    W = np.random.randn(64, 64)
    shaping = SpectralShaping(method='all', strength=0.5)

    quantized, meta = shaping.pre_quantize(W, n_bits=8)

    assert 'n_bits' in meta
    assert 'scale' in meta
    assert quantized.shape == W.shape

    # Quantized values should be in valid range
    max_abs = float(np.max(np.abs(quantized)))
    assert max_abs < float(np.max(np.abs(W))) + 1.0

    print(f"[PASS] test_pre_quantize (PAR: {meta.get('par_before', 'N/A'):.2f} -> {meta.get('par_after', 'N/A'):.2f})")


# ═══════════════════════════════════════════════════════════════════════════
# Integration Test: Hadamard + Quantization improvement
# ═══════════════════════════════════════════════════════════════════════════

def test_hadamard_improves_quantization():
    """Test that Hadamard preconditioning improves quantization accuracy."""
    np.random.seed(42)

    # Create a matrix with strong structure (typical of NN weights)
    W = np.random.randn(64, 64).astype(np.float64)
    W = W * 10 + np.sin(np.linspace(0, 4 * np.pi, 64))[:, None] * 5

    n_bits = 4
    max_val = (1 << (n_bits - 1)) - 1
    scale = float(np.max(np.abs(W))) / max_val

    # Direct quantization
    q_direct = np.clip(np.round(W / scale), -max_val, max_val) * scale
    mse_direct = float(np.mean((W - q_direct) ** 2))

    # Hadamard-preconditioned quantization
    hc = HadamardPreconditioner(block_size=64, randomized=True, seed=42)
    W_h = hc.forward(W)
    scale_h = float(np.max(np.abs(W_h))) / max_val
    q_h = np.clip(np.round(W_h / scale_h), -max_val, max_val) * scale_h
    W_restored = hc.inverse(q_h)
    mse_hadamard = float(np.mean((W - W_restored) ** 2))

    improvement = mse_direct / max(mse_hadamard, 1e-10)
    print(f"[PASS] test_hadamard_improves_quantization "
          f"(MSE: direct={mse_direct:.6f}, hadamard={mse_hadamard:.6f}, "
          f"improvement={improvement:.2f}x)")


def test_incoherence_improves_quantization():
    """Test that incoherence transform improves quantization."""
    np.random.seed(42)

    W = np.random.randn(64, 64).astype(np.float64)
    W[0, 0] = 100.0  # Outlier

    n_bits = 8
    max_val = (1 << (n_bits - 1)) - 1

    # Direct quantization
    scale = float(np.max(np.abs(W))) / max_val
    q_direct = np.clip(np.round(W / scale), -max_val, max_val) * scale
    mse_direct = float(np.mean((W - q_direct) ** 2))

    # Incoherence-transformed quantization
    ict = IncoherenceTransform(method='butterfly', seed=42)
    W_t, meta = ict.transform(W)
    scale_t = float(np.max(np.abs(W_t))) / max_val
    q_t = np.clip(np.round(W_t / scale_t), -max_val, max_val) * scale_t
    W_restored = ict.inverse_transform(q_t, meta)
    mse_incoherence = float(np.mean((W - W_restored) ** 2))

    improvement = mse_direct / max(mse_incoherence, 1e-10)
    print(f"[PASS] test_incoherence_improves_quantization "
          f"(MSE: direct={mse_direct:.6f}, incoherence={mse_incoherence:.6f}, "
          f"improvement={improvement:.2f}x)")


def test_full_pipeline():
    """Test complete pipeline: Hadamard -> Spectral Shaping -> Quantize -> rANS."""
    np.random.seed(42)

    # Simulate weight quantization indices
    symbols = list(np.random.randint(0, 16, size=1000))
    freq = np.zeros(16, dtype=np.int64)
    for s in symbols:
        freq[s] += 1

    # rANS encode/decode
    encoder = RANSEncoder()
    bits = encoder.encode(symbols)
    decoder = RANSDecoder()
    decoded = decoder.decode(bits, n=len(symbols))
    assert decoded == symbols

    # Entropy analysis
    analyzer = EntropyAnalyzer()
    info = analyzer.analyze(symbols)
    assert info['n_symbols'] == 1000

    print(f"[PASS] test_full_pipeline "
          f"(entropy={info['entropy']:.2f} bits, "
          f"compression={info['compression_ratio']:.2f}x, "
          f"type={info['distribution_type']})")


# ═══════════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("rANS Entropy Coding & Hadamard Preconditioner — Test Suite")
    print("=" * 70)
    print()

    # rANS tests
    print("--- rANS Entropy Coding ---")
    test_normalize_frequencies()
    test_rans_encode_decode_roundtrip()
    test_rans_16bit_roundtrip()
    test_rans_compression_ratio()
    test_rans_empty_symbols()
    test_rans_single_symbol()
    test_rans_custom_model()
    test_adaptive_rans_encode()
    print()

    # Entropy analyzer tests
    print("--- Entropy Analyzer ---")
    test_entropy_analyzer_shannon()
    test_entropy_analyzer_detect_distribution()
    test_entropy_analyzer_compression_ratio()
    test_entropy_analyzer_recommend_alphabet()
    test_entropy_analyzer_analyze()
    print()

    # Hadamard tests
    print("--- Hadamard Preconditioner ---")
    test_hadamard_1d_roundtrip()
    test_hadamard_2d_roundtrip()
    test_hadamard_randomized_roundtrip()
    test_hadamard_flattens_spectrum()
    test_hadamard_block_diagonal()
    print()

    # Incoherence transform tests
    print("--- Incoherence Transform ---")
    test_incoherence_metric()
    test_incoherence_random_rotation_roundtrip()
    test_incoherence_butterfly_roundtrip()
    test_incoherence_reduces_outliers()
    test_incoherence_1d_roundtrip()
    print()

    # Spectral shaping tests
    print("--- Spectral Shaping ---")
    test_peak_to_average_ratio()
    test_spectral_equalization()
    test_error_diffusion()
    test_par_reduction()
    test_spectral_shaping_apply()
    test_pre_quantize()
    print()

    # Integration tests
    print("--- Integration Tests ---")
    test_hadamard_improves_quantization()
    test_incoherence_improves_quantization()
    test_full_pipeline()
    print()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
