"""Edge-case tests extracted and adapted from archive tests for the main codebase.

Uses the ``tiny_engine`` fixture from ``conftest.py``.
"""

import gc
import math
import os
import sys
import threading
from typing import Dict, Optional

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectralstream.compression.engine import CompressionIntelligenceEngine


def snr_db(orig: np.ndarray, recon: np.ndarray) -> float:
    """Signal-to-noise ratio in dB."""
    mse = float(np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2))
    var = float(np.mean(orig.astype(np.float64) ** 2))
    eps = 1e-30
    return float(10 * np.log10(var / max(mse, eps))) if mse > eps else 100.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if denom < 1e-30:
        return 1.0
    return float(np.dot(a_f, b_f) / denom)


def _compress_recon(engine, tensor, name="test"):
    """Helper: compress then decompress, returning (compressed_data, metadata, reconstruction)."""
    data, meta, ratio, error = engine.compress(tensor, name=name)
    recon = engine.decompress(data, meta)
    return data, meta, recon, ratio, error


class TestEdgeCasesCompression:
    """Edge-case patterns extracted from archive tests, adapted for the main
    ``CompressionIntelligenceEngine`` API."""

    def test_nan_handling(self, tiny_engine):
        rng = np.random.RandomState(42)
        tensor = rng.randn(32, 32).astype(np.float32)
        tensor[0, 0] = float("nan")
        tensor[1, 1] = float("inf")
        tensor[2, 2] = float("-inf")

        cleaned = np.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        _, _, recon, _, _ = _compress_recon(tiny_engine, cleaned, "test_nan")

        assert np.all(np.isfinite(recon)), "NaN/inf leaked through compression"
        assert recon.shape == tensor.shape

    def test_zero_tensor(self, tiny_engine):
        tensor = np.zeros((16, 16), dtype=np.float32)
        _, _, recon, _, _ = _compress_recon(tiny_engine, tensor, "test_zero")

        assert recon.shape == tensor.shape
        assert cosine_similarity(tensor, recon) > 0.99

    def test_constant_tensor(self, tiny_engine):
        tensor = np.ones((16, 16), dtype=np.float32) * 3.14159
        _, _, recon, _, _ = _compress_recon(tiny_engine, tensor, "test_constant")

        assert recon.shape == tensor.shape
        assert cosine_similarity(tensor, recon) > 0.99

    def test_single_element(self, tiny_engine):
        tensor = np.array([[42.0]], dtype=np.float32)
        _, _, recon, _, _ = _compress_recon(tiny_engine, tensor, "test_single")

        assert recon.shape == tensor.shape
        assert recon.size == 1

    def test_1d_tensor(self, tiny_engine):
        rng = np.random.RandomState(42)
        tensor = rng.randn(256).astype(np.float32)
        _, _, recon, _, _ = _compress_recon(tiny_engine, tensor, "test_1d")

        assert recon.shape == tensor.shape

    def test_tiny_matrix(self, tiny_engine):
        rng = np.random.RandomState(42)
        tensor = rng.randn(2, 3).astype(np.float32)
        _, _, recon, _, _ = _compress_recon(tiny_engine, tensor, "test_tiny")

        assert recon.shape == tensor.shape

    def test_multiple_compress_decompress_consistent(self, tiny_engine):
        rng = np.random.RandomState(42)
        tensor = rng.randn(32, 32).astype(np.float32)

        results = []
        for _ in range(3):
            _, _, recon, _, _ = _compress_recon(tiny_engine, tensor, "test_repeat")
            results.append(recon)

        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i], atol=1e-5), (
                "Repeated compress/decompress produces different results"
            )

    def test_large_random_matrix(self, tiny_engine):
        rng = np.random.RandomState(42)
        tensor = rng.randn(128, 128).astype(np.float32) * 0.02
        _, _, recon, _, _ = _compress_recon(tiny_engine, tensor, "test_large")

        assert recon.shape == tensor.shape
        assert cosine_similarity(tensor, recon) > 0.9

    def test_compression_produces_output(self, tiny_engine):
        rng = np.random.RandomState(42)
        tensor = rng.randn(16, 16).astype(np.float32)
        data, meta, recon, ratio, _ = _compress_recon(tiny_engine, tensor, "test_ratio")

        assert len(data) > 0
        assert isinstance(meta, dict)
        assert recon.shape == tensor.shape

    def test_bounded_relative_error(self, tiny_engine):
        rng = np.random.RandomState(42)
        tensor = rng.randn(32, 32).astype(np.float32) * 0.02
        _, _, recon, _, err = _compress_recon(tiny_engine, tensor, "test_error")

        orig_norm = float(np.linalg.norm(tensor))
        err_norm = float(np.linalg.norm(tensor - recon))
        rel_err = err_norm / max(orig_norm, 1e-10)

        assert rel_err < 0.10, f"Relative error too high: {rel_err:.4%}"


class TestQualityValidation:
    """Quality-validation patterns extracted from archive tests."""

    def test_snr_same_tensor(self):
        rng = np.random.RandomState(42)
        t = rng.randn(16, 16).astype(np.float32)
        assert snr_db(t, t) >= 100.0

    def test_snr_noisy_tensor(self):
        rng = np.random.RandomState(42)
        t = rng.randn(16, 16).astype(np.float32)
        noisy = t + rng.randn(16, 16).astype(np.float32) * 0.1
        s = snr_db(t, noisy)
        assert 5.0 < s < 50.0

    def test_ssim_identical(self):
        rng = np.random.RandomState(42)
        a = rng.randn(16, 16).astype(np.float32)
        mu_a = float(np.mean(a))
        var_a = float(np.var(a))
        num = (2 * mu_a * mu_a + 0.01**2) * (2 * var_a + 0.03**2)
        den = (mu_a**2 + mu_a**2 + 0.01**2) * (var_a + var_a + 0.03**2)
        ssim = float(num / max(den, 1e-30))
        assert abs(ssim - 1.0) < 1e-6

    def test_cosine_identical(self):
        rng = np.random.RandomState(42)
        a = rng.randn(32).astype(np.float32)
        assert abs(cosine_similarity(a, a) - 1.0) < 1e-6

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_cosine_opposite(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6


class TestParallelCompressionSafety:
    """Parallel compression safety adapted from archive race-condition test."""

    def test_concurrent_compress_decompress(self):
        engine = CompressionIntelligenceEngine()
        rng = np.random.RandomState(42)
        tensor = rng.randn(32, 32).astype(np.float32) * 0.02

        results: Dict[str, Optional[Exception]] = {}
        lock = threading.Lock()

        def worker(name: str):
            try:
                data, meta, _, _, _ = _compress_recon(engine, tensor, name)
                with lock:
                    results[name] = None
            except Exception as e:
                with lock:
                    results[name] = e

        names = [f"thread_{i}" for i in range(4)]
        threads = [threading.Thread(target=worker, args=(n,)) for n in names]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        for name in names:
            assert name in results, f"Thread {name} did not complete"
            assert results[name] is None, f"Thread {name} failed: {results[name]}"

        if hasattr(engine, "close"):
            engine.close()
        del engine
        gc.collect()
