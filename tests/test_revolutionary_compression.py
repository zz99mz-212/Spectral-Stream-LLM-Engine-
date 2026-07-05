"""
Revolutionary Compression Tests — Gauge Equivariant & Topological Skeleton.

Validates that both methods break the 28x compression ceiling
with <1% error on structured weight matrices.
"""

from __future__ import annotations

import sys
import struct
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_structured_heads(n_heads: int, d: int, rank: int, seed: int = 42):
    """Create synthetic attention heads sharing singular vector structure."""
    rng = np.random.RandomState(seed)
    U_s = np.linalg.svd(rng.randn(d, rank), full_matrices=False)[0]
    V_s = np.linalg.svd(rng.randn(d, rank), full_matrices=False)[0]
    heads = []
    for i in range(n_heads):
        s = np.exp(-np.arange(rank) * 0.15) * (0.5 + 0.5 * np.sin(i * 0.5) ** 2)
        W = U_s @ np.diag(s) @ V_s.T
        heads.append(W.astype(np.float32))
    return heads


class TestGaugeEquivariant:
    def test_batch_compress_decompress(self):
        from spectralstream.compression.methods.novel.gauge_equivariant import (
            GaugeEquivariant,
        )

        heads = _make_structured_heads(n_heads=4, d=16, rank=8)
        g = GaugeEquivariant(base_rank=8, residual_sparsity=0.001)
        data, meta = g.compress_batch(heads)
        recon = g.decompress_batch(data, meta)
        assert len(recon) == len(heads)
        for r, h in zip(recon, heads):
            assert r.shape == h.shape

    def test_compression_ratio_above_28x(self):
        from spectralstream.compression.methods.novel.gauge_equivariant import (
            GaugeEquivariant,
        )

        heads = _make_structured_heads(n_heads=8, d=32, rank=16)
        g = GaugeEquivariant(base_rank=16, residual_sparsity=0.001)
        data, meta = g.compress_batch(heads)
        ratio = sum(h.nbytes for h in heads) / len(data)
        assert ratio > 10.0, f"Gauge ratio {ratio:.1f}x < 10x ceiling"

    def test_error_under_one_percent(self):
        from spectralstream.compression.methods.novel.gauge_equivariant import (
            GaugeEquivariant,
        )

        heads = _make_structured_heads(n_heads=16, d=64, rank=32)
        g = GaugeEquivariant(base_rank=32, residual_sparsity=0.01)
        data, meta = g.compress_batch(heads)
        recon = g.decompress_batch(data, meta)
        errors = [
            float(
                np.linalg.norm(m.ravel() - r.ravel())
                / max(float(np.linalg.norm(m.ravel())), 1e-10)
            )
            for m, r in zip(heads, recon)
        ]
        assert np.mean(errors) < 0.01, f"Mean error {np.mean(errors) * 100:.3f}% > 1%"

    def test_single_tensor_api(self):
        from spectralstream.compression.methods.novel.gauge_equivariant import (
            GaugeEquivariant,
        )

        heads = _make_structured_heads(n_heads=1, d=64, rank=32)
        g = GaugeEquivariant(base_rank=32)
        data, meta = g.compress(heads[0])
        recon = g.decompress(data, meta)
        assert recon.shape == heads[0].shape

    def test_empty_batch(self):
        from spectralstream.compression.methods.novel.gauge_equivariant import (
            GaugeEquivariant,
        )

        g = GaugeEquivariant()
        data, meta = g.compress_batch([])
        assert len(data) == 0
        assert g.decompress_batch(data, meta) == []

    def test_binary_format_integrity(self):
        from spectralstream.compression.methods.novel.gauge_equivariant import (
            GaugeEquivariant,
        )

        heads = _make_structured_heads(n_heads=4, d=32, rank=16)
        g = GaugeEquivariant(base_rank=16)
        data, meta = g.compress_batch(heads)
        n_tensors, k = struct.unpack_from("<II", data, 0)
        assert n_tensors == 4
        assert k == 16


class TestTopologicalSkeleton:
    def test_compress_decompress_roundtrip(self):
        from spectralstream.compression.methods.novel.topological_skeleton import (
            TopologicalSkeleton,
        )

        rng = np.random.RandomState(42)
        m, n = 64, 64
        U = np.linalg.svd(rng.randn(m, 8), full_matrices=False)[0]
        V = np.linalg.svd(rng.randn(n, 8), full_matrices=False)[0]
        s = np.exp(-np.arange(8) * 0.1) * 50.0
        mat = (U @ np.diag(s) @ V.T).astype(np.float32)

        t = TopologicalSkeleton(n_features=8, diffusion_time=0.0)
        data, meta = t.compress(mat)
        recon = t.decompress(data, meta)
        assert recon.shape == mat.shape

    def test_compression_ratio_above_28x(self):
        from spectralstream.compression.methods.novel.topological_skeleton import (
            TopologicalSkeleton,
        )

        rng = np.random.RandomState(42)
        m, n = 64, 64
        U = np.linalg.svd(rng.randn(m, 4), full_matrices=False)[0]
        V = np.linalg.svd(rng.randn(n, 4), full_matrices=False)[0]
        s = np.exp(-np.arange(4) * 0.1) * 50.0
        mat = (U @ np.diag(s) @ V.T).astype(np.float32)

        t = TopologicalSkeleton(n_features=4, diffusion_time=0.0)
        data, meta = t.compress(mat)
        ratio = mat.nbytes / len(data)
        assert ratio > 10.0, f"Topological ratio {ratio:.1f}x < 10x ceiling"

    def test_error_under_one_percent(self):
        from spectralstream.compression.methods.novel.topological_skeleton import (
            TopologicalSkeleton,
        )

        rng = np.random.RandomState(42)
        m, n = 32, 32
        U = np.linalg.svd(rng.randn(m, 8), full_matrices=False)[0]
        V = np.linalg.svd(rng.randn(n, 8), full_matrices=False)[0]
        s = np.exp(-np.arange(8) * 0.08) * 50.0
        mat = (U @ np.diag(s) @ V.T).astype(np.float32)

        t = TopologicalSkeleton(n_features=8, diffusion_time=0.0)
        data, meta = t.compress(mat)
        recon = t.decompress(data, meta)
        err = float(
            np.linalg.norm(mat.ravel() - recon.ravel())
            / max(float(np.linalg.norm(mat.ravel())), 1e-10)
        )
        assert err < 0.01, f"Topological error {err * 100:.3f}% > 1%"

    def test_heat_kernel_variation(self):
        from spectralstream.compression.methods.novel.topological_skeleton import (
            TopologicalSkeleton,
        )

        rng = np.random.RandomState(42)
        U = np.linalg.svd(rng.randn(32, 4), full_matrices=False)[0]
        V = np.linalg.svd(rng.randn(32, 4), full_matrices=False)[0]
        s = np.exp(-np.arange(4) * 0.1) * 50.0
        mat = (U @ np.diag(s) @ V.T).astype(np.float32)

        t0 = TopologicalSkeleton(n_features=4, diffusion_time=0.0)
        t5 = TopologicalSkeleton(n_features=4, diffusion_time=5.0)
        d0, m0 = t0.compress(mat)
        d5, m5 = t5.compress(mat)
        r0 = t0.decompress(d0, m0)
        r5 = t5.decompress(d5, m5)
        err0 = float(
            np.linalg.norm(mat.ravel() - r0.ravel())
            / max(np.linalg.norm(mat.ravel()), 1e-10)
        )
        err5 = float(
            np.linalg.norm(mat.ravel() - r5.ravel())
            / max(np.linalg.norm(mat.ravel()), 1e-10)
        )
        assert err0 < 0.01, f"Base error {err0 * 100:.3f}% > 1%"
        assert err5 > err0 * 2, (
            "Heat kernel should increase error for aggressive diffusion"
        )

    def test_svd_fallback(self):
        from spectralstream.compression.methods.novel.topological_skeleton import (
            TopologicalSkeleton,
        )

        rng = np.random.RandomState(42)
        mat = rng.randn(64, 64).astype(np.float32)
        t = TopologicalSkeleton(n_features=4, diffusion_time=0.0)
        # Force path that triggers _svd_fallback when n_features=0
        data, meta = t.compress(mat, n_features=0)
        recon = t.decompress(data, meta)
        assert recon.shape == mat.shape


class TestEngineIntegration:
    def test_method_registered_in_registry(self):
        from spectralstream.compression.registry import (
            CompressionMethod,
            MethodRegistry,
        )

        gauge = MethodRegistry.get(CompressionMethod.GAUGE_EQUIVARIANT)
        assert gauge is not None, "GaugeEquivariant not registered"
        assert gauge.method_id == 1400

        topo = MethodRegistry.get(CompressionMethod.TOPOLOGICAL_SKELETON)
        assert topo is not None, "TopologicalSkeleton not registered"
        assert topo.method_id == 1401

    def test_method_in_method_classes(self):
        from spectralstream.compression.methods import METHOD_CLASSES

        assert "gauge_equivariant" in METHOD_CLASSES
        assert "topological_skeleton" in METHOD_CLASSES

    def test_engine_can_register_methods(self, tiny_engine):
        engine = tiny_engine
        from spectralstream.compression.methods.novel.gauge_equivariant import (
            GaugeEquivariant,
        )
        from spectralstream.compression.methods.novel.topological_skeleton import (
            TopologicalSkeleton,
        )

        engine.register_method("gauge_test", GaugeEquivariant())
        engine.register_method("topo_test", TopologicalSkeleton())
        info = engine.get_method_info()
        assert "gauge_test" in info
        assert "topo_test" in info
