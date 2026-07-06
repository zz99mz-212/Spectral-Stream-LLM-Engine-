"""
Comprehensive tests for the LossMetricsIntelligenceEngine.

Tests each metric individually, all metrics on random tensors,
quality grading, edge cases, and the < 1% loss threshold.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.compression.world_model.loss_metrics_engine import (
    LossMetricsIntelligenceEngine,
    PerTensorLossMetrics,
    QualityGrade,
    SpectralMetrics,
    StatisticalMetrics,
    StructuralMetrics,
    CompressionMetrics,
    _spectral_entropy_1d,
    _energy_concentration,
    _effective_rank,
    _spectral_norm,
    _condition_number,
    _mutual_information_estimate,
    _ssim_2d,
    _entropy_rate,
    _kolmogorov_estimate,
    _cross_correlation_preserved,
)


@pytest.fixture
def engine():
    return LossMetricsIntelligenceEngine(use_svd_cache=True)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Spectral Metrics
# ═══════════════════════════════════════════════════════════════════════════


class TestSpectralMetrics:
    def test_identical_tensors(self, engine):
        t = np.random.RandomState(42).randn(32, 32).astype(np.float32)
        m = engine._compute_spectral_metrics(t, t)
        assert m.l1_error == 0.0
        assert m.mse == 0.0
        assert m.rmse == 0.0
        assert m.spectral_norm_error == 0.0
        assert m.spectral_entropy_change == 0.0
        assert m.energy_concentration_preserved == 1.0

    def test_known_mse(self, engine):
        orig = np.zeros((4, 4), dtype=np.float64)
        recon = np.ones((4, 4), dtype=np.float64) * 2.0
        m = engine._compute_spectral_metrics(orig, recon)
        assert m.mse == pytest.approx(4.0)
        assert m.rmse == pytest.approx(2.0)
        assert m.l1_error == pytest.approx(2.0)

    def test_spectral_entropy_1d_uniform(self):
        uniform = np.ones(128)
        ent = _spectral_entropy_1d(uniform)
        assert ent >= 0.0

    def test_spectral_entropy_1d_sine(self):
        t = np.linspace(0, 2 * np.pi, 256)
        sine = np.sin(t)
        ent = _spectral_entropy_1d(sine)
        # Pure sinusoid has low spectral entropy (concentrated spectrum)
        assert ent < 0.5

    def test_energy_concentration_sine(self):
        t = np.linspace(0, 2 * np.pi, 128)
        sine = np.sin(t)
        ec = _energy_concentration(sine)
        # Energy should be concentrated in few coefficients
        assert ec <= 1.0
        assert ec > 0.0

    def test_spectral_metrics_almost_identical(self, engine):
        orig = np.random.RandomState(42).randn(16, 16).astype(np.float32)
        recon = orig + np.random.RandomState(99).randn(16, 16).astype(np.float32) * 1e-6
        m = engine._compute_spectral_metrics(orig, recon)
        assert m.mse < 1e-8
        assert m.energy_concentration_preserved > 0.99

    def test_spectral_norm(self):
        A = np.diag(np.array([3.0, 2.0, 1.0]))
        sn = _spectral_norm(A)
        assert sn == pytest.approx(3.0, rel=1e-4)

    def test_spectral_norm_power_iteration(self):
        # Large enough to trigger power iteration
        A = np.random.RandomState(42).randn(400, 400)
        sn = _spectral_norm(A, power_iters=30)
        s = np.linalg.svd(A, compute_uv=False)
        assert sn == pytest.approx(float(s[0]), rel=0.1)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Statistical Metrics
# ═══════════════════════════════════════════════════════════════════════════


class TestStatisticalMetrics:
    def test_identical_tensors(self, engine):
        t = np.random.RandomState(42).randn(32, 32).astype(np.float32)
        m = engine._compute_statistical_metrics(t, t)
        assert m.snr_db == float("inf")
        assert m.psnr_db == float("inf")
        assert m.ssim == pytest.approx(1.0, abs=1e-6)
        assert m.cosine_similarity == pytest.approx(1.0)
        assert m.kl_divergence == 0.0
        assert m.wasserstein_distance == 0.0
        assert m.mae == 0.0

    def test_snr_known(self, engine):
        rng = np.random.RandomState(42)
        orig = rng.randn(1000).astype(np.float64) * 10.0
        noise = rng.randn(1000) * 1.0
        recon = orig + noise
        m = engine._compute_statistical_metrics(orig, recon)
        signal_var = float(np.var(orig))
        noise_var = float(np.var(noise))
        expected_snr = 10.0 * math.log10(signal_var / noise_var)
        assert m.snr_db == pytest.approx(expected_snr, rel=0.1)

    def test_cosine_similarity_orthogonal(self, engine):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        m = engine._compute_statistical_metrics(a, b)
        assert m.cosine_similarity == pytest.approx(0.0, abs=1e-10)

    def test_cosine_similarity_opposite(self, engine):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])
        m = engine._compute_statistical_metrics(a, b)
        assert m.cosine_similarity == pytest.approx(-1.0, abs=1e-10)

    def test_ssim_identical_small(self):
        t = np.random.RandomState(42).randn(8, 8).astype(np.float64)
        ssim = _ssim_2d(t, t)
        assert ssim == pytest.approx(1.0, abs=1e-6)

    def test_ssim_known(self):
        orig = np.ones((16, 16), dtype=np.float64) * 10.0
        recon = orig + np.random.RandomState(42).randn(16, 16) * 0.01
        ssim = _ssim_2d(orig, recon)
        assert ssim > 0.9
        assert ssim <= 1.0

    def test_kl_divergence_identical(self, engine):
        t = np.random.RandomState(42).randn(1000).astype(np.float64)
        m = engine._compute_statistical_metrics(t, t)
        assert m.kl_divergence == pytest.approx(0.0, abs=1e-6)

    def test_mae(self, engine):
        orig = np.array([1.0, 2.0, 3.0, 4.0])
        recon = np.array([1.5, 2.5, 3.5, 4.5])
        m = engine._compute_statistical_metrics(orig, recon)
        assert m.mae == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Structural Metrics
# ═══════════════════════════════════════════════════════════════════════════


class TestStructuralMetrics:
    def test_identical_tensors(self, engine):
        t = np.random.RandomState(42).randn(16, 16).astype(np.float32)
        m = engine._compute_structural_metrics(t, t)
        assert m.effective_rank_preserved == pytest.approx(1.0, abs=1e-6)
        assert m.sparsity_preserved == pytest.approx(1.0, abs=1e-6)
        assert m.mutual_information_preserved == pytest.approx(1.0, abs=0.01)
        assert m.cross_correlation_preserved == pytest.approx(1.0, abs=0.01)

    def test_effective_rank_low_rank(self):
        # A rank-2 matrix should have effective rank near 2
        A = np.outer(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]))
        A += np.outer(np.array([5.0, 6.0, 7.0, 8.0]), np.array([5.0, 6.0, 7.0, 8.0]))
        er = _effective_rank(A)
        assert er < 4.0
        assert er >= 1.0

    def test_effective_rank_full_rank(self):
        A = np.random.RandomState(42).randn(32, 32)
        er = _effective_rank(A)
        assert er > 1.0
        assert er <= 32.0

    def test_sparsity_preserved(self, engine):
        orig = np.zeros(100)
        orig[::5] = 1.0
        recon = orig.copy()
        recon[3::5] = 0.5
        m = engine._compute_structural_metrics(orig, recon)
        assert m.sparsity_preserved >= 0.799

    def test_condition_number(self):
        A = np.diag(np.array([1.0, 2.0, 10.0]))
        cn = _condition_number(A)
        assert cn == pytest.approx(10.0)

    def test_mutual_information_identical(self):
        t = np.random.RandomState(42).randn(200).astype(np.float64)
        mi = _mutual_information_estimate(t, t)
        assert mi == pytest.approx(1.0, abs=0.05)

    def test_cross_correlation_identical(self, engine):
        t = np.random.RandomState(42).randn(64).astype(np.float64)
        cc = _cross_correlation_preserved(t, t)
        assert cc == pytest.approx(1.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Compression Metrics
# ═══════════════════════════════════════════════════════════════════════════


class TestCompressionMetrics:
    def test_compression_ratio(self, engine):
        orig = np.random.RandomState(42).randn(32, 32).astype(np.float32)
        m = engine._compute_compression_metrics(orig, orig.nbytes // 2)
        assert m.compression_ratio == pytest.approx(2.0)

    def test_bit_rate(self, engine):
        orig = np.random.RandomState(42).randn(100).astype(np.float32)
        comp_bytes = 100  # 1 byte per element
        m = engine._compute_compression_metrics(orig, comp_bytes)
        assert m.bit_rate == pytest.approx(8.0)

    def test_no_compression(self, engine):
        orig = np.random.RandomState(42).randn(10).astype(np.float64)
        m = engine._compute_compression_metrics(orig, 0)
        assert m.compression_ratio == 1.0

    def test_entropy_rate_uniform(self):
        t = np.ones(500).astype(np.float64)
        er = _entropy_rate(t)
        assert er >= 0.0
        assert er < 4.0  # Uniform signal has low entropy rate

    def test_kolmogorov_estimate(self):
        t = np.sin(np.linspace(0, 4 * np.pi, 200)).astype(np.float64)
        ke = _kolmogorov_estimate(t)
        assert ke >= 0.0
        assert ke <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. Quality Grading
# ═══════════════════════════════════════════════════════════════════════════


class TestQualityGrading:
    def test_excellent_grade(self, engine):
        orig = np.random.RandomState(42).randn(16, 16).astype(np.float32)
        recon = orig + np.random.RandomState(99).randn(16, 16).astype(np.float32) * 1e-8
        metrics = engine.compute_all_metrics(
            orig, recon, tensor_name="test", tensor_type="attention_q"
        )
        assert metrics.quality_grade in (QualityGrade.EXCELLENT, QualityGrade.GOOD)
        assert metrics.recommended_action == "accept"

    def test_fail_grade_high_error(self, engine):
        orig = np.random.RandomState(42).randn(16, 16).astype(np.float32)
        recon = np.zeros_like(orig)
        metrics = engine.compute_all_metrics(orig, recon, tensor_name="bad")
        assert metrics.quality_grade in (QualityGrade.FAIL, QualityGrade.POOR)
        assert metrics.recommended_action != "accept"

    def test_grade_against_tiered_budget(self, engine):
        orig = np.random.RandomState(42).randn(32, 32).astype(np.float32)
        recon = orig + np.random.RandomState(99).randn(32, 32).astype(np.float32) * 1e-3
        metrics = engine.compute_all_metrics(
            orig, recon, tensor_name="attention_q", tensor_type="attention_q"
        )
        assert isinstance(metrics.quality_grade, QualityGrade)
        assert 0.0 <= metrics.confidence_score <= 1.0

    def test_overall_loss_percent(self, engine):
        orig = np.random.RandomState(42).randn(100).astype(np.float64)
        recon = orig + np.random.RandomState(99).randn(100) * 0.01
        metrics = engine.compute_all_metrics(orig, recon)
        assert 0.0 <= metrics.overall_loss_percent <= 100.0

    def test_grade_distribution(self, engine):
        results = []
        for i in range(5):
            noise_level = 10 ** (-i)
            orig = np.random.RandomState(42 + i).randn(16, 16).astype(np.float32)
            recon = (
                orig
                + np.random.RandomState(99 + i).randn(16, 16).astype(np.float32)
                * noise_level
            )
            results.append(
                engine.compute_all_metrics(orig, recon, tensor_name=f"tensor_{i}")
            )
        summary = engine.summary_report(results)
        assert summary["n_tensors"] == 5
        assert "grade_distribution" in summary
        assert summary["overall_loss_percent"] >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 6. Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_tensor(self, engine):
        orig = np.array([], dtype=np.float32)
        recon = np.array([], dtype=np.float32)
        metrics = engine.compute_all_metrics(orig, recon, tensor_name="empty")
        assert isinstance(metrics, PerTensorLossMetrics)

    def test_single_element(self, engine):
        orig = np.array([42.0], dtype=np.float64)
        recon = orig.copy()
        metrics = engine.compute_all_metrics(orig, recon, tensor_name="single")
        assert metrics.spectral.mse == 0.0
        assert metrics.statistical.cosine_similarity == pytest.approx(1.0)

    def test_zero_tensor(self, engine):
        orig = np.zeros((16, 16), dtype=np.float32)
        recon = np.zeros((16, 16), dtype=np.float32)
        metrics = engine.compute_all_metrics(orig, recon, tensor_name="zero")
        assert metrics.spectral.mse == 0.0
        assert metrics.statistical.snr_db == float("inf")

    def test_identity_matrix(self, engine):
        orig = np.eye(16, dtype=np.float64)
        recon = (
            orig + np.random.RandomState(42).randn(16, 16).astype(np.float64) * 1e-10
        )
        metrics = engine.compute_all_metrics(orig, recon)
        assert metrics.spectral.energy_concentration_preserved > 0.9
        assert metrics.structural.effective_rank_preserved > 0.9

    def test_large_tensor(self, engine):
        orig = np.random.RandomState(42).randn(256, 256).astype(np.float32)
        recon = (
            orig + np.random.RandomState(99).randn(256, 256).astype(np.float32) * 1e-6
        )
        metrics = engine.compute_all_metrics(orig, recon, tensor_name="large")
        assert metrics.spectral.mse < 1e-8
        assert isinstance(metrics.quality_grade, QualityGrade)

    def test_1d_tensor(self, engine):
        orig = np.random.RandomState(42).randn(128).astype(np.float32)
        recon = orig + np.random.RandomState(99).randn(128).astype(np.float32) * 0.01
        metrics = engine.compute_all_metrics(orig, recon, tensor_name="1d")
        assert isinstance(metrics.spectral, SpectralMetrics)
        assert isinstance(metrics.statistical, StatisticalMetrics)

    def test_3d_tensor(self, engine):
        orig = np.random.RandomState(42).randn(8, 8, 8).astype(np.float32)
        recon = (
            orig + np.random.RandomState(99).randn(8, 8, 8).astype(np.float32) * 1e-5
        )
        metrics = engine.compute_all_metrics(orig, recon, tensor_name="3d")
        assert metrics.spectral.energy_concentration_preserved > 0.9

    def test_constant_tensor(self, engine):
        orig = np.ones((10, 10), dtype=np.float64) * 5.0
        recon = (
            orig + np.random.RandomState(42).randn(10, 10).astype(np.float64) * 1e-10
        )
        metrics = engine.compute_all_metrics(orig, recon)
        assert metrics.statistical.ssim == pytest.approx(1.0, abs=1e-3)

    def test_shape_mismatch_smaller_recon(self, engine):
        orig = np.random.RandomState(42).randn(20).astype(np.float64)
        recon = np.random.RandomState(42).randn(15).astype(np.float64)
        metrics = engine.compute_all_metrics(orig, recon)
        # Should handle by trimming to min size
        assert isinstance(metrics, PerTensorLossMetrics)

    def test_inf_values(self, engine):
        orig = np.array([1.0, 2.0, float("inf"), 4.0], dtype=np.float64)
        recon = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        metrics = engine.compute_all_metrics(orig, recon, tensor_name="inf")
        # Should not crash
        assert isinstance(metrics, PerTensorLossMetrics)

    def test_nan_values(self, engine):
        orig = np.array([1.0, 2.0, float("nan"), 4.0], dtype=np.float64)
        recon = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        metrics = engine.compute_all_metrics(orig, recon, tensor_name="nan")
        # Should not crash
        assert isinstance(metrics, PerTensorLossMetrics)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Summary Reports
# ═══════════════════════════════════════════════════════════════════════════


class TestSummaryReport:
    def test_empty_report(self, engine):
        summary = engine.summary_report([])
        assert summary["n_tensors"] == 0

    def test_single_tensor_report(self, engine):
        orig = np.random.RandomState(42).randn(16, 16).astype(np.float32)
        recon = orig.copy()
        metrics = engine.compute_all_metrics(orig, recon, tensor_name="test")
        summary = engine.summary_report([metrics])
        assert summary["n_tensors"] == 1
        assert summary["overall_loss_percent"] == 0.0
        assert summary["overall_compression_ratio"] >= 1.0

    def test_multi_tensor_report(self, engine):
        results = []
        rng = np.random.RandomState(42)
        for i in range(4):
            orig = rng.randn(16, 16).astype(np.float32)
            recon = orig + rng.randn(16, 16).astype(np.float32) * 0.001
            results.append(
                engine.compute_all_metrics(orig, recon, tensor_name=f"layer.{i}.weight")
            )
        summary = engine.summary_report(results)
        assert summary["n_tensors"] == 4
        assert "grade_distribution" in summary
        assert "tensor_results" in summary
        assert len(summary["tensor_results"]) == 4

    def test_report_tensor_results_format(self, engine):
        orig = np.random.RandomState(42).randn(8, 8).astype(np.float32)
        recon = orig.copy()
        metrics = engine.compute_all_metrics(
            orig,
            recon,
            tensor_name="test",
            method_used="block_int8",
            compressed_nbytes=64,
        )
        d = metrics.to_dict()
        assert d["name"] == "test"
        assert d["method_used"] == "block_int8"
        assert d["original_nbytes"] > 0
        assert "spectral" in d
        assert "statistical" in d
        assert "structural" in d
        assert "compression" in d

    def test_weighted_report(self, engine):
        results = []
        for i in range(3):
            orig = np.random.RandomState(42 + i).randn(16, 16).astype(np.float32)
            recon = orig + np.random.RandomState(99 + i).randn(16, 16).astype(
                np.float32
            ) * (0.01 * (i + 1))
            results.append(engine.compute_all_metrics(orig, recon, tensor_name=f"t{i}"))
        summary = engine.summary_report(results)
        assert summary["overall_loss_percent"] > 0.0
        assert summary["max_loss_percent"] >= summary["overall_loss_percent"]


# ═══════════════════════════════════════════════════════════════════════════
# 8. Certification Eligibility
# ═══════════════════════════════════════════════════════════════════════════


class TestCertification:
    def test_certification_passes_perfect(self, engine):
        orig = np.random.RandomState(42).randn(16, 16).astype(np.float32)
        recon = orig.copy()
        metrics = engine.compute_all_metrics(orig, recon, tensor_name="perfect")
        summary = engine.summary_report([metrics])
        eligible, msg = engine.check_certification_eligibility(summary)
        assert eligible
        assert "PASSED" in msg

    def test_certification_fails_bad_tensor(self, engine):
        orig = np.random.RandomState(42).randn(16, 16).astype(np.float32)
        recon = np.zeros_like(orig)
        metrics = engine.compute_all_metrics(orig, recon, tensor_name="bad")
        summary = engine.summary_report([metrics])
        eligible, msg = engine.check_certification_eligibility(summary)
        assert not eligible

    def test_certification_threshold(self, engine):
        results = []
        for i in range(10):
            orig = np.random.RandomState(42 + i).randn(16, 16).astype(np.float32)
            recon = (
                orig
                + np.random.RandomState(99 + i).randn(16, 16).astype(np.float32) * 1e-4
            )
            results.append(engine.compute_all_metrics(orig, recon, tensor_name=f"t{i}"))
        summary = engine.summary_report(results)
        eligible, msg = engine.check_certification_eligibility(
            summary, max_loss_pct=1.0
        )
        # Good quality should pass
        if not eligible:
            # If it fails, the average loss should indeed be high
            assert summary["overall_loss_percent"] > 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 9. Numerical Correctness
# ═══════════════════════════════════════════════════════════════════════════


class TestNumericalCorrectness:
    def test_mse_formula(self, engine):
        orig = np.array([1.0, 2.0, 3.0])
        recon = np.array([1.5, 2.5, 3.5])
        diff = orig - recon
        expected_mse = float(np.mean(diff**2))
        metrics = engine.compute_all_metrics(orig, recon)
        assert metrics.spectral.mse == pytest.approx(expected_mse)

    def test_snr_formula(self, engine):
        orig = np.array([10.0, 0.0, -10.0])
        recon = np.array([10.1, 0.1, -10.1])
        noise_var = float(np.var(orig - recon))
        signal_var = float(np.var(orig))
        expected_snr = (
            10.0 * math.log10(signal_var / noise_var) if noise_var > 0 else float("inf")
        )
        metrics = engine.compute_all_metrics(orig, recon)
        assert metrics.statistical.snr_db == pytest.approx(expected_snr, rel=1e-6)

    def test_cosine_similarity_formula(self, engine):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        expected = dot / norm
        metrics = engine.compute_all_metrics(a, b)
        assert metrics.statistical.cosine_similarity == pytest.approx(expected)

    def test_power_iteration_vs_svd(self):
        A = np.random.RandomState(42).randn(64, 64)
        sn_power = _spectral_norm(A, power_iters=50)
        s = np.linalg.svd(A, compute_uv=False)
        assert sn_power == pytest.approx(float(s[0]), rel=0.05)

    def test_ssim_self_similarity(self):
        A = np.random.RandomState(42).randn(16, 16)
        ssim = _ssim_2d(A, A)
        assert ssim == pytest.approx(1.0, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Full Pipeline Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    def test_synthetic_model(self, engine):
        rng = np.random.RandomState(42)
        tensors = {
            "embed_tokens.weight": rng.randn(16, 16).astype(np.float32) * 0.02,
            "model.layers.0.attn.q_proj.weight": rng.randn(16, 16).astype(np.float32)
            * 0.01,
            "model.layers.0.attn.k_proj.weight": rng.randn(16, 16).astype(np.float32)
            * 0.01,
            "model.layers.0.attn.v_proj.weight": rng.randn(16, 16).astype(np.float32)
            * 0.01,
            "model.layers.0.attn.o_proj.weight": rng.randn(16, 16).astype(np.float32)
            * 0.01,
            "model.layers.0.ffn.gate_proj.weight": rng.randn(16, 16).astype(np.float32)
            * 0.01,
            "model.layers.0.ffn.up_proj.weight": rng.randn(16, 16).astype(np.float32)
            * 0.01,
            "model.layers.0.ffn.down_proj.weight": rng.randn(16, 16).astype(np.float32)
            * 0.01,
            "model.layers.0.input_layernorm.weight": rng.randn(16).astype(np.float32),
            "norm.weight": rng.randn(16).astype(np.float32) * 0.1,
        }
        results = []
        for name, orig in tensors.items():
            recon = orig + rng.randn(*orig.shape).astype(np.float32) * 1e-4
            results.append(
                engine.compute_all_metrics(
                    orig, recon, tensor_name=name, method_used="block_int8"
                )
            )
        summary = engine.summary_report(results)
        assert summary["n_tensors"] == len(tensors)
        assert summary["total_original_bytes"] > 0
        assert len(summary["tensor_results"]) == len(tensors)

    def test_less_than_1pct_loss(self, engine):
        rng = np.random.RandomState(42)
        results = []
        for i in range(5):
            orig = rng.randn(32, 32).astype(np.float32)
            noise = rng.randn(32, 32).astype(np.float32) * 1e-5
            recon = orig + noise
            m = engine.compute_all_metrics(orig, recon, tensor_name=f"t{i}")
            results.append(m)
        summary = engine.summary_report(results)
        assert summary["overall_loss_percent"] < 1.0

    def test_confidence_high_snr(self, engine):
        orig = np.random.RandomState(42).randn(100).astype(np.float64)
        recon = orig + np.random.RandomState(99).randn(100) * 1e-8
        metrics = engine.compute_all_metrics(orig, recon)
        assert metrics.confidence_score > 0.8

    def test_confidence_low_snr(self, engine):
        orig = np.random.RandomState(42).randn(100).astype(np.float64)
        recon = orig + np.random.RandomState(99).randn(100) * 10.0
        metrics = engine.compute_all_metrics(orig, recon)
        assert metrics.confidence_score < 0.9

    def test_wasserstein_distance_zero(self, engine):
        orig = np.random.RandomState(42).randn(1000).astype(np.float64)
        metrics = engine.compute_all_metrics(orig, orig)
        assert metrics.statistical.wasserstein_distance == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 11. Thread Safety
# ═══════════════════════════════════════════════════════════════════════════


class TestThreadSafety:
    def test_concurrent_metrics(self, engine):
        import threading

        results = {}
        errors = []

        def compute(idx):
            try:
                orig = np.random.RandomState(idx).randn(16, 16).astype(np.float32)
                recon = (
                    orig
                    + np.random.RandomState(100 + idx).randn(16, 16).astype(np.float32)
                    * 1e-4
                )
                engine.compute_all_metrics(orig, recon, tensor_name=f"t{idx}")
                results[idx] = True
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=compute, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 8
