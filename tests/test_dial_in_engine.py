"""Tests for CompressionDialInEngine — R&D dial-in automation."""

import gc
import json
import os
import sys
import tempfile
import time

import numpy as np
import pytest

sys.path.insert(0, ".")

from spectralstream.compression.world_model.dial_in_engine import (
    CompressionDialInEngine,
    DialInReport,
    DialInTensorProfile,
    MethodTestRecord,
    CascadeDiscoveryResult,
    SensitivityEntry,
    OptimalParamEntry,
    ModelRatioPlan,
    _make_synthetic_tensors,
)


@pytest.fixture(scope="module")
def engine():
    """Create a CompressionDialInEngine for testing."""
    eng = CompressionDialInEngine(
        target_ratio=400.0,
        max_error=0.01,
        max_workers=2,
    )
    return eng


@pytest.fixture(scope="module")
def synthetic_tensors():
    """Generate synthetic tensors for testing."""
    return _make_synthetic_tensors(
        seed=42,
        tensor_types=["attention_q", "ffn_gate", "norm", "embedding"],
    )


@pytest.fixture(scope="module")
def flat_tensors(synthetic_tensors):
    """Return flat tensor dict (name -> array) from synthetic data."""
    return {k: v[0] for k, v in synthetic_tensors.items()}


# ── Test: Model Scan ──────────────────────────────────────────────────────────


class TestModelScan:
    def test_scan_model_returns_profiles(self, engine, flat_tensors):
        profiles = engine.scan_model(flat_tensors)
        assert len(profiles) > 0
        for p in profiles:
            assert isinstance(p, DialInTensorProfile)
            assert p.name in flat_tensors
            assert p.nbytes > 0
            assert p.tensor_type in (
                "attention_q",
                "ffn_gate",
                "norm",
                "embedding",
                "weight",
                "ffn",
                "attention",
            )
            assert 0.0 <= p.sensitivity <= 1.0
            assert 0.0 <= p.compressibility <= 1.0

    def test_scan_model_all_metrics_present(self, engine, flat_tensors):
        profiles = engine.scan_model(flat_tensors)
        for p in profiles:
            assert p.effective_rank > 0.0
            assert p.entropy >= 0.0
            assert p.spectral_decay >= 0.0

    def test_scan_model_orders_by_size(self, engine, flat_tensors):
        profiles = engine.scan_model(flat_tensors)
        for i in range(len(profiles) - 1):
            assert profiles[i].nbytes >= profiles[i + 1].nbytes


# ── Test: Tensor Classification ────────────────────────────────────────────────


class TestTensorClassification:
    def test_classify_attention_q(self, engine):
        assert (
            engine._classify_tensor("model.layers.0.self_attn.q_proj.weight")
            == "attention_q"
        )
        assert engine._classify_tensor("query.weight") == "attention_q"

    def test_classify_attention_k(self, engine):
        assert (
            engine._classify_tensor("model.layers.0.self_attn.k_proj.weight")
            == "attention_k"
        )
        assert engine._classify_tensor("key.weight") == "attention_k"

    def test_classify_attention_v(self, engine):
        assert (
            engine._classify_tensor("model.layers.0.self_attn.v_proj.weight")
            == "attention_v"
        )
        assert engine._classify_tensor("value.weight") == "attention_v"

    def test_classify_attention_o(self, engine):
        assert (
            engine._classify_tensor("model.layers.0.self_attn.o_proj.weight")
            == "attention_o"
        )

    def test_classify_ffn_gate(self, engine):
        assert (
            engine._classify_tensor("model.layers.0.mlp.gate_proj.weight") == "ffn_gate"
        )

    def test_classify_ffn_up(self, engine):
        assert engine._classify_tensor("model.layers.0.mlp.up_proj.weight") == "ffn_up"

    def test_classify_ffn_down(self, engine):
        assert (
            engine._classify_tensor("model.layers.0.mlp.down_proj.weight") == "ffn_down"
        )

    def test_classify_embedding(self, engine):
        assert engine._classify_tensor("model.embed_tokens.weight") == "embedding"

    def test_classify_norm(self, engine):
        assert (
            engine._classify_tensor("model.layers.0.input_layernorm.weight") == "norm"
        )
        assert engine._classify_tensor("model.norm.weight") == "norm"


# ── Test: Method Profiling ────────────────────────────────────────────────────


def _make_tiny_tensors():
    """Small tensors for fast method profiling (avoids engine hangs)."""
    rng = np.random.RandomState(42)
    return {
        "attention_q": (rng.randn(64, 64).astype(np.float32), "attention_q"),
        "ffn_gate": (rng.randn(64, 128).astype(np.float32), "ffn_gate"),
    }


class TestMethodProfiling:
    def test_discover_methods_returns_dict(self, engine):
        methods = engine.discover_methods()
        assert isinstance(methods, dict)
        assert len(methods) > 0

    def test_profile_methods_returns_by_type(self, engine):
        tiny = _make_tiny_tensors()
        profiles = engine.profile_methods(
            tiny,
            max_methods=3,
        )
        assert isinstance(profiles, dict)

    def test_profile_method_records_have_all_fields(self, engine):
        tiny = _make_tiny_tensors()
        profiles = engine.profile_methods(
            tiny,
            max_methods=3,
        )
        for ttype, records in profiles.items():
            for r in records[:2]:
                assert isinstance(r, MethodTestRecord)
                assert r.method_name
                assert r.ratio >= 1.0
                assert r.error >= 0.0
                assert r.tier in (1, 2, 3, 4, 5)

    def test_find_top_methods_per_type(self, engine):
        tiny = _make_tiny_tensors()
        method_profiles = engine.profile_methods(tiny, max_methods=3)
        top = engine.find_top_methods_per_type(method_profiles, top_n=3)
        assert isinstance(top, dict)
        for ttype, methods in top.items():
            for m in methods:
                assert "method" in m
                assert "ratio" in m
                assert "error" in m
                assert "score" in m
                assert m["score"] >= 0.0

    def test_find_method_complementarity(self, engine):
        tiny = _make_tiny_tensors()
        method_profiles = engine.profile_methods(tiny, max_methods=3)
        pairs = engine.find_method_complementarity(method_profiles)
        assert isinstance(pairs, dict)


# ── Test: Cascade Discovery ───────────────────────────────────────────────────


class TestCascadeDiscovery:
    def test_discover_cascades_returns_results(self, engine):
        tiny = _make_tiny_tensors()
        cascades = engine.discover_cascades(tiny, exhaustive=False)
        assert isinstance(cascades, list)

    def test_discover_cascades_exhaustive(self, engine):
        tiny = _make_tiny_tensors()
        cascades = engine.discover_cascades(tiny, exhaustive=True)
        assert isinstance(cascades, list)

    def test_best_cascade_per_type(self, engine):
        tiny = _make_tiny_tensors()
        cascades = engine.discover_cascades(tiny, exhaustive=False)
        best: dict = {}
        for cd in cascades:
            if (
                cd.tensor_type not in best
                or cd.composite_score > best[cd.tensor_type].composite_score
            ):
                best[cd.tensor_type] = cd
        if best:
            for tt, cd in best.items():
                assert cd.total_ratio > 0
                assert cd.composite_score > 0


# ── Test: Sensitivity Analysis ────────────────────────────────────────────────


class TestSensitivityAnalysis:
    def test_analyze_sensitivity_returns_map(self, engine, flat_tensors):
        profiles = engine.scan_model(flat_tensors)
        sens = engine.analyze_sensitivity(profiles)
        assert isinstance(sens, dict)
        for name, entry in sens.items():
            assert isinstance(entry, SensitivityEntry)
            assert entry.tier in ("critical", "high", "medium", "low")

    def test_sensitivity_tiers_assigned(self, engine, flat_tensors):
        profiles = engine.scan_model(flat_tensors)
        sens = engine.analyze_sensitivity(profiles)
        tiers = set(s.tier for s in sens.values())
        assert len(tiers) > 0

    def test_critical_tensors_identified(self, engine, flat_tensors):
        profiles = engine.scan_model(flat_tensors)
        sens = engine.analyze_sensitivity(profiles)
        critical = engine.get_critical_tensors(sens)
        assert isinstance(critical, list)

    def test_robust_tensors_identified(self, engine, flat_tensors):
        profiles = engine.scan_model(flat_tensors)
        sens = engine.analyze_sensitivity(profiles)
        robust = engine.get_robust_tensors(sens)
        assert isinstance(robust, list)

    def test_critical_and_robust_are_disjoint(self, engine, flat_tensors):
        profiles = engine.scan_model(flat_tensors)
        sens = engine.analyze_sensitivity(profiles)
        critical = set(engine.get_critical_tensors(sens))
        robust = set(engine.get_robust_tensors(sens))
        assert critical.isdisjoint(robust)

    def test_recommended_max_error(self, engine):
        """Critical tensors get stricter error budgets."""
        from unittest.mock import MagicMock

        p_critical = MagicMock(
            spec=DialInTensorProfile, sensitivity=0.8, nbytes=1000000, shape=(100, 100)
        )
        p_robust = MagicMock(
            spec=DialInTensorProfile, sensitivity=0.1, nbytes=1000000, shape=(100, 100)
        )

        err_critical = engine._recommend_max_error("critical", p_critical)
        err_robust = engine._recommend_max_error("low", p_robust)
        assert err_critical < err_robust


# ── Test: Parameter Tuning ────────────────────────────────────────────────────


class TestParameterTuning:
    def test_tune_parameters_returns_entries(self, engine, synthetic_tensors):
        tuned = engine.tune_parameters(
            synthetic_tensors,
            param_grid={
                "block_int8": {"block_size": [32, 128]},
                "svd_compress": {"rank": [8, 32]},
            },
        )
        assert isinstance(tuned, dict)
        if tuned:
            for ttype, entries in tuned.items():
                for e in entries:
                    assert isinstance(e, OptimalParamEntry)
                    assert e.score > 0.0

    def test_tune_parameters_empty_grid(self, engine, synthetic_tensors):
        tuned = engine.tune_parameters(
            synthetic_tensors,
            param_grid={},
        )
        assert isinstance(tuned, dict)

    def test_tune_parameters_sorts_by_score(self, engine, synthetic_tensors):
        tuned = engine.tune_parameters(
            synthetic_tensors,
            param_grid={
                "block_int8": {"block_size": [32, 128, 512]},
            },
        )
        for ttype, entries in tuned.items():
            scores = [e.score for e in entries]
            assert scores == sorted(scores, reverse=True)


# ── Test: Ratio Optimization ──────────────────────────────────────────────────


class TestRatioOptimization:
    def test_optimize_ratio_returns_plan(self, engine, flat_tensors):
        profiles = engine.scan_model(flat_tensors)
        sens = engine.analyze_sensitivity(profiles)
        top = {
            p.tensor_type: [
                {"method": "block_int8", "ratio": 4.0, "error": 0.01, "score": 0.5}
            ]
            for p in profiles
        }
        plan = engine.optimize_ratio(profiles, sens, top)
        assert isinstance(plan, ModelRatioPlan)
        assert plan.total_original_bytes > 0
        assert plan.overall_ratio > 0

    def test_optimize_ratio_all_tensors_in_plan(self, engine, flat_tensors):
        profiles = engine.scan_model(flat_tensors)
        sens = engine.analyze_sensitivity(profiles)
        top = {
            p.tensor_type: [
                {"method": "block_int8", "ratio": 4.0, "error": 0.01, "score": 0.5}
            ]
            for p in profiles
        }
        plan = engine.optimize_ratio(profiles, sens, top)
        for p in profiles:
            assert p.name in plan.tensor_plans

    def test_optimize_ratio_plan_has_expected_keys(self, engine, flat_tensors):
        profiles = engine.scan_model(flat_tensors)
        sens = engine.analyze_sensitivity(profiles)
        top = {
            p.tensor_type: [
                {"method": "block_int4", "ratio": 8.0, "error": 0.02, "score": 0.6}
            ]
            for p in profiles
        }
        plan = engine.optimize_ratio(profiles, sens, top)
        for name, tp in plan.tensor_plans.items():
            assert "method" in tp
            assert "target_ratio" in tp
            assert "max_error" in tp
            assert "original_bytes" in tp
            assert "estimated_compressed" in tp


# ── Test: Metrics Helpers ─────────────────────────────────────────────────────


class TestMetrics:
    def test_compute_entropy(self, engine):
        flat = np.random.RandomState(42).randn(10000).astype(np.float64)
        ent = engine._compute_entropy(flat)
        assert ent > 0.0

    def test_compute_entropy_uniform(self, engine):
        flat = np.ones(1000, dtype=np.float64)
        ent = engine._compute_entropy(flat)
        assert ent >= 0.0

    def test_compute_effective_rank(self, engine):
        tensor = np.random.RandomState(42).randn(64, 64).astype(np.float32)
        rank = engine._compute_effective_rank(tensor)
        assert rank > 0.0
        assert rank <= 64.0

    def test_compute_effective_rank_1d(self, engine):
        tensor = np.random.RandomState(42).randn(64).astype(np.float32)
        rank = engine._compute_effective_rank(tensor)
        assert rank > 0.0

    def test_compute_spectral_decay(self, engine):
        tensor = np.random.RandomState(42).randn(64, 64).astype(np.float32)
        decay = engine._compute_spectral_decay(tensor)
        assert decay >= 0.0

    def test_compute_snr(self, engine):
        original = np.random.RandomState(42).randn(1000).astype(np.float32)
        recon = (
            original + np.random.RandomState(1).randn(1000).astype(np.float32) * 0.01
        )
        snr = engine._compute_snr(original, recon)
        assert snr > 0.0
        assert snr != float("inf")

    def test_compute_snr_identical(self, engine):
        original = np.random.RandomState(42).randn(1000).astype(np.float32)
        snr = engine._compute_snr(original, original)
        assert snr == float("inf") or snr > 100.0

    def test_estimate_sensitivity(self, engine):
        tensor = np.random.RandomState(42).randn(256, 256).astype(np.float32)
        sens = engine._estimate_sensitivity(tensor, eff_rank=16.0)
        assert 0.0 <= sens <= 1.0

    def test_score_compressibility(self, engine):
        tensor = np.random.RandomState(42).randn(256, 256).astype(np.float32)
        score = engine._score_compressibility(
            tensor, eff_rank=16.0, entropy=4.0, sensitivity=0.3
        )
        assert 0.0 <= score <= 1.0


# ── Test: Full Pipeline ───────────────────────────────────────────────────────


class TestFullPipeline:
    @pytest.mark.timeout(90)
    def test_dial_in_minimal_returns_report(self, engine):
        """Run dial-in with minimal tensor types and quick mode."""
        tensor_data = _make_synthetic_tensors(
            seed=42,
            tensor_types=["attention_q"],
        )
        tensors = {k: v[0] for k, v in tensor_data.items()}
        report = engine.dial_in(
            model_name="test_minimal",
            tensors=tensors,
            quick=True,
        )
        assert isinstance(report, DialInReport)
        assert report.n_tensors == 1
        assert report.avg_ratio > 0
        assert report.elapsed_seconds > 0
        assert report.weighted_grade != ""

    @pytest.mark.timeout(90)
    def test_dial_in_two_types(self, engine):
        tensor_data = _make_synthetic_tensors(
            seed=42,
            tensor_types=["attention_q", "ffn_gate"],
        )
        tensors = {k: v[0] for k, v in tensor_data.items()}
        report = engine.dial_in(
            model_name="test_two_types",
            tensors=tensors,
            quick=True,
        )
        assert report.n_tensors == 2
        assert report.n_tensor_types >= 1

    @pytest.mark.timeout(90)
    def test_dial_in_report_has_all_sections(self, engine):
        tensor_data = _make_synthetic_tensors(
            seed=42,
            tensor_types=["attention_q"],
        )
        tensors = {k: v[0] for k, v in tensor_data.items()}
        report = engine.dial_in(
            model_name="test_sections",
            tensors=tensors,
            quick=True,
        )
        assert report.method_profiles is not None
        assert report.sensitivity_map is not None
        assert report.ratio_plan is not None
        assert report.weighted_grade != ""

    @pytest.mark.timeout(90)
    def test_dial_in_saves_report(self, engine):
        tensor_data = _make_synthetic_tensors(
            seed=42,
            tensor_types=["attention_q"],
        )
        tensors = {k: v[0] for k, v in tensor_data.items()}
        with tempfile.TemporaryDirectory() as tmpdir:
            report = engine.dial_in(
                model_name="test_save",
                tensors=tensors,
                quick=True,
                output_dir=tmpdir,
            )
            json_path = os.path.join(tmpdir, "dial_in_report.json")
            html_path = os.path.join(tmpdir, "dial_in_report.html")
            txt_path = os.path.join(tmpdir, "dial_in_report.txt")
            assert os.path.exists(json_path), f"JSON report not found at {json_path}"
            assert os.path.exists(html_path), f"HTML report not found at {html_path}"
            assert os.path.exists(txt_path), f"TXT report not found at {txt_path}"
            with open(json_path) as f:
                data = json.load(f)
            assert "model_name" in data
            assert "avg_ratio" in data
            assert "avg_error" in data

    @pytest.mark.timeout(90)
    def test_dial_in_grade_assignment(self, engine):
        tensor_data = _make_synthetic_tensors(
            seed=42,
            tensor_types=["attention_q"],
        )
        tensors = {k: v[0] for k, v in tensor_data.items()}
        report = engine.dial_in(
            model_name="test_grade",
            tensors=tensors,
            quick=True,
        )
        assert report.weighted_grade in (
            "EXCELLENT",
            "GOOD",
            "FAIR",
            "POOR",
            "FAIL",
            "UNKNOWN",
        )


# ── Test: Report Output ───────────────────────────────────────────────────────


class TestReport:
    def test_report_summary_text(self, engine):
        report = engine.dial_in(
            model_name="test_summary",
            quick=True,
        )
        text = report.summary_text()
        assert isinstance(text, str)
        assert len(text) > 100
        assert "Dial-In Engine" in text

    def test_report_to_html(self, engine):
        report = engine.dial_in(
            model_name="test_html",
            quick=True,
        )
        html = report.to_html()
        assert isinstance(html, str)
        assert "<html" in html
        assert "Dial-In Report" in html

    def test_report_to_dict(self, engine):
        report = engine.dial_in(
            model_name="test_dict",
            quick=True,
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "model_name" in d
        assert "avg_ratio" in d
        assert "avg_error" in d
        assert "weighted_grade" in d

    def test_report_save_formats(self, engine):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = engine.dial_in(
                model_name="test_formats",
                quick=True,
            )
            saved = report.save(tmpdir, formats=["json", "txt"])
            assert "json" in saved
            assert "txt" in saved


# ── Test: Synthetic Data Generator ────────────────────────────────────────────


class TestSyntheticData:
    def test_make_synthetic_tensors_returns_dict(self):
        tensors = _make_synthetic_tensors()
        assert isinstance(tensors, dict)
        assert len(tensors) > 0

    def test_make_synthetic_tensors_shapes(self):
        tensors = _make_synthetic_tensors(tensor_types=["attention_q", "ffn_gate"])
        for name, (tensor, ttype) in tensors.items():
            assert isinstance(tensor, np.ndarray)
            assert tensor.dtype == np.float32
            assert tensor.size > 0

    def test_make_synthetic_tensors_seed_reproducibility(self):
        t1 = _make_synthetic_tensors(seed=42)
        t2 = _make_synthetic_tensors(seed=42)
        for key in t1:
            np.testing.assert_array_equal(t1[key][0], t2[key][0])

    def test_make_synthetic_tensors_limited_types(self):
        tensors = _make_synthetic_tensors(
            tensor_types=["attention_q", "attention_k", "attention_v"]
        )
        assert set(tensors.keys()) == {"attention_q", "attention_k", "attention_v"}
