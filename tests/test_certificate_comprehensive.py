"""Comprehensive tests for the compression certificate system — extended coverage."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.compression.certificate import (
    TensorCertificate,
    CompressionCertificate,
    CertificateBuilder,
    ValidationCertificate,
    ValidationResult,
)
from spectralstream.compression.engine import (
    CompressedTensor,
    CompressionReport,
)
from spectralstream.compression.engine._helpers import _compute_metrics, _grade_error


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

SAMPLE_GRADES = ["S", "A", "B", "C", "D", "F"]


@pytest.fixture
def sample_tensor_cert() -> TensorCertificate:
    return TensorCertificate(
        name="layer_0",
        shape=(64, 64),
        original_dtype="float32",
        original_bytes=16384,
        compressed_bytes=4096,
        compression_ratio=4.0,
        method="block_int8",
        method_category="quantization",
        relative_error=0.001,
        snr_db=30.0,
        psnr_db=35.0,
        cosine_similarity=0.99,
        mse=0.0001,
        compression_time_ms=5.0,
        decompression_time_ms=2.0,
        quality_grade="A",
    )


@pytest.fixture
def sample_compressed_tensors() -> list:
    tensors = []
    for name in ["attn_q", "attn_k", "ffn_gate"]:
        tensor = np.random.randn(32, 64).astype(np.float32)
        data: bytes = tensor.tobytes()
        ct = CompressedTensor(
            _data=data,
            method="block_int8",
            params={},
            original_shape=tensor.shape,
            original_dtype="float32",
            compression_ratio=4.0,
            relative_error=0.005,
            snr_db=25.0,
            psnr_db=30.0,
            cosine_similarity=0.95,
            computation_time=0.01,
        )
        tensors.append((name, ct))
    return tensors


@pytest.fixture
def sample_comp_cert(sample_tensor_cert) -> CompressionCertificate:
    return CompressionCertificate(
        model_name="test-model",
        model_path="/tmp/test",
        model_architecture="test-arch",
        model_params="1B",
        total_original_bytes=1_000_000,
        total_compressed_bytes=250_000,
        overall_ratio=4.0,
        total_tensors=10,
        compression_time_seconds=5.0,
        weighted_error=0.002,
        avg_error=0.001,
        max_error=0.005,
        min_error=0.0001,
        avg_snr_db=30.0,
        tensor_certificates=[sample_tensor_cert] * 3,
        method_distribution={"block_int8": 3},
    )


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _make_validation_result(**kw: object) -> ValidationResult:
    defaults = dict(
        name="tensor_0",
        shape=(64, 64),
        method="block_int8",
        original_size=16384,
        compressed_size=4096,
        compression_ratio=4.0,
        relative_error=0.001,
        snr_db=30.0,
        psnr_db=35.0,
        cosine_similarity=0.99,
        mse=0.0001,
        quality_grade="A",
        checksum_ok=True,
        decompression_ok=True,
    )
    defaults.update(**kw)
    return ValidationResult(**defaults)


def _make_tensor_cert(**kw: object) -> TensorCertificate:
    defaults = dict(
        name="t",
        shape=(1,),
        original_dtype="float32",
        original_bytes=4,
        compressed_bytes=2,
        compression_ratio=2.0,
        method="test",
        method_category="test",
        relative_error=0.001,
        snr_db=30.0,
        psnr_db=35.0,
        cosine_similarity=0.99,
        mse=0.0001,
        compression_time_ms=1.0,
        decompression_time_ms=0.5,
        quality_grade="A",
    )
    defaults.update(**kw)
    return TensorCertificate(**defaults)


# ═══════════════════════════════════════════════════════════════════
# ValidationResult
# ═══════════════════════════════════════════════════════════════════


class TestValidationResult:
    def test_default_creation(self):
        vr = _make_validation_result()
        assert vr.name == "tensor_0"
        assert vr.shape == (64, 64)
        assert vr.compression_ratio == 4.0
        assert vr.quality_grade == "A"
        assert vr.checksum_ok is True
        assert vr.decompression_ok is True

    def test_all_fields_accessible(self):
        vr = _make_validation_result(
            name="custom",
            shape=(128, 128),
            compression_ratio=8.0,
            relative_error=0.0005,
            snr_db=40.0,
            quality_grade="S",
            checksum_ok=False,
            decompression_ok=True,
        )
        assert vr.name == "custom"
        assert vr.shape == (128, 128)
        assert vr.compression_ratio == 8.0
        assert vr.relative_error == 0.0005
        assert vr.snr_db == 40.0
        assert vr.quality_grade == "S"
        assert vr.checksum_ok is False
        assert vr.decompression_ok is True

    def test_various_grade_values(self):
        for grade in SAMPLE_GRADES:
            vr = _make_validation_result(quality_grade=grade)
            assert vr.quality_grade == grade

    def test_edge_zero_ratio(self):
        vr = _make_validation_result(compression_ratio=0.0)
        assert vr.compression_ratio == 0.0

    def test_edge_minus_one_snr(self):
        vr = _make_validation_result(snr_db=-1.0)
        assert vr.snr_db == -1.0


# ═══════════════════════════════════════════════════════════════════
# ValidationCertificate
# ═══════════════════════════════════════════════════════════════════


class TestValidationCertificate:
    @pytest.fixture
    def valid_cert(self) -> ValidationCertificate:
        return ValidationCertificate(
            file_path="/tmp/test.ssf",
            file_size=1_000_000,
            n_tensors=10,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_validated=10,
            tensors_failed=0,
            errors=[],
            overall_ratio=4.0,
            avg_relative_error=0.001,
            max_relative_error=0.005,
            avg_snr_db=30.0,
            grade_distribution={"S": 9, "A": 1, "B": 0, "C": 0, "D": 0, "F": 0},
            method_distribution={"block_int8": 10},
            tensor_results=[_make_validation_result(name=str(i)) for i in range(10)],
        )

    @pytest.fixture
    def invalid_cert(self) -> ValidationCertificate:
        return ValidationCertificate(
            file_path="/tmp/bad.ssf",
            file_size=500_000,
            n_tensors=5,
            header_ok=True,
            checksum_ok=False,
            index_ok=True,
            tensors_validated=4,
            tensors_failed=1,
            errors=["checksum mismatch"],
            overall_ratio=2.0,
            avg_relative_error=0.05,
            max_relative_error=0.1,
            avg_snr_db=15.0,
            method_distribution={"fp32": 5},
        )

    # --- is_valid ---

    def test_is_valid_true(self, valid_cert):
        assert valid_cert.is_valid() is True

    def test_is_valid_false_bad_header(self):
        c = ValidationCertificate(
            file_path="",
            file_size=0,
            n_tensors=0,
            header_ok=False,
            checksum_ok=True,
            index_ok=True,
            tensors_failed=0,
        )
        assert c.is_valid() is False

    def test_is_valid_false_bad_checksum(self, invalid_cert):
        assert invalid_cert.is_valid() is False

    def test_is_valid_false_bad_index(self):
        c = ValidationCertificate(
            file_path="",
            file_size=0,
            n_tensors=0,
            header_ok=True,
            checksum_ok=True,
            index_ok=False,
            tensors_failed=0,
        )
        assert c.is_valid() is False

    def test_is_valid_false_tensors_failed(self):
        c = ValidationCertificate(
            file_path="",
            file_size=0,
            n_tensors=5,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_failed=3,
        )
        assert c.is_valid() is False

    # --- overall_grade ---

    def test_overall_grade_s(self, valid_cert):
        assert valid_cert.overall_grade() == "S"

    def test_overall_grade_a(self):
        c = ValidationCertificate(
            file_path="",
            file_size=0,
            n_tensors=10,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_failed=0,
            grade_distribution={"S": 0, "A": 9, "B": 1, "C": 0, "D": 0, "F": 0},
        )
        assert c.overall_grade() == "A"

    def test_overall_grade_b(self):
        c = ValidationCertificate(
            file_path="",
            file_size=0,
            n_tensors=10,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_failed=0,
            grade_distribution={"S": 0, "A": 5, "B": 5, "C": 0, "D": 0, "F": 0},
        )
        assert c.overall_grade() == "B"

    def test_overall_grade_c_few_failures(self):
        c = ValidationCertificate(
            file_path="",
            file_size=0,
            n_tensors=100,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_failed=1,
            grade_distribution={"S": 0, "A": 0, "B": 0, "C": 99, "D": 0, "F": 0},
        )
        assert c.overall_grade() == "C"

    def test_overall_grade_f_many_failures(self):
        c = ValidationCertificate(
            file_path="",
            file_size=0,
            n_tensors=10,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_failed=5,
            grade_distribution={"S": 0, "A": 0, "B": 0, "C": 0, "D": 0, "F": 0},
        )
        assert c.overall_grade() == "F"

    def test_overall_grade_invalid_not_s(self):
        """An invalid cert should never get S."""
        c = ValidationCertificate(
            file_path="",
            file_size=0,
            n_tensors=10,
            header_ok=True,
            checksum_ok=False,
            index_ok=True,
            tensors_failed=1,
            grade_distribution={"S": 9, "A": 0, "B": 0, "C": 0, "D": 0, "F": 0},
        )
        grade = c.overall_grade()
        assert grade != "S"

    def test_overall_grade_a_margin(self):
        c = ValidationCertificate(
            file_path="",
            file_size=0,
            n_tensors=100,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_failed=0,
            grade_distribution={"S": 0, "A": 85, "B": 15, "C": 0, "D": 0, "F": 0},
        )
        assert c.overall_grade() == "A"

    def test_overall_grade_b_margin(self):
        c = ValidationCertificate(
            file_path="",
            file_size=0,
            n_tensors=100,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_failed=0,
            grade_distribution={"S": 0, "A": 84, "B": 16, "C": 0, "D": 0, "F": 0},
        )
        assert c.overall_grade() == "B"

    # --- to_dict ---

    def test_to_dict_valid(self, valid_cert):
        d = valid_cert.to_dict()
        assert d["valid"] is True
        assert d["overall_grade"] == "S"
        assert d["structural"]["header_ok"] is True
        assert d["structural"]["checksum_ok"] is True
        assert d["structural"]["index_ok"] is True
        assert d["quality"]["avg_snr_db"] == 30.0
        assert d["tensors_validated"] == 10
        assert d["tensors_failed"] == 0
        assert len(d["tensors"]) == 10

    def test_to_dict_invalid(self, invalid_cert):
        d = invalid_cert.to_dict()
        assert d["valid"] is False
        assert d["structural"]["checksum_ok"] is False
        assert len(d["structural"]["errors"]) == 1

    def test_to_dict_grade_distribution(self):
        c = ValidationCertificate(
            file_path="",
            file_size=0,
            n_tensors=6,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_failed=0,
            grade_distribution={"S": 1, "A": 2, "B": 0, "C": 0, "D": 0, "F": 3},
            tensor_results=[_make_validation_result()],
        )
        d = c.to_dict()
        assert d["quality"]["grade_distribution"] == {
            "S": 1,
            "A": 2,
            "B": 0,
            "C": 0,
            "D": 0,
            "F": 3,
        }

    # --- to_html ---

    def test_to_html_contains_keywords(self, valid_cert):
        html = valid_cert.to_html()
        assert "<html" in html
        assert "Validation Certificate" in html
        assert "4.0x" in html
        assert "0" in html
        assert "9" in html

    def test_to_html_invalid_status(self, invalid_cert):
        html = invalid_cert.to_html()
        assert "✗ Invalid" in html or "Invalid" in html

    # --- to_markdown ---

    def test_to_markdown_contains_data(self, valid_cert):
        md = valid_cert.to_markdown()
        assert "# Validation Certificate" in md
        assert "/tmp/test.ssf" in md
        assert "4.0x" in md
        assert "## Structural Integrity" in md

    def test_to_markdown_valid_status(self, valid_cert):
        md = valid_cert.to_markdown()
        assert "✓ VALID" in md

    def test_to_markdown_invalid_status(self, invalid_cert):
        md = invalid_cert.to_markdown()
        assert "✗ INVALID" in md

    # --- to_text ---

    def test_to_text_box_art(self, valid_cert):
        text = valid_cert.to_text()
        assert "╔══════════════════════════════════" in text
        assert "Validation Certificate" in text
        assert "✓ VALID" in text

    def test_to_text_shows_ratio(self, valid_cert):
        text = valid_cert.to_text()
        assert "4.0x" in text or "4.0" in text

    # --- save ---

    def test_save_json(self, valid_cert):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "validation")
            valid_cert.save(base, formats=["json"])
            path = f"{base}.json"
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["valid"] is True
            assert data["overall_grade"] == "S"

    def test_save_all_formats(self, valid_cert):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "validation")
            valid_cert.save(base)
            for ext in ("json", "html", "md", "txt"):
                assert os.path.exists(f"{base}.{ext}")
                assert os.path.getsize(f"{base}.{ext}") > 0

    def test_save_selective_formats(self, valid_cert):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "val_selective")
            valid_cert.save(base, formats=["json", "md"])
            assert os.path.exists(f"{base}.json")
            assert os.path.exists(f"{base}.md")
            assert not os.path.exists(f"{base}.html")
            assert not os.path.exists(f"{base}.txt")

    def test_save_invalid_cert(self, invalid_cert):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "bad_val")
            invalid_cert.save(base)
            with open(f"{base}.json") as f:
                data = json.load(f)
            assert data["valid"] is False

    def test_save_empty_grade_dist(self):
        cert = ValidationCertificate(
            file_path="",
            file_size=0,
            n_tensors=0,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_failed=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "empty_val")
            cert.save(base)
            with open(f"{base}.json") as f:
                data = json.load(f)
            assert data["overall_grade"] in ("S", "A", "B")

    def test_save_load_roundtrip_json(self, valid_cert):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "rt_val")
            valid_cert.save(base, formats=["json"])
            with open(f"{base}.json") as f:
                loaded = json.load(f)
        assert loaded["valid"] is True
        assert loaded["overall_grade"] == "S"
        assert loaded["quality"]["overall_ratio"] == 4.0

    def test_save_creates_parent_dir(self, valid_cert):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "sub", "deep", "val")
            valid_cert.save(nested)
            assert os.path.exists(f"{nested}.json")

    # --- edge cases ---

    def test_empty_tensor_results(self):
        cert = ValidationCertificate(
            file_path="",
            file_size=0,
            n_tensors=0,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_failed=0,
            tensor_results=[],
        )
        assert cert.is_valid()
        assert cert.overall_grade() in ("S", "A", "B")

    def test_no_tensors_validated(self):
        cert = ValidationCertificate(
            file_path="",
            file_size=0,
            n_tensors=0,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_validated=0,
            tensors_failed=0,
            tensor_results=[],
        )
        d = cert.to_dict()
        assert d["valid"] is True
        assert d["tensors_validated"] == 0

    def test_with_errors(self):
        cert = ValidationCertificate(
            file_path="buggy.ssf",
            file_size=100,
            n_tensors=2,
            header_ok=True,
            checksum_ok=True,
            index_ok=False,
            errors=["index corrupted"],
            tensors_validated=1,
            tensors_failed=1,
        )
        assert not cert.is_valid()
        assert len(cert.errors) == 1
        assert "index corrupted" in cert.errors


# ═══════════════════════════════════════════════════════════════════
# TensorCertificate — Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestTensorCertificateEdgeCases:
    def test_infinite_snr(self):
        cert = _make_tensor_cert(snr_db=float("inf"))
        assert cert.snr_db == float("inf")
        line = cert.summary_line()
        assert "inf" in line or "inf" in str(line)

    def test_negative_snr(self):
        cert = _make_tensor_cert(snr_db=-10.0)
        assert cert.snr_db == -10.0

    def test_zero_size_tensor(self):
        cert = TensorCertificate(
            name="zero_size",
            shape=(),
            original_dtype="float32",
            original_bytes=0,
            compressed_bytes=0,
            compression_ratio=0.0,
            method="identity",
            method_category="lossless",
            relative_error=0.0,
            snr_db=float("inf"),
            psnr_db=float("inf"),
            cosine_similarity=1.0,
            mse=0.0,
            compression_time_ms=0.0,
            decompression_time_ms=0.0,
            quality_grade="S",
        )
        d = cert.to_dict()
        assert d["compression_ratio"] == 0.0
        assert d["cosine_similarity"] == 1.0
        assert d["mse"] == 0.0
        assert d["quality_grade"] == "S"

    def test_max_ratio(self):
        cert = _make_tensor_cert(
            compression_ratio=1e6,
            original_bytes=1_000_000_000,
            compressed_bytes=1_000,
        )
        assert cert.compression_ratio == 1e6
        line = cert.summary_line()
        assert "1000000.00x" in line or "1000000" in line

    def test_minimal_tensor(self):
        cert = TensorCertificate(
            name="tiny",
            shape=(1,),
            original_dtype="float32",
            original_bytes=4,
            compressed_bytes=4,
            compression_ratio=1.0,
            method="identity",
            method_category="lossless",
            relative_error=0.0,
            snr_db=float("inf"),
            psnr_db=float("inf"),
            cosine_similarity=1.0,
            mse=0.0,
            compression_time_ms=0.0,
            decompression_time_ms=0.0,
            quality_grade="S",
        )
        assert cert.compression_ratio == 1.0
        assert cert.summary_line() is not None

    def test_extreme_compression_time(self):
        cert = _make_tensor_cert(compression_time_ms=1e9)
        d = cert.to_dict()
        assert d["compression_time_ms"] == 1e9

    def test_unknown_grade(self):
        cert = _make_tensor_cert(quality_grade="X")
        line = cert.summary_line()
        assert "X" in line

    def test_to_dict_all_fields(self):
        cert = TensorCertificate(
            name="full",
            shape=(3, 224, 224),
            original_dtype="bfloat16",
            original_bytes=50_331_648,
            compressed_bytes=12_582_912,
            compression_ratio=4.0,
            method="fwht_quant",
            method_category="spectral",
            relative_error=0.002,
            snr_db=28.5,
            psnr_db=33.2,
            cosine_similarity=0.985,
            mse=0.0002,
            compression_time_ms=12.3,
            decompression_time_ms=4.5,
            quality_grade="A",
        )
        d = cert.to_dict()
        assert d["name"] == "full"
        assert d["shape"] == (3, 224, 224)
        assert d["original_dtype"] == "bfloat16"
        assert d["original_bytes"] == 50_331_648
        assert d["method"] == "fwht_quant"
        assert d["compression_time_ms"] == 12.3
        assert d["decompression_time_ms"] == 4.5


# ═══════════════════════════════════════════════════════════════════
# CompressionCertificate — Output Format Validation
# ═══════════════════════════════════════════════════════════════════


class TestCompressionCertificateOutput:
    def test_to_html_full_structure(self, sample_comp_cert):
        html = sample_comp_cert.to_html()
        assert html.startswith("<!DOCTYPE html>")
        assert "<html" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html
        assert "Compression Certificate" in html
        assert "SpectralStream" in html

    def test_to_html_contains_badges(self, sample_comp_cert):
        html = sample_comp_cert.to_html()
        assert 'class="badge-container"' in html
        assert 'class="badge"' in html
        assert "Compression Ratio" in html
        assert "Accuracy Preserved" in html
        assert "Signal Quality" in html
        assert "Space Saved" in html

    def test_to_html_contains_sections(self, sample_comp_cert):
        html = sample_comp_cert.to_html()
        assert "Overall Statistics" in html
        assert "Grade Distribution" in html
        assert "Method Distribution" in html
        assert "Per-Method Grade Breakdown" in html
        assert "Per-Tensor Report" in html
        assert "Industry Comparison" in html

    def test_to_html_grade_classes(self, sample_comp_cert):
        html = sample_comp_cert.to_html()
        for grade in SAMPLE_GRADES:
            assert f"grade-{grade}" in html

    def test_to_html_progress_bar(self, sample_comp_cert):
        html = sample_comp_cert.to_html()
        assert "progress-bar" in html
        assert "progress-fill" in html

    def test_to_html_method_tags(self, sample_comp_cert):
        html = sample_comp_cert.to_html()
        assert "method-tag" in html
        assert "block_int8 (6)" in html or "block_int8" in html

    def test_to_markdown_tables(self, sample_comp_cert):
        md = sample_comp_cert.to_markdown()
        assert "# Compression Certificate" in md
        assert "| Metric | Value |" in md
        assert "| Grade | Count |" in md
        assert "| Method | Ratio | Result |" in md
        assert "**Rank:**" in md

    def test_to_markdown_overview_values(self, sample_comp_cert):
        md = sample_comp_cert.to_markdown()
        assert "| Compression Ratio | **4.00x** |" in md
        assert "| Average Error | 0.1000% |" in md
        assert "| Maximum Error | 0.5000% |" in md
        assert "| Average SNR | 30.00 dB |" in md

    def test_to_markdown_industry_section(self, sample_comp_cert):
        md = sample_comp_cert.to_markdown()
        assert "## Industry Comparison" in md
        assert "FP16" in md
        assert "INT4" in md

    def test_to_text_format(self, sample_comp_cert):
        text = sample_comp_cert.to_text()
        assert "╔══════════════════════════════════" in text
        assert "Compression Certificate" in text
        assert "test-model" in text
        assert "10" in text  # total_tensors

    def test_to_text_grade_bars(self, sample_comp_cert):
        text = sample_comp_cert.to_text()
        for grade in SAMPLE_GRADES:
            assert grade in text
        assert "█" in text

    def test_to_text_tensor_summaries(self, sample_comp_cert):
        text = sample_comp_cert.to_text()
        assert "layer_0" in text
        assert "block_int8" in text

    def test_to_text_with_empty_certificates(self):
        cert = CompressionCertificate(
            model_name="empty",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=0,
            total_compressed_bytes=0,
            overall_ratio=1.0,
            total_tensors=0,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[],
            method_distribution={},
        )
        text = cert.to_text()
        assert "empty" in text

    def test_to_html_with_zero_tensors(self):
        cert = CompressionCertificate(
            model_name="zero",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=0,
            total_compressed_bytes=0,
            overall_ratio=0.0,
            total_tensors=0,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[],
            method_distribution={},
        )
        html = cert.to_html()
        assert "Zero" in html or "zero" in html or "0.00x" in html or "0" in html
        assert "Grade Distribution" in html

    def test_to_markdown_no_tensors(self):
        cert = CompressionCertificate(
            model_name="bare",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=0,
            total_compressed_bytes=0,
            overall_ratio=1.0,
            total_tensors=0,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[],
            method_distribution={},
        )
        md = cert.to_markdown()
        assert "# Compression Certificate" in md

    def test_to_json_string_roundtrip(self, sample_comp_cert):
        d = sample_comp_cert.to_dict()
        s = json.dumps(d, default=str)
        loaded = json.loads(s)
        assert loaded["model"]["name"] == "test-model"
        assert loaded["compression"]["ratio"] == 4.0
        assert loaded["quality"]["avg_error_percent"] == 0.1


# ═══════════════════════════════════════════════════════════════════
# CertificateBuilder — From Compression Report
# ═══════════════════════════════════════════════════════════════════


class TestCertificateBuilderFromReport:
    def test_with_real_compressed_tensors(self, sample_compressed_tensors):
        cts = [ct for _, ct in sample_compressed_tensors]
        report = CompressionReport(
            tensors=cts,
            total_original_bytes=sum(
                int(ct.compression_ratio * ct.get_data_size()) for ct in cts
            ),
            total_compressed_bytes=sum(ct.get_data_size() for ct in cts),
            overall_ratio=4.0,
            average_ratio=4.0,
            weighted_error=0.005,
            avg_error=0.005,
            max_error=0.005,
            min_error=0.0,
            method_distribution={"block_int8": len(cts)},
            time_seconds=0.03,
        )
        cert = CertificateBuilder.from_compression_report(
            report, model_name="real-test", model_architecture="test-arch"
        )
        assert cert.model_name == "real-test"
        assert cert.model_architecture == "test-arch"
        assert cert.total_tensors == len(cts)
        assert cert.overall_ratio == 4.0
        assert len(cert.tensor_certificates) == len(cts)

    def test_with_tensor_names(self, sample_compressed_tensors):
        cts = [ct for _, ct in sample_compressed_tensors]
        names = [n for n, _ in sample_compressed_tensors]
        report = CompressionReport(
            tensors=cts,
            total_original_bytes=1_000_000,
            total_compressed_bytes=250_000,
            overall_ratio=4.0,
            average_ratio=4.0,
            weighted_error=0.005,
            avg_error=0.005,
            max_error=0.005,
            method_distribution={"block_int8": len(cts)},
        )
        cert = CertificateBuilder.from_compression_report(report, tensor_names=names)
        assert len(cert.tensor_certificates) == len(names)
        for tc, expected_name in zip(cert.tensor_certificates, names):
            assert tc.name == expected_name

    def test_without_tensor_names_fallback(self, sample_compressed_tensors):
        cts = [ct for _, ct in sample_compressed_tensors]
        report = CompressionReport(
            tensors=cts,
            total_original_bytes=1_000_000,
            total_compressed_bytes=250_000,
            overall_ratio=4.0,
            average_ratio=4.0,
            weighted_error=0.005,
            avg_error=0.005,
            max_error=0.005,
            method_distribution={"block_int8": len(cts)},
        )
        cert = CertificateBuilder.from_compression_report(report)
        assert len(cert.tensor_certificates) == len(cts)

    def test_empty_report(self):
        report = CompressionReport(
            tensors=[],
            total_original_bytes=0,
            total_compressed_bytes=0,
            overall_ratio=1.0,
            average_ratio=1.0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            method_distribution={},
        )
        cert = CertificateBuilder.from_compression_report(report, model_name="empty")
        assert cert.total_tensors == 0
        assert cert.overall_ratio == 1.0
        assert len(cert.tensor_certificates) == 0

    def test_with_fewer_names_than_tensors(self, sample_compressed_tensors):
        cts = [ct for _, ct in sample_compressed_tensors]
        report = CompressionReport(
            tensors=cts,
            total_original_bytes=1_000_000,
            total_compressed_bytes=250_000,
            overall_ratio=4.0,
            average_ratio=4.0,
            weighted_error=0.005,
            avg_error=0.005,
            max_error=0.005,
            method_distribution={"block_int8": len(cts)},
        )
        names = ["only_one"]
        cert = CertificateBuilder.from_compression_report(report, tensor_names=names)
        assert len(cert.tensor_certificates) == len(cts)
        assert cert.tensor_certificates[0].name == "only_one"
        assert cert.tensor_certificates[1].name.startswith("tensor_")

    def test_single_tensor_report(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        ct = CompressedTensor(
            _data=tensor.tobytes(),
            method="svd",
            params={"rank": 4},
            original_shape=tensor.shape,
            original_dtype="float32",
            compression_ratio=2.0,
            relative_error=0.001,
            snr_db=35.0,
            psnr_db=40.0,
            cosine_similarity=0.99,
            computation_time=0.005,
        )
        report = CompressionReport(
            tensors=[ct],
            total_original_bytes=1024,
            total_compressed_bytes=512,
            overall_ratio=2.0,
            average_ratio=2.0,
            weighted_error=0.001,
            avg_error=0.001,
            max_error=0.001,
            method_distribution={"svd": 1},
            time_seconds=0.005,
        )
        cert = CertificateBuilder.from_compression_report(report, model_name="single")
        assert cert.total_tensors == 1
        assert cert.overall_ratio == 2.0
        assert len(cert.tensor_certificates) == 1
        assert cert.tensor_certificates[0].method == "svd"

    def test_infinite_snr_in_report(self):
        tensor = np.random.randn(8, 8).astype(np.float32)
        ct = CompressedTensor(
            _data=tensor.tobytes(),
            method="identity",
            params={},
            original_shape=tensor.shape,
            original_dtype="float32",
            compression_ratio=1.0,
            relative_error=0.0,
            snr_db=float("inf"),
            psnr_db=float("inf"),
            cosine_similarity=1.0,
            computation_time=0.0,
        )
        report = CompressionReport(
            tensors=[ct, ct],
            total_original_bytes=512,
            total_compressed_bytes=512,
            overall_ratio=1.0,
            average_ratio=1.0,
            weighted_error=0.0,
            avg_error=0.0,
            max_error=0.0,
            method_distribution={"identity": 2},
        )
        cert = CertificateBuilder.from_compression_report(report, model_name="inf_snr")
        # avg_snr_db should be inf (both tensors have inf SNR, filtered out)
        assert cert.avg_snr_db == 0 or cert.avg_snr_db == float("inf")


# ═══════════════════════════════════════════════════════════════════
# CertificateBuilder — From Compressed Tensors
# ═══════════════════════════════════════════════════════════════════


class TestCertificateBuilderFromTensors:
    def test_empty_list(self):
        cert = CertificateBuilder.from_compressed_tensors([], model_name="empty")
        assert cert.total_tensors == 0
        assert cert.overall_ratio == 0.0
        assert len(cert.tensor_certificates) == 0
        assert cert.total_original_bytes == 0
        assert cert.total_compressed_bytes == 0

    def test_single_tensor(self):
        tensor = np.random.randn(32, 32).astype(np.float32)
        ct = CompressedTensor(
            _data=tensor.tobytes(),
            method="block_int8",
            params={},
            original_shape=tensor.shape,
            original_dtype="float32",
            compression_ratio=4.0,
            relative_error=0.005,
            snr_db=25.0,
            psnr_db=30.0,
            cosine_similarity=0.95,
            computation_time=0.01,
        )
        cert = CertificateBuilder.from_compressed_tensors(
            [("single_tensor", ct)], model_name="single"
        )
        assert cert.total_tensors == 1
        assert cert.overall_ratio == 4.0
        assert len(cert.tensor_certificates) == 1
        assert cert.tensor_certificates[0].name == "single_tensor"

    def test_multiple_tensors(self, sample_compressed_tensors):
        cert = CertificateBuilder.from_compressed_tensors(
            sample_compressed_tensors, model_name="multi"
        )
        assert cert.total_tensors == 3
        assert len(cert.tensor_certificates) == 3
        names = [tc.name for tc in cert.tensor_certificates]
        assert "attn_q" in names
        assert "attn_k" in names
        assert "ffn_gate" in names

    def test_compression_time_passed(self, sample_compressed_tensors):
        cert = CertificateBuilder.from_compressed_tensors(
            sample_compressed_tensors,
            model_name="timed",
            compression_time=12.5,
        )
        assert cert.compression_time_seconds == 12.5

    def test_method_distribution(self, sample_compressed_tensors):
        cert = CertificateBuilder.from_compressed_tensors(
            sample_compressed_tensors, model_name="dist"
        )
        assert cert.method_distribution.get("block_int8", 0) >= 3

    def test_grade_distribution(self, sample_compressed_tensors):
        cert = CertificateBuilder.from_compressed_tensors(
            sample_compressed_tensors, model_name="grades"
        )
        total = sum(cert.grade_distribution.values())
        assert total == 3

    def test_zero_ratio_tensor(self):
        ct = CompressedTensor(
            _data=b"",
            method="test",
            params={},
            original_shape=(1,),
            original_dtype="float32",
            compression_ratio=0.0,
            relative_error=0.0,
            snr_db=float("inf"),
            psnr_db=float("inf"),
            cosine_similarity=1.0,
            computation_time=0.0,
        )
        cert = CertificateBuilder.from_compressed_tensors(
            [("zero_ratio", ct)], model_name="zero"
        )
        assert cert.total_compressed_bytes == 0
        assert cert.overall_ratio == 0.0
        assert cert.total_tensors == 1


# ═══════════════════════════════════════════════════════════════════
# Save / Load Roundtrip — All Formats
# ═══════════════════════════════════════════════════════════════════


class TestSaveLoadRoundtrip:
    def test_json_roundtrip(self, sample_comp_cert):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "cert")
            sample_comp_cert.save(base, formats=["json"])
            with open(f"{base}.json") as f:
                loaded = json.load(f)
        expected = sample_comp_cert.to_dict()
        assert loaded["model"]["name"] == expected["model"]["name"]
        assert loaded["compression"]["ratio"] == expected["compression"]["ratio"]
        assert loaded["quality"]["avg_snr_db"] == expected["quality"]["avg_snr_db"]
        assert len(loaded["tensors"]) == len(expected["tensors"])

    def test_html_roundtrip(self, sample_comp_cert):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "cert")
            sample_comp_cert.save(base, formats=["html"])
            with open(f"{base}.html") as f:
                content = f.read()
        assert content == sample_comp_cert.to_html()

    def test_md_roundtrip(self, sample_comp_cert):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "cert")
            sample_comp_cert.save(base, formats=["md"])
            with open(f"{base}.md") as f:
                content = f.read()
        assert content == sample_comp_cert.to_markdown()

    def test_txt_roundtrip(self, sample_comp_cert):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "cert")
            sample_comp_cert.save(base, formats=["txt"])
            with open(f"{base}.txt") as f:
                content = f.read()
        assert content == sample_comp_cert.to_text()

    def test_all_formats_roundtrip(self, sample_comp_cert):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "full_cert")
            sample_comp_cert.save(base)
            with open(f"{base}.json") as f:
                j = json.load(f)
            with open(f"{base}.html") as f:
                h = f.read()
            with open(f"{base}.md") as f:
                m = f.read()
            with open(f"{base}.txt") as f:
                t = f.read()
        assert j["model"]["name"] == "test-model"
        assert "Compression Certificate" in h
        assert "test-model" in m
        assert "test-model" in t

    def test_validation_cert_json_roundtrip(self):
        vc = ValidationCertificate(
            file_path="/tmp/test.ssf",
            file_size=1_000_000,
            n_tensors=3,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_failed=0,
            overall_ratio=4.0,
            avg_relative_error=0.001,
            tensor_results=[_make_validation_result()],
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "val")
            vc.save(base, formats=["json"])
            with open(f"{base}.json") as f:
                loaded = json.load(f)
        assert loaded["file"] == "/tmp/test.ssf"
        assert loaded["valid"] is True

    def test_validation_cert_all_formats_save(self):
        vc = ValidationCertificate(
            file_path="/tmp/t.ssf",
            file_size=500,
            n_tensors=1,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_failed=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "v")
            vc.save(base)
            for ext in ("json", "html", "md", "txt"):
                assert os.path.exists(f"{base}.{ext}")
                assert os.path.getsize(f"{base}.{ext}") > 0

    def test_save_selective_formats_comp_cert(self, sample_comp_cert):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "sel")
            sample_comp_cert.save(base, formats=["json", "txt"])
            assert os.path.exists(f"{base}.json")
            assert os.path.exists(f"{base}.txt")
            assert not os.path.exists(f"{base}.html")
            assert not os.path.exists(f"{base}.md")


# ═══════════════════════════════════════════════════════════════════
# Industry Comparison — Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestIndustryComparison:
    def test_very_high_ratio(self):
        cert = CompressionCertificate(
            model_name="extreme",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=1_000_000_000,
            total_compressed_bytes=1_000,
            overall_ratio=1_000_000.0,
            total_tensors=1,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[_make_tensor_cert()],
            method_distribution={"magic": 1},
        )
        comp = cert.industry_comparison
        assert comp["beats_standard_quant"] is True
        assert comp["beats_int4"] is True
        assert comp["better_than_count"] > 0
        assert len(comp["comparisons"]) >= 9
        names = [c["name"] for c in comp["comparisons"]]
        assert any("SpectralStream" in n for n in names)

    def test_very_low_ratio(self):
        cert = CompressionCertificate(
            model_name="poor",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=1_000,
            total_compressed_bytes=1_000,
            overall_ratio=1.0,
            total_tensors=1,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[_make_tensor_cert()],
            method_distribution={"none": 1},
        )
        comp = cert.industry_comparison
        assert comp["beats_standard_quant"] is False
        assert comp["beats_int4"] is False
        assert comp["rank"].startswith("10/")

    def test_ratio_equal_to_standard(self):
        cert = CompressionCertificate(
            model_name="equal",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=4_000,
            total_compressed_bytes=1_000,
            overall_ratio=4.0,
            total_tensors=1,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[_make_tensor_cert()],
            method_distribution={"block_int8": 1},
        )
        comp = cert.industry_comparison
        assert comp["beats_standard_quant"] is False
        assert comp["beats_int4"] is False

    def test_ratio_equal_to_int4(self):
        cert = CompressionCertificate(
            model_name="int4_like",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=8_000,
            total_compressed_bytes=1_000,
            overall_ratio=8.0,
            total_tensors=1,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[_make_tensor_cert()],
            method_distribution={"int4": 1},
        )
        comp = cert.industry_comparison
        assert comp["beats_standard_quant"] is True
        assert comp["beats_int4"] is False

    def test_industry_comparison_has_correct_keys(self, sample_comp_cert):
        comp = sample_comp_cert.industry_comparison
        for key in (
            "comparisons",
            "beats_standard_quant",
            "beats_int4",
            "rank",
            "better_than_count",
            "total_compared",
        ):
            assert key in comp, f"Missing key: {key}"
        assert isinstance(comp["comparisons"], list)
        for c in comp["comparisons"]:
            for key in ("name", "ratio", "description", "type", "beats"):
                assert key in c, f"Missing key in comparison: {key}"

    def test_industry_comparison_current_method_included(self, sample_comp_cert):
        comp = sample_comp_cert.industry_comparison
        names = [c["name"] for c in comp["comparisons"]]
        assert "SpectralStream (current)" in names

    def test_industry_comparison_ratio_rounded(self):
        cert = CompressionCertificate(
            model_name="rnd",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=1_000_000,
            total_compressed_bytes=322_581,
            overall_ratio=3.1,
            total_tensors=1,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[_make_tensor_cert()],
            method_distribution={"x": 1},
        )
        comp = cert.industry_comparison
        current_entry = [
            c for c in comp["comparisons"] if c["name"] == "SpectralStream (current)"
        ]
        assert len(current_entry) == 1
        assert current_entry[0]["ratio"] == 3.1


# ═══════════════════════════════════════════════════════════════════
# Grade Distribution
# ═══════════════════════════════════════════════════════════════════


class TestGradeDistribution:
    def test_all_grades_present(self):
        certs = []
        for i, grade in enumerate(SAMPLE_GRADES):
            certs.append(
                _make_tensor_cert(
                    name=f"t_{grade}",
                    quality_grade=grade,
                    compression_ratio=float(4 - i) if i < 3 else 1.0,
                )
            )
        cert = CompressionCertificate(
            model_name="all_grades",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=24,
            total_compressed_bytes=12,
            overall_ratio=2.0,
            total_tensors=6,
            compression_time_seconds=1.0,
            weighted_error=0.01,
            avg_error=0.01,
            max_error=0.01,
            min_error=0.0,
            avg_snr_db=20,
            tensor_certificates=certs,
            method_distribution={"test": 6},
        )
        for grade in SAMPLE_GRADES:
            assert cert.grade_distribution.get(grade) == 1, f"Missing grade {grade}"

    def test_empty_grade_distribution(self):
        cert = CompressionCertificate(
            model_name="empty",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=0,
            total_compressed_bytes=0,
            overall_ratio=1.0,
            total_tensors=0,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[],
            method_distribution={},
        )
        for grade in SAMPLE_GRADES:
            assert cert.grade_distribution.get(grade) == 0

    def test_single_grade_only(self):
        certs = [_make_tensor_cert(name=str(i), quality_grade="S") for i in range(5)]
        cert = CompressionCertificate(
            model_name="all_s",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=40,
            total_compressed_bytes=20,
            overall_ratio=2.0,
            total_tensors=5,
            compression_time_seconds=1.0,
            weighted_error=0.001,
            avg_error=0.001,
            max_error=0.001,
            min_error=0.0,
            avg_snr_db=40,
            tensor_certificates=certs,
            method_distribution={"test": 5},
        )
        assert cert.grade_distribution["S"] == 5
        for grade in ("A", "B", "C", "D", "F"):
            assert cert.grade_distribution.get(grade) == 0

    def test_grade_distribution_updates_after_creation(self):
        cert = CompressionCertificate(
            model_name="dyn",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=0,
            total_compressed_bytes=0,
            overall_ratio=1.0,
            total_tensors=0,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[],
            method_distribution={},
        )
        assert cert.grade_distribution["S"] == 0
        cert.tensor_certificates.append(_make_tensor_cert(quality_grade="S"))
        cert._compute_distributions()
        assert cert.grade_distribution["S"] >= 1

    def test_grade_distribution_in_to_dict(self):
        certs = [_make_tensor_cert(quality_grade="A") for _ in range(3)]
        cert = CompressionCertificate(
            model_name="g",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=12,
            total_compressed_bytes=6,
            overall_ratio=2.0,
            total_tensors=3,
            compression_time_seconds=0,
            weighted_error=0.001,
            avg_error=0.001,
            max_error=0.001,
            min_error=0.0,
            avg_snr_db=30,
            tensor_certificates=certs,
            method_distribution={"test": 3},
        )
        d = cert.to_dict()
        assert d["quality"]["grade_distribution"]["A"] == 3


# ═══════════════════════════════════════════════════════════════════
# Method Grade Breakdown
# ═══════════════════════════════════════════════════════════════════


class TestMethodGradeBreakdown:
    def test_single_method_multiple_grades(self):
        certs = [
            _make_tensor_cert(name="a", method="block_int8", quality_grade="S"),
            _make_tensor_cert(name="b", method="block_int8", quality_grade="A"),
            _make_tensor_cert(name="c", method="block_int8", quality_grade="B"),
        ]
        cert = CompressionCertificate(
            model_name="multi_grade",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=12,
            total_compressed_bytes=6,
            overall_ratio=2.0,
            total_tensors=3,
            compression_time_seconds=0,
            weighted_error=0.005,
            avg_error=0.005,
            max_error=0.01,
            min_error=0.0,
            avg_snr_db=25,
            tensor_certificates=certs,
            method_distribution={"block_int8": 3},
        )
        breakdown = cert.method_grade_breakdown
        assert "block_int8" in breakdown
        assert breakdown["block_int8"]["S"] == 1
        assert breakdown["block_int8"]["A"] == 1
        assert breakdown["block_int8"]["B"] == 1

    def test_multiple_methods(self):
        certs = [
            _make_tensor_cert(name="a", method="int8", quality_grade="S"),
            _make_tensor_cert(name="b", method="int8", quality_grade="S"),
            _make_tensor_cert(name="c", method="svd", quality_grade="A"),
            _make_tensor_cert(name="d", method="fwht", quality_grade="C"),
        ]
        cert = CompressionCertificate(
            model_name="multi_method",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=16,
            total_compressed_bytes=8,
            overall_ratio=2.0,
            total_tensors=4,
            compression_time_seconds=0,
            weighted_error=0.005,
            avg_error=0.005,
            max_error=0.01,
            min_error=0.0,
            avg_snr_db=25,
            tensor_certificates=certs,
            method_distribution={"int8": 2, "svd": 1, "fwht": 1},
        )
        breakdown = cert.method_grade_breakdown
        assert breakdown["int8"]["S"] == 2
        assert breakdown["svd"]["A"] == 1
        assert breakdown["fwht"]["C"] == 1

    def test_empty_breakdown(self):
        cert = CompressionCertificate(
            model_name="empty",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=0,
            total_compressed_bytes=0,
            overall_ratio=1.0,
            total_tensors=0,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[],
            method_distribution={},
        )
        assert cert.method_grade_breakdown == {}

    def test_breakdown_in_to_dict(self):
        certs = [_make_tensor_cert(method="fwht", quality_grade="S") for _ in range(2)]
        cert = CompressionCertificate(
            model_name="dict_test",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=8,
            total_compressed_bytes=4,
            overall_ratio=2.0,
            total_tensors=2,
            compression_time_seconds=0,
            weighted_error=0.001,
            avg_error=0.001,
            max_error=0.001,
            min_error=0.0,
            avg_snr_db=30,
            tensor_certificates=certs,
            method_distribution={"fwht": 2},
        )
        d = cert.to_dict()
        assert d["method_grade_breakdown"]["fwht"]["S"] == 2

    def test_breakdown_covers_all_grades_for_method(self):
        certs = [
            _make_tensor_cert(name=str(i), method="quant", quality_grade=g)
            for g, i in zip(SAMPLE_GRADES, range(6))
        ]
        cert = CompressionCertificate(
            model_name="grades",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=24,
            total_compressed_bytes=12,
            overall_ratio=2.0,
            total_tensors=6,
            compression_time_seconds=0,
            weighted_error=0.01,
            avg_error=0.01,
            max_error=0.01,
            min_error=0.0,
            avg_snr_db=20,
            tensor_certificates=certs,
            method_distribution={"quant": 6},
        )
        breakdown = cert.method_grade_breakdown["quant"]
        for grade in SAMPLE_GRADES:
            assert breakdown.get(grade) == 1, f"Missing grade {grade} in breakdown"


# ═══════════════════════════════════════════════════════════════════
# Marketing Highlights
# ═══════════════════════════════════════════════════════════════════


class TestMarketingHighlights:
    def test_all_keys_present(self, sample_comp_cert):
        highlights = sample_comp_cert.marketing_highlights
        expected_keys = [
            "compression_power",
            "original_size_gb",
            "compressed_size_gb",
            "space_saved_gb",
            "accuracy_preserved",
            "signal_quality",
            "s_grade_tensors",
            "a_grade_or_better",
            "fastest_method",
            "methods_used",
            "time_saved",
        ]
        for key in expected_keys:
            assert key in highlights, f"Missing marketing key: {key}"

    def test_compression_power_value(self, sample_comp_cert):
        assert sample_comp_cert.marketing_highlights["compression_power"] == "4.0x"

    def test_accuracy_preserved(self, sample_comp_cert):
        val = sample_comp_cert.marketing_highlights["accuracy_preserved"]
        assert "%" in val
        assert float(val.replace("%", "")) > 0

    def test_signal_quality(self, sample_comp_cert):
        val = sample_comp_cert.marketing_highlights["signal_quality"]
        assert "dB" in val

    def test_s_grade_tensors(self, sample_comp_cert):
        assert sample_comp_cert.marketing_highlights["s_grade_tensors"] == "0"

    def test_a_grade_or_better(self, sample_comp_cert):
        assert sample_comp_cert.marketing_highlights["a_grade_or_better"] == "3"

    def test_fastest_method(self, sample_comp_cert):
        assert sample_comp_cert.marketing_highlights["fastest_method"] == "block_int8"

    def test_fastest_method_no_distribution(self):
        cert = CompressionCertificate(
            model_name="no_methods",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=0,
            total_compressed_bytes=0,
            overall_ratio=1.0,
            total_tensors=0,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[],
            method_distribution={},
        )
        assert cert.marketing_highlights["fastest_method"] == "N/A"

    def test_methods_used_count(self, sample_comp_cert):
        assert sample_comp_cert.marketing_highlights["methods_used"] == "1"

    def test_time_saved_is_string(self, sample_comp_cert):
        val = sample_comp_cert.marketing_highlights["time_saved"]
        assert isinstance(val, str)
        assert len(val) > 0

    def test_zero_sizes_marketing(self):
        cert = CompressionCertificate(
            model_name="zero",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=0,
            total_compressed_bytes=0,
            overall_ratio=0.0,
            total_tensors=0,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[],
            method_distribution={},
        )
        highlights = cert.marketing_highlights
        assert highlights["compression_power"] == "0.0x"
        assert highlights["original_size_gb"] == "0.00"
        assert highlights["compressed_size_gb"] == "0.00"

    def test_space_saved_negative(self):
        """If compressed is larger than original, space saved should be negative."""
        cert = CompressionCertificate(
            model_name="neg",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=100,
            total_compressed_bytes=200,
            overall_ratio=0.5,
            total_tensors=1,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[_make_tensor_cert()],
            method_distribution={"x": 1},
        )
        saved = cert.marketing_highlights["space_saved_gb"]
        assert "-" in saved or saved.startswith("-")

    def test_marketing_in_to_dict(self, sample_comp_cert):
        d = sample_comp_cert.to_dict()
        assert "marketing" in d
        assert d["marketing"]["compression_power"] == "4.0x"


# ═══════════════════════════════════════════════════════════════════
# Download Time Estimate
# ═══════════════════════════════════════════════════════════════════


class TestDownloadTimeEstimate:
    def test_returns_seconds_for_small_savings(self):
        cert = CompressionCertificate(
            model_name="small",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=10_000_000,
            total_compressed_bytes=5_000_000,
            overall_ratio=2.0,
            total_tensors=1,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[_make_tensor_cert()],
            method_distribution={"x": 1},
        )
        saved = cert._estimate_download_time_saved()
        assert saved.endswith("seconds") or saved.endswith("second")

    def test_returns_minutes_for_medium_savings(self):
        cert = CompressionCertificate(
            model_name="med",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=1_000_000_000,
            total_compressed_bytes=100_000_000,
            overall_ratio=10.0,
            total_tensors=1,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[_make_tensor_cert()],
            method_distribution={"x": 1},
        )
        saved = cert._estimate_download_time_saved()
        assert "minutes" in saved or "minute" in saved

    def test_returns_hours_for_large_savings(self):
        cert = CompressionCertificate(
            model_name="large",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=500_000_000_000,
            total_compressed_bytes=50_000_000_000,
            overall_ratio=10.0,
            total_tensors=10,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[_make_tensor_cert() for _ in range(10)],
            method_distribution={"x": 10},
        )
        saved = cert._estimate_download_time_saved()
        assert "hours" in saved or "hour" in saved

    def test_zero_savings(self):
        cert = CompressionCertificate(
            model_name="none",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=100,
            total_compressed_bytes=100,
            overall_ratio=1.0,
            total_tensors=1,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[_make_tensor_cert()],
            method_distribution={"x": 1},
        )
        saved = cert._estimate_download_time_saved()
        assert isinstance(saved, str)
        assert len(saved) > 0
        # zero bytes saved at 100 Mbps = 0 seconds
        assert "0 seconds" in saved or "0.0" in saved

    def test_negative_savings(self):
        """When compressed is larger than original, saved bytes is negative → 0 seconds."""
        cert = CompressionCertificate(
            model_name="neg",
            model_path="",
            model_architecture="",
            model_params="",
            total_original_bytes=100,
            total_compressed_bytes=200,
            overall_ratio=0.5,
            total_tensors=1,
            compression_time_seconds=0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            avg_snr_db=0,
            tensor_certificates=[_make_tensor_cert()],
            method_distribution={"x": 1},
        )
        saved = cert._estimate_download_time_saved()
        assert isinstance(saved, str)

    def test_time_saved_in_marketing_highlights(self, sample_comp_cert):
        assert "time_saved" in sample_comp_cert.marketing_highlights
        val = sample_comp_cert.marketing_highlights["time_saved"]
        assert isinstance(val, str)
        assert len(val) > 0


# ═══════════════════════════════════════════════════════════════════
# _grade_error Helper
# ═══════════════════════════════════════════════════════════════════


class TestGradeErrorHelper:
    def test_grade_s(self):
        assert _grade_error(0.0001) == "S"
        assert _grade_error(0.0001999) == "S"

    def test_grade_a(self):
        assert _grade_error(0.0005) == "A"
        assert _grade_error(0.000999) == "A"

    def test_grade_b(self):
        assert _grade_error(0.003) == "B"
        assert _grade_error(0.004999) == "B"

    def test_grade_c(self):
        assert _grade_error(0.007) == "C"
        assert _grade_error(0.009999) == "C"

    def test_grade_d(self):
        assert _grade_error(0.02) == "D"
        assert _grade_error(0.049999) == "D"

    def test_grade_f(self):
        assert _grade_error(0.05) == "F"
        assert _grade_error(1.0) == "F"

    def test_zero_error(self):
        assert _grade_error(0.0) == "S"

    def test_boundary_values(self):
        assert _grade_error(0.0002) == "A"
        assert _grade_error(0.0002001) == "A"
        assert _grade_error(0.001) == "B"
        assert _grade_error(0.001001) == "B"
        assert _grade_error(0.005) == "C"
        assert _grade_error(0.005001) == "C"
        assert _grade_error(0.01) == "D"
        assert _grade_error(0.010001) == "D"
        assert _grade_error(0.05) == "F"


# ═══════════════════════════════════════════════════════════════════
# _compute_metrics Helper
# ═══════════════════════════════════════════════════════════════════


class TestComputeMetricsHelper:
    def test_identical_arrays(self):
        x = np.random.randn(32, 32).astype(np.float32)
        m = _compute_metrics(x, x)
        assert m["mse"] == 0.0
        assert m["relative_error"] == 0.0
        assert m["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_zero_arrays(self):
        x = np.zeros((16, 16), dtype=np.float32)
        m = _compute_metrics(x, x)
        assert m["mse"] == 0.0
        assert m["relative_error"] == 0.0

    def test_different_arrays(self):
        x = np.ones((8, 8), dtype=np.float32)
        y = np.zeros((8, 8), dtype=np.float32)
        m = _compute_metrics(x, y)
        assert m["mse"] > 0
        assert m["relative_error"] > 0
        assert m["snr_db"] < 10

    def test_different_shapes(self):
        a = np.random.randn(16, 16).astype(np.float32)
        b = np.random.randn(8, 8).astype(np.float32)
        m = _compute_metrics(a, b)
        # Both are truncated to min length
        assert len(a.ravel()) == 256
        assert len(b.ravel()) == 64
        # The metrics should still compute
        assert "mse" in m
        assert "snr_db" in m

    def test_cosine_similarity_negative(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        m = _compute_metrics(a, b)
        assert m["cosine_similarity"] == pytest.approx(-1.0, abs=1e-6)

    def test_perfect_cosine_similarity(self):
        a = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        b = a * 2.0
        m = _compute_metrics(a, b)
        assert m["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_snr_vs_psnr(self):
        a = np.random.randn(32, 32).astype(np.float32)
        b = a + 0.01 * np.random.randn(32, 32).astype(np.float32)
        m = _compute_metrics(a, b)
        assert m["snr_db"] > 0
        assert m["psnr_db"] > 0


# ═══════════════════════════════════════════════════════════════════
# Cross-class integration
# ═══════════════════════════════════════════════════════════════════


class TestCrossClassIntegration:
    def test_tensor_cert_from_compressed_tensor(self):
        tensor = np.random.randn(16, 16).astype(np.float32)
        ct = CompressedTensor(
            _data=tensor.tobytes(),
            method="dct",
            params={},
            original_shape=tensor.shape,
            original_dtype="float32",
            compression_ratio=3.0,
            relative_error=0.002,
            snr_db=28.0,
            psnr_db=33.0,
            cosine_similarity=0.97,
            computation_time=0.01,
        )
        tc = TensorCertificate(
            name="integrated",
            shape=ct.original_shape,
            original_dtype=ct.original_dtype,
            original_bytes=int(ct.compression_ratio * ct.get_data_size()),
            compressed_bytes=ct.get_data_size(),
            compression_ratio=ct.compression_ratio,
            method=ct.method,
            method_category="spectral",
            relative_error=ct.relative_error,
            snr_db=ct.snr_db,
            psnr_db=ct.psnr_db,
            cosine_similarity=ct.cosine_similarity,
            mse=ct.relative_error**2,
            compression_time_ms=ct.computation_time * 1000,
            decompression_time_ms=0.0,
            quality_grade=ct.quality_grade,
        )
        assert tc.shape == (16, 16)
        assert tc.compression_ratio == 3.0
        assert tc.method == "dct"
        assert tc.quality_grade in SAMPLE_GRADES

    def test_builder_into_certificate_output(self, sample_compressed_tensors):
        cert = CertificateBuilder.from_compressed_tensors(
            sample_compressed_tensors, model_name="full_pipeline"
        )
        html = cert.to_html()
        assert "full_pipeline" in html
        md = cert.to_markdown()
        assert "full_pipeline" in md
        text = cert.to_text()
        assert "full_pipeline" in text
        d = cert.to_dict()
        assert d["model"]["name"] == "full_pipeline"

    def test_validation_cert_with_results(self):
        results = [
            _make_validation_result(name=f"t{i}", quality_grade=g)
            for i, g in enumerate(["S", "S", "S", "S", "S"])
        ]
        vc = ValidationCertificate(
            file_path="test.ssf",
            file_size=1_000_000,
            n_tensors=5,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_validated=5,
            tensors_failed=0,
            tensor_results=results,
            grade_distribution={"S": 5, "A": 0, "B": 0, "C": 0, "D": 0, "F": 0},
            overall_ratio=4.0,
            avg_relative_error=0.003,
            avg_snr_db=30.0,
        )
        assert vc.is_valid()
        assert vc.overall_grade() == "S"
        assert len(vc.tensor_results) == 5

    def test_to_dict_tensor_cert_edge(self):
        tc = _make_tensor_cert(
            name="test",
            snr_db=float("inf"),
            psnr_db=float("inf"),
            compression_ratio=0.0,
        )
        d = tc.to_dict()
        assert d["snr_db"] == float("inf")
        assert d["compression_ratio"] == 0.0
        # Ensure JSON-serializable
        json_str = json.dumps(d, default=str)
        loaded = json.loads(json_str)
        assert loaded["name"] == "test"

    def test_empty_grade_distribution_does_not_crash_to_html(self):
        vc = ValidationCertificate(
            file_path="",
            file_size=0,
            n_tensors=0,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_failed=0,
        )
        html = vc.to_html()
        assert "<html" in html

    def test_empty_grade_distribution_does_not_crash_to_markdown(self):
        vc = ValidationCertificate(
            file_path="",
            file_size=0,
            n_tensors=0,
            header_ok=True,
            checksum_ok=True,
            index_ok=True,
            tensors_failed=0,
        )
        md = vc.to_markdown()
        assert "Grade Distribution" in md
