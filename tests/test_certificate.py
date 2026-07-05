"""Tests for the compression certificate system — TensorCertificate, CompressionCertificate, CertificateBuilder."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.compression.certificate import (
    TensorCertificate,
    CompressionCertificate,
    CertificateBuilder,
)
from spectralstream.compression.engine import (
    CompressedTensor,
    CompressionReport,
)


@pytest.fixture
def sample_tensor_certificate() -> TensorCertificate:
    return TensorCertificate(
        name="test_tensor",
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
        data = tensor.tobytes()
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


class TestTensorCertificate:
    def test_to_dict(self, sample_tensor_certificate):
        d = sample_tensor_certificate.to_dict()
        assert d["name"] == "test_tensor"
        assert d["compression_ratio"] == 4.0
        assert d["quality_grade"] == "A"

    def test_summary_line(self, sample_tensor_certificate):
        line = sample_tensor_certificate.summary_line()
        assert "test_tensor" in line
        assert "4.00x" in line
        assert "A" in line

    def test_grade_s(self):
        cert = TensorCertificate(
            name="perfect",
            shape=(1,),
            original_dtype="float32",
            original_bytes=4,
            compressed_bytes=2,
            compression_ratio=2.0,
            method="test",
            method_category="test",
            relative_error=0.0001,
            snr_db=50,
            psnr_db=55,
            cosine_similarity=1.0,
            mse=0.0,
            compression_time_ms=0,
            decompression_time_ms=0,
            quality_grade="S",
        )
        assert cert.quality_grade == "S"

    def test_grade_f(self):
        cert = TensorCertificate(
            name="bad",
            shape=(1,),
            original_dtype="float32",
            original_bytes=4,
            compressed_bytes=2,
            compression_ratio=2.0,
            method="test",
            method_category="test",
            relative_error=0.5,
            snr_db=5,
            psnr_db=10,
            cosine_similarity=0.3,
            mse=0.5,
            compression_time_ms=0,
            decompression_time_ms=0,
            quality_grade="F",
        )
        assert cert.quality_grade == "F"

    def test_zero_ratio(self):
        cert = TensorCertificate(
            name="zero",
            shape=(1,),
            original_dtype="float32",
            original_bytes=0,
            compressed_bytes=0,
            compression_ratio=0.0,
            method="test",
            method_category="test",
            relative_error=0,
            snr_db=0,
            psnr_db=0,
            cosine_similarity=1.0,
            mse=0,
            compression_time_ms=0,
            decompression_time_ms=0,
            quality_grade="S",
        )
        assert cert.compression_ratio == 0.0


class TestCompressionCertificate:
    @pytest.fixture
    def sample_certificate(self, sample_tensor_certificate):
        return CompressionCertificate(
            model_name="test-model",
            model_path="/tmp/test",
            model_architecture="test-arch",
            model_params="1B",
            total_original_bytes=1000000,
            total_compressed_bytes=250000,
            overall_ratio=4.0,
            total_tensors=10,
            compression_time_seconds=5.0,
            weighted_error=0.002,
            avg_error=0.001,
            max_error=0.005,
            min_error=0.0001,
            avg_snr_db=30.0,
            tensor_certificates=[sample_tensor_certificate] * 3,
            method_distribution={"block_int8": 3},
        )

    def test_to_dict(self, sample_certificate):
        d = sample_certificate.to_dict()
        assert d["model"]["name"] == "test-model"
        assert d["compression"]["ratio"] == 4.0
        assert len(d["tensors"]) == 3

    def test_grade_distribution(self, sample_certificate):
        assert sample_certificate.grade_distribution["A"] == 3

    def test_method_distribution(self, sample_certificate):
        assert sample_certificate.method_distribution["block_int8"] >= 3

    def test_marketing_highlights(self, sample_certificate):
        highlights = sample_certificate.marketing_highlights
        assert "compression_power" in highlights
        assert highlights["compression_power"] == "4.0x"
        assert "accuracy_preserved" in highlights

    def test_industry_comparison(self, sample_certificate):
        comp = sample_certificate.industry_comparison
        assert "comparisons" in comp
        assert comp["beats_standard_quant"] is False
        assert comp["beats_int4"] is False

    def test_download_time_estimate(self, sample_certificate):
        saved = sample_certificate._estimate_download_time_saved()
        assert isinstance(saved, str)
        assert len(saved) > 0

    def test_grade_distribution_all_grades(self):
        certs = []
        for grade in ["S", "A", "B", "C", "D", "F"]:
            certs.append(
                TensorCertificate(
                    name=f"tensor_{grade}",
                    shape=(1,),
                    original_dtype="float32",
                    original_bytes=4,
                    compressed_bytes=2,
                    compression_ratio=2.0,
                    method="test",
                    method_category="test",
                    relative_error=0.01,
                    snr_db=20,
                    psnr_db=25,
                    cosine_similarity=0.9,
                    mse=0.01,
                    compression_time_ms=0,
                    decompression_time_ms=0,
                    quality_grade=grade,
                )
            )
        cert = CompressionCertificate(
            model_name="test",
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
        for grade in ["S", "A", "B", "C", "D", "F"]:
            assert cert.grade_distribution[grade] == 1, f"Missing grade {grade}"

    def test_to_html(self, sample_certificate):
        html = sample_certificate.to_html()
        assert "<html" in html
        assert "4.0x" in html
        assert "A" in html

    def test_to_markdown(self, sample_certificate):
        md = sample_certificate.to_markdown()
        assert "test-model" in md
        assert "4.00x" in md

    def test_to_json_string(self, sample_certificate):
        d = sample_certificate.to_dict()
        json_str = json.dumps(d)
        data = json.loads(json_str)
        assert data["model"]["name"] == "test-model"

    def test_serialization_roundtrip(self, sample_certificate):
        d = sample_certificate.to_dict()
        json_str = json.dumps(d)
        data = json.loads(json_str)
        assert data["compression"]["ratio"] == 4.0

    def test_save_load_roundtrip(self, sample_certificate):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "cert")
            sample_certificate.save(base)
            assert os.path.exists(f"{base}.json")
            assert os.path.exists(f"{base}.md")
            with open(f"{base}.json") as f:
                data = json.load(f)
            assert data["model"]["name"] == "test-model"

    def test_method_grade_breakdown(self, sample_certificate):
        breakdown = sample_certificate.method_grade_breakdown
        assert "block_int8" in breakdown
        assert breakdown["block_int8"]["A"] == 3


class TestCertificateBuilder:
    def test_from_compression_report(self, sample_compressed_tensors):
        cts = [ct for _, ct in sample_compressed_tensors]
        report = CompressionReport(
            tensors=cts,
            total_original_bytes=sum(
                int(ct.compression_ratio * len(ct.data)) for ct in cts
            ),
            total_compressed_bytes=sum(len(ct.data) for ct in cts),
            overall_ratio=4.0,
            average_ratio=4.0,
            weighted_error=0.005,
            avg_error=0.005,
            max_error=0.005,
            method_distribution={"block_int8": len(cts)},
        )
        cert = CertificateBuilder.from_compression_report(
            report, model_name="test-model"
        )
        assert isinstance(cert, CompressionCertificate)
        assert cert.model_name == "test-model"
        assert cert.total_tensors == len(cts)

    def test_from_compressed_tensors(self, sample_compressed_tensors):
        cert = CertificateBuilder.from_compressed_tensors(
            sample_compressed_tensors, model_name="test-model"
        )
        assert isinstance(cert, CompressionCertificate)
        assert cert.total_tensors == len(sample_compressed_tensors)
        assert cert.overall_ratio > 1.0

    def test_from_report_with_tensor_names(self, sample_compressed_tensors):
        cts = [ct for _, ct in sample_compressed_tensors]
        report = CompressionReport(
            tensors=cts,
            total_original_bytes=1000000,
            total_compressed_bytes=250000,
            overall_ratio=4.0,
            average_ratio=4.0,
            weighted_error=0.005,
            avg_error=0.005,
            max_error=0.005,
            method_distribution={"block_int8": len(cts)},
        )
        names = [n for n, _ in sample_compressed_tensors]
        cert = CertificateBuilder.from_compression_report(report, tensor_names=names)
        assert len(cert.tensor_certificates) == len(names)

    def test_from_report_empty(self):
        report = CompressionReport(
            tensors=[],
            total_original_bytes=0,
            total_compressed_bytes=0,
            overall_ratio=1.0,
            average_ratio=1.0,
            weighted_error=0,
            avg_error=0,
            max_error=0,
            min_error=0,
            method_distribution={},
        )
        cert = CertificateBuilder.from_compression_report(report, model_name="empty")
        assert cert.total_tensors == 0
        assert cert.overall_ratio == 1.0
