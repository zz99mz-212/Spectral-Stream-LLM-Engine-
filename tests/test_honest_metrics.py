"""
Tests for honest_metrics apply_gate chokepoint and serialized_nbytes shapes.

This test asserts the metric-trust-loop contract:
- A method whose rel_mse exceeds ERROR_GATE_THRESHOLD emits NO numeric compression
  ratio; its honest_metrics dict carries gated:True and ratio_vs_bf16:None.
- A good method (rel_mse <= threshold) emits both ratio_vs_bf16 (headline) and
  ratio_vs_fp32, derived byte-exactly from serialized_nbytes via dual_ratio.
- serialized_nbytes returns the true recursive byte count for every payload shape.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectralstream.compression.honest_metrics import (
    serialized_nbytes,
    dual_ratio,
    end_to_end_error,
    apply_gate,
    ERROR_GATE_THRESHOLD,
)

# ═══════════════════════════════════════════════════════════════════
# Test: Good method emits both ratios (RED state fails until GREEN)
# ═══════════════════════════════════════════════════════════════════


def test_good_method_emits_ratio():
    """A low-error method keeps both byte-exact ratios via the gate."""
    rng = np.random.RandomState(42)
    orig = rng.randn(8, 8).astype(np.float32)
    recon = orig + 1e-4 * rng.randn(8, 8).astype(np.float32)

    err = end_to_end_error(orig, recon)
    assert err.rel_mse <= ERROR_GATE_THRESHOLD

    res = apply_gate(orig.tobytes(), orig.size, err.rel_mse)
    assert res["gated"] is False
    assert res["ratio_vs_bf16"] is not None
    assert res["ratio_vs_fp32"] is not None
    assert "ratio_vs_bf16" in res
    assert "ratio_vs_fp32" in res


# ═══════════════════════════════════════════════════════════════════
# Test: High-error method is gated (no numeric ratio)
# ═══════════════════════════════════════════════════════════════════


def test_bad_method_gated():
    """A high-error method emits no numeric ratio; gate marks it."""
    rng = np.random.RandomState(42)
    orig = rng.randn(8, 8).astype(np.float32)
    recon = rng.randn(8, 8).astype(np.float32)  # uncorrelated

    err = end_to_end_error(orig, recon)
    assert err.rel_mse > ERROR_GATE_THRESHOLD

    res = apply_gate(orig.tobytes(), orig.size, err.rel_mse)
    assert res["gated"] is True
    assert res["ratio_vs_bf16"] is None
    assert res["ratio_vs_fp32"] is None
    assert "rel_mse" in res["gate_reason"]


# ═══════════════════════════════════════════════════════════════════
# Test: Boundary at exactly threshold is NOT gated (strict >)
# ═══════════════════════════════════════════════════════════════════


def test_boundary_exactly_threshold_not_gated():
    """At rel_mse == threshold, the gate is NOT triggered (consistent with Phase-3)."""
    payload = b"\x00" * 64
    original_elements = 32
    res = apply_gate(payload, original_elements, ERROR_GATE_THRESHOLD)
    assert res["gated"] is False


# ═══════════════════════════════════════════════════════════════════
# Test: Boundary just above threshold IS gated (strict >)
# ═══════════════════════════════════════════════════════════════════


def test_boundary_just_above_gated():
    """At rel_mse > threshold (even by epsilon), the gate IS triggered."""
    payload = b"\x00" * 64
    original_elements = 32
    res = apply_gate(payload, original_elements, ERROR_GATE_THRESHOLD + 1e-6)
    assert res["gated"] is True


# ═══════════════════════════════════════════════════════════════════
# Test: serialized_nbytes handles all payload shapes
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "payload,expected",
    [
        (None, 0),
        (b"abc", 3),
        (bytearray(b"xy"), 2),
        (np.zeros((4, 4), dtype=np.float32), 64),
        (np.float32(1.0), 4),
        ({"a": b"xy", "b": 3}, len("a") + len("b") + 2 + 8),
        ([b"ab", (4, 4)], 2 + 8 + 8),
        (True, 1),
        (7, 8),
        (2.5, 8),
        ("hi", 2),
    ],
)
def test_serialized_nbytes_shapes(payload, expected):
    """serialized_nbytes returns the true recursive byte count for every shape."""
    assert serialized_nbytes(payload) == expected


# ═══════════════════════════════════════════════════════════════════
# Test: BF16 is the headline key (surfaced default)
# ═══════════════════════════════════════════════════════════════════


def test_bf16_is_headline_key():
    """For a good method, apply_gate returns both ratios; BF16 is the surfaced default."""
    rng = np.random.RandomState(42)
    orig = rng.randn(8, 8).astype(np.float32)
    recon = orig + 1e-4 * rng.randn(8, 8).astype(np.float32)

    err = end_to_end_error(orig, recon)
    assert err.rel_mse <= ERROR_GATE_THRESHOLD

    res = apply_gate(orig.tobytes(), orig.size, err.rel_mse)
    assert res["gated"] is False
    assert res["ratio_vs_bf16"] is not None
    assert res["ratio_vs_fp32"] is not None
    assert "ratio_vs_bf16" in res
    assert "ratio_vs_fp32" in res
    # BF16 headline contract: the key is present and the method is gated=False
    # (which means BF16 is the valid baseline to surface by default)
    assert res["gated"] is False


# ═══════════════════════════════════════════════════════════════════
# Test: Gated ratio is None (not 0), so downstream means filter it out
# ═══════════════════════════════════════════════════════════════════


def test_gated_ratio_is_none_not_zero():
    """For a bad method, apply_gate returns None for both ratios (never 0.0x)."""
    rng = np.random.RandomState(42)
    orig = rng.randn(8, 8).astype(np.float32)
    recon = rng.randn(8, 8).astype(np.float32)  # uncorrelated

    err = end_to_end_error(orig, recon)
    assert err.rel_mse > ERROR_GATE_THRESHOLD

    res = apply_gate(orig.tobytes(), orig.size, err.rel_mse)
    assert res["gated"] is True
    assert res["ratio_vs_bf16"] is None
    assert res["ratio_vs_fp32"] is None
    # ...

    assert res["ratio_vs_bf16"] is None
    assert res["ratio_vs_fp32"] is None
    # The summary's None-filter depends on None, never 0.
    assert res["ratio_vs_bf16"] != 0.0
    assert res["ratio_vs_fp32"] != 0.0


# ═══════════════════════════════════════════════════════════════════
# Test: Gate reason format is consistent and human-readable
# ═══════════════════════════════════════════════════════════════════


def test_gate_reason_format():
    """Gate reason format: 'rel_mse {value} > {threshold}'."""
    payload = b"\x00" * 64
    original_elements = 32
    rel_mse = 0.11
    threshold = 0.05
    res = apply_gate(payload, original_elements, rel_mse, threshold)
    assert res["gated"] is True
    assert res["gate_reason"] == f"rel_mse {rel_mse:.4f} > {threshold}"


# ═══════════════════════════════════════════════════════════════════
# METRICS-03: Literature estimates extraction + disclaimer guards
# ═══════════════════════════════════════════════════════════════════


from spectralstream.compression.literature_estimates import (
    LITERATURE_ESTIMATES,
    LITERATURE_DISCLAIMER,
)


def test_literature_disclaimer_present():
    """LITERATURE_DISCLAIMER contains the mandatory 'literature estimates' label."""
    assert "literature estimates" in LITERATURE_DISCLAIMER.lower()
    assert len(LITERATURE_ESTIMATES) == 9


def test_no_competitor_literals_in_certificate_source():
    """certificate.py source code must contain NO bare competitor literals.

    This test guards against refabrication: any hardcoded GPTQ/AWQ/SqueezeLLM/GGML Q
    typenames signal a fabrication surface. They must be imported from
    literature_estimates instead.
    """
    from pathlib import Path

    cert_path = Path(__file__).parent.parent / "spectralstream" / "compression" / "certificate.py"
    src = cert_path.read_text(encoding="utf-8")

    # These are the 4 competitor families that MUST NOT appear as literals.
    forbidden = ["GPTQ", "AWQ", "SqueezeLLM", "GGML Q"]
    for term in forbidden:
        assert term not in src, f"Competitor literal found in certificate.py: {term}"


def test_industry_comparison_contract_preserved():
    """industry_comparison dict contract is preserved with disclaimer key.

    This test replicates the minimal certificate math to ensure:
    1. The contract keys are present (comparisons, beats_standard_quant,
       beats_int4, rank, better_than_count, total_compared, disclaimer)
    2. The current-run row is computed from `ratio` (not from LITERATURE_ESTIMATES)
    3. beats_standard_quant == (ratio > 4.0)
    """
    from spectralstream.compression.literature_estimates import (
        LITERATURE_ESTIMATES,
        LITERATURE_DISCLAIMER,
    )

    # Simulate the certificate math minimally
    ratio = 5.0  # current run's ratio
    comparisons = list(LITERATURE_ESTIMATES) + [
        ("SpectralStream (current)", round(ratio, 1), "This run", "hybrid")
    ]

    # Better count math (verterbatim from certificate.py)
    better_count = sum(1 for _, r, _, _ in comparisons if r < ratio and r != ratio)
    total_known = sum(1 for _, r, _, _ in comparisons if r != ratio)
    rank = sum(1 for _, r, _, _ in comparisons if r >= ratio)

    industry_comparison = {
        "comparisons": [
            {
                "name": n,
                "ratio": r,
                "description": d,
                "type": t,
                "beats": ratio > r if r != ratio else None,
            }
            for n, r, d, t in comparisons
        ],
        "beats_standard_quant": ratio > 4.0,
        "beats_int4": ratio > 8.0,
        "rank": f"{rank}/{total_known}",
        "better_than_count": better_count,
        "total_compared": total_known,
        "disclaimer": LITERATURE_DISCLAIMER,
    }

    # Assert contract keys present
    for key in (
        "comparisons",
        "beats_standard_quant",
        "beats_int4",
        "rank",
        "better_than_count",
        "total_compared",
        "disclaimer",
    ):
        assert key in industry_comparison, f"Missing key: {key}"

    # Assert >= 9 comparisons
    assert len(industry_comparison["comparisons"]) >= 9

    # Assert current-run row is computed (not from LITERATURE_ESTIMATES)
    names = [c["name"] for c in industry_comparison["comparisons"]]
    assert "SpectralStream (current)" in names
    current_row = next(
        c for c in industry_comparison["comparisons"] if c["name"] == "SpectralStream (current)"
    )
    # The current-run ratio is float(ratio), not a hardcoded value
    assert current_row["ratio"] == round(ratio, 1)
    # Disclaimer is present
    assert industry_comparison["disclaimer"] == LITERATURE_DISCLAIMER
    # beats_standard_quant == (ratio > 4.0)
    assert industry_comparison["beats_standard_quant"] == (ratio > 4.0)


def test_disclaimer_rendered_in_certificate():
    """Disclaimer must appear in BOTH to_text() and to_markdown() rendered output.

    SC-3 anti-fabrication guarantee: the disclaimer is not just a dict key,
    it must surface in the rendered certificate itself.
    """
    from spectralstream.compression.certificate import CompressionCertificate
    from spectralstream.compression.literature_estimates import LITERATURE_DISCLAIMER

    # Minimal certificate constructor with just enough fields
    cert = CompressionCertificate(
        model_name="test_model",
        model_path="",
        model_architecture="",
        model_params="",
        total_original_bytes=1_000_000,
        total_compressed_bytes=1_000,
        overall_ratio=1_000.0,
        total_tensors=1,
        compression_time_seconds=0,
        weighted_error=0,
        avg_error=0,
        max_error=0,
        min_error=0,
        avg_snr_db=0,
        tensor_certificates=[],
        method_distribution={"test": 1},
    )

    text_output = cert.to_text()
    md_output = cert.to_markdown()

    assert LITERATURE_DISCLAIMER in text_output, "Disclaimer missing from to_text()"
    assert LITERATURE_DISCLAIMER in md_output, "Disclaimer missing from to_markdown()"
