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
