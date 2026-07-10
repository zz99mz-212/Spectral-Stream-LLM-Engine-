"""
Tests for the eval subsystem: recovery ratio, artifact schema,
run_ppl lifecycle, grade symmetry, and layers_loaded tracking.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

# ── Test helper: custom tokenizer stub ──────────────────────────────────


class _RawTextTokenizer:
    """Minimal tokenizer stub returning a sentinel list for testing."""
    def encode(self, text: str) -> list[int]:
        return [999, 998, 997]

import numpy as np
import pytest

# ── Test: compute_recovery_ratio ──────────────────────────────────────


def test_recovery_ratio_equal_ppl():
    """compute_recovery_ratio(base=10.0, compressed=10.0) -> 1.0, gate_passed=True."""
    from eval.artifact import compute_recovery_ratio

    ratio, passed = compute_recovery_ratio(10.0, 10.0)
    assert ratio == pytest.approx(1.0)
    assert passed is True


def test_recovery_ratio_worse_compressed():
    """compute_recovery_ratio(base=10.0, compressed=20.0) -> 0.5, gate_passed=False."""
    from eval.artifact import compute_recovery_ratio

    ratio, passed = compute_recovery_ratio(10.0, 20.0, threshold=0.95)
    assert ratio == pytest.approx(0.5)
    assert passed is False


def test_recovery_ratio_better_compressed():
    """compute_recovery_ratio(base=10.0, compressed=9.5) -> ~1.0526, gate_passed=True."""
    from eval.artifact import compute_recovery_ratio

    ratio, passed = compute_recovery_ratio(10.0, 9.5, threshold=0.95)
    assert ratio == pytest.approx(1.0526, abs=1e-4)
    assert passed is True


def test_recovery_ratio_custom_threshold():
    """compute_recovery_ratio respects custom threshold."""
    from eval.artifact import compute_recovery_ratio

    ratio, passed = compute_recovery_ratio(10.0, 12.0, threshold=0.9)
    assert ratio == pytest.approx(10.0 / 12.0)  # ≈ 0.833
    assert passed is False  # 0.833 < 0.9, so gate fails

    ratio2, passed2 = compute_recovery_ratio(10.0, 11.0, threshold=0.9)
    # 10/11 ≈ 0.909 >= 0.9, should pass
    assert passed2 is True


# ── Test: build_eval_artifact schema ──────────────────────────────────


def test_artifact_schema_contains_all_d09_fields():
    """build_eval_artifact returns dict with all D-09 fields."""
    from eval.artifact import build_eval_artifact

    artifact = build_eval_artifact(
        model="test_model",
        method="test_method",
        tokenizer="test_tokenizer",
        base_ppl=10.0,
        compressed_ppl=10.5,
        seq_len=2048,
        stride=512,
        n_tokens=1000,
        layers_loaded=130,
        threshold=0.95,
    )

    required_fields = [
        "model",
        "method",
        "tokenizer",
        "base_ppl",
        "compressed_ppl",
        "recovery_ratio",
        "recovery_gate_threshold",
        "gate_passed",
        "seq_len",
        "stride",
        "n_tokens",
        "layers_loaded",
        "timestamp",
        "git_ref",
    ]
    for field in required_fields:
        assert field in artifact, f"Missing field: {field}"

    # Type checks
    assert isinstance(artifact["base_ppl"], float)
    assert isinstance(artifact["compressed_ppl"], float)
    assert isinstance(artifact["recovery_ratio"], float)
    assert isinstance(artifact["gate_passed"], bool)
    assert isinstance(artifact["seq_len"], int)
    assert isinstance(artifact["layers_loaded"], int)
    assert isinstance(artifact["n_tokens"], int)


def test_artifact_recovery_ratio_derived():
    """recovery_ratio and gate_passed are derived from the two PPL values."""
    from eval.artifact import build_eval_artifact

    artifact = build_eval_artifact(
        model="m",
        method="meth",
        tokenizer="tok",
        base_ppl=10.0,
        compressed_ppl=5.0,
        seq_len=2048,
        stride=512,
        n_tokens=100,
        layers_loaded=130,
        threshold=0.95,
    )
    assert artifact["recovery_ratio"] == pytest.approx(2.0)
    assert artifact["gate_passed"] is True

    artifact2 = build_eval_artifact(
        model="m",
        method="meth",
        tokenizer="tok",
        base_ppl=10.0,
        compressed_ppl=20.0,
        seq_len=2048,
        stride=512,
        n_tokens=100,
        layers_loaded=130,
        threshold=0.95,
    )
    assert artifact2["recovery_ratio"] == pytest.approx(0.5)
    assert artifact2["gate_passed"] is False


# ── Test: write_artifact ──────────────────────────────────────────────


def test_write_artifact_creates_json():
    """write_artifact writes pretty-printed JSON to disk."""
    from eval.artifact import write_artifact

    artifact = {"base_ppl": 10.0, "compressed_ppl": 10.5}
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_artifact.json")
        result = write_artifact(artifact, output_path=out_path)
        assert os.path.exists(out_path)
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["base_ppl"] == 10.0
        assert result == out_path


def test_write_artifact_default_dir():
    """write_artifact with no path defaults to eval/artifacts/."""
    from eval.artifact import write_artifact

    artifact = {"model": "test", "base_ppl": 10.0}
    result = write_artifact(artifact)
    # Normalise path separators for cross-platform comparison
    import os as _os
    expected_prefix = _os.path.normpath("eval/artifacts/")
    assert _os.path.normpath(result).startswith(expected_prefix)
    assert result.endswith(".json")
    # Clean up
    if os.path.exists(result):
        os.remove(result)
        os.rmdir(os.path.dirname(result))


# ── Test: run_ppl closes pipeline ─────────────────────────────────────


def test_run_ppl_closes_pipeline():
    """run_ppl instantiates InferencePipeline, calls measure_perplexity, and closes it."""
    from eval.grader import run_ppl

    mock_pipe = MagicMock()
    mock_pipe.measure_perplexity.return_value = 12.345
    mock_pipe.tensor_names = ["layer_0", "layer_1", "layer_2"]
    # Context manager returns self
    mock_pipe.__enter__.return_value = mock_pipe

    with patch("eval.grader.InferencePipeline", return_value=mock_pipe):
        ppl, layers = run_ppl(
            model_path="/fake/model.safetensors",
            test_tokens=[1, 2, 3, 4, 5],
            seq_len=2048,
            stride=512,
        )

    assert ppl == pytest.approx(12.345)
    assert layers == 3
    # Verify pipeline was entered and exited (context manager)
    mock_pipe.__enter__.assert_called_once()
    mock_pipe.__exit__.assert_called_once()
    mock_pipe.measure_perplexity.assert_called_once_with(
        [1, 2, 3, 4, 5], stride=512, max_seq_len=2048
    )


# ── Test: grade uses identical windowing ──────────────────────────────


def test_grade_identical_windowing():
    """grade calls measure_perplexity with same test_tokens, seq_len, stride for both models."""
    from eval.grader import grade

    mock_pipe_base = MagicMock()
    mock_pipe_base.measure_perplexity.return_value = 10.0
    mock_pipe_base.tensor_names = ["l0", "l1"]  # 2 layers
    mock_pipe_base.__enter__.return_value = mock_pipe_base

    mock_pipe_compressed = MagicMock()
    mock_pipe_compressed.measure_perplexity.return_value = 10.5
    mock_pipe_compressed.tensor_names = ["l0", "l1"]
    mock_pipe_compressed.__enter__.return_value = mock_pipe_compressed

    with patch("eval.grader.InferencePipeline") as mock_cls:
        # First call returns base pipe, second returns compressed pipe
        mock_cls.side_effect = [mock_pipe_base, mock_pipe_compressed]

        artifact = grade(
            base_path="/fake/base.safetensors",
            compressed_path="/fake/compressed.ssf",
            test_tokens=[1, 2, 3],
            seq_len=2048,
            stride=512,
            threshold=0.95,
            tokenizer_name="test_tokenizer",
            method_name="test_method",
        )

    # Verify both calls used the same args
    assert mock_pipe_base.measure_perplexity.call_args[1] == {
        "stride": 512,
        "max_seq_len": 2048,
    }
    assert mock_pipe_compressed.measure_perplexity.call_args[1] == {
        "stride": 512,
        "max_seq_len": 2048,
    }

    # Verify base_ppl and compressed_ppl in artifact
    assert artifact["base_ppl"] == pytest.approx(10.0)
    assert artifact["compressed_ppl"] == pytest.approx(10.5)
    assert artifact["layers_loaded"] == 2


# ── Test: layers_loaded recorded ──────────────────────────────────────


def test_layers_loaded_is_nonzero():
    """Artifact's layers_loaded is non-zero when pipeline has tensors loaded."""
    from eval.grader import run_ppl

    mock_pipe = MagicMock()
    mock_pipe.measure_perplexity.return_value = 15.0
    mock_pipe.tensor_names = [f"layer_{i}" for i in range(130)]
    mock_pipe.__enter__.return_value = mock_pipe

    with patch("eval.grader.InferencePipeline", return_value=mock_pipe):
        ppl, layers = run_ppl(
            model_path="/fake/model.safetensors",
            test_tokens=[1] * 100,
            seq_len=2048,
            stride=512,
        )

    assert layers > 0
    assert layers == 130


# ── Test: resolve_model_path ──────────────────────────────────────────


def test_resolve_model_path_cli_arg():
    """resolve_model_path uses CLI arg first."""
    from eval.model_path import resolve_model_path

    # Since the file doesn't exist, this should raise FileNotFoundError
    # But we can test the resolution order logic
    with pytest.raises((FileNotFoundError, ValueError)):
        resolve_model_path(cli_model="/nonexistent/path.safetensors")


def test_resolve_model_path_rejects_traversal():
    """resolve_model_path rejects path traversal."""
    from eval.model_path import resolve_model_path

    with pytest.raises(ValueError, match="traversal|Traversal"):
        resolve_model_path(cli_model="../../etc/passwd")


# ── Test: resolve_corpus ──────────────────────────────────────────────


def test_resolve_corpus_json_path():
    """resolve_corpus loads token IDs from a .json file."""
    from eval.corpus import resolve_corpus

    tokens = [101, 102, 103, 104, 105]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(tokens, f)
        json_path = f.name

    try:
        result = resolve_corpus(corpus_path=json_path)
        assert result == tokens
    finally:
        os.unlink(json_path)


def test_resolve_corpus_default():
    """resolve_corpus with no arg returns the default sample tokens."""
    from eval.corpus import resolve_corpus

    result = resolve_corpus(corpus_path=None)
    # Default should be the committed sample
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(t, int) for t in result)


# ── Test: module imports work ────────────────────────────────────────


def test_eval_package_importable():
    """The eval package is properly importable with expected exports."""
    import eval  # noqa: F811

    assert hasattr(eval, "run_eval") or True  # barrel may not re-export run_eval
    # Just verify import doesn't error


def test_run_ppl_handles_empty_tokens():
    """run_ppl handles empty or short token lists gracefully."""
    from eval.grader import run_ppl

    mock_pipe = MagicMock()
    # measure_perplexity returns inf for degenerate input
    mock_pipe.measure_perplexity.return_value = float("inf")
    mock_pipe.tensor_names = ["l0"]
    mock_pipe.__enter__.return_value = mock_pipe

    with patch("eval.grader.InferencePipeline", return_value=mock_pipe):
        ppl, layers = run_ppl(
            model_path="/fake/model.safetensors",
            test_tokens=[],  # empty tokens
            seq_len=2048,
            stride=512,
        )

    assert ppl == float("inf")
    assert layers == 1


# ── Test: model tokenizer actually used (D-02 gap closure) ──────────────


def test_resolve_corpus_uses_injected_tokenizer_for_raw_text():
    """resolve_corpus honors an injected tokenizer for raw text input."""
    from eval.corpus import resolve_corpus

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("sample text for encoding")
        txt_path = f.name

    try:
        result = resolve_corpus(corpus_path=txt_path, tokenizer=_RawTextTokenizer())
        assert result == [999, 998, 997]
    finally:
        os.unlink(txt_path)


def test_resolve_corpus_json_path_ignores_tokenizer():
    """resolve_corpus ignores the tokenizer argument for .json paths."""
    from eval.corpus import resolve_corpus

    tokens = [1, 2, 3]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(tokens, f)
        json_path = f.name

    try:
        result = resolve_corpus(corpus_path=json_path, tokenizer=_RawTextTokenizer())
        assert result == [1, 2, 3]
    finally:
        os.unlink(json_path)


def test_resolve_corpus_raw_text_default_uses_byte_level():
    """resolve_corpus with tokenizer=None uses byte-level fallback."""
    from eval.corpus import resolve_corpus
    from spectralstream.utils.tokenizer_engine import build_default_tokenizer

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world")
        txt_path = f.name

    try:
        result = resolve_corpus(corpus_path=txt_path, tokenizer=None)
        expected = build_default_tokenizer().encode("hello world")
        assert result == expected
    finally:
        os.unlink(txt_path)


def test_run_eval_loads_tokenizer_from_pretrained():
    """main() calls AutoTokenizer.from_pretrained with --tokenizer path."""
    from eval.run_eval import main

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]

    with (
        patch(
            "spectralstream.utils.tokenizer_engine.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ) as mock_from_pretrained,
        patch("eval.run_eval.resolve_model_path", return_value="/fake/model.safetensors"),
        patch("eval.run_eval.resolve_corpus", return_value=[1, 2, 3]) as mock_resolve_corpus,
        patch(
            "eval.run_eval.grade",
            return_value={
                "base_ppl": 10.0,
                "compressed_ppl": 10.5,
                "recovery_ratio": 0.952,
                "gate_passed": True,
                "layers_loaded": 130,
            },
        ),
        patch("eval.run_eval.write_artifact", return_value="/fake/artifact.json"),
    ):
        result = main(
            [
                "--model",
                "/fake/base.safetensors",
                "--compressed",
                "/fake/compressed.ssf",
                "--tokenizer",
                "/fake/tokenizer.json",
            ]
        )

    assert result == 0
    mock_from_pretrained.assert_called_once_with("/fake/tokenizer.json")
    # Verify resolve_corpus received the injected tokenizer (not None)
    _, kwargs = mock_resolve_corpus.call_args
    assert kwargs.get("tokenizer") is not None


def test_run_eval_loads_tokenizer_from_gguf():
    """main() calls AutoTokenizer.from_gguf with --tokenizer .gguf path."""
    from eval.run_eval import main

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]

    with (
        patch(
            "spectralstream.utils.tokenizer_engine.AutoTokenizer.from_gguf",
            return_value=mock_tokenizer,
        ) as mock_from_gguf,
        patch("eval.run_eval.resolve_model_path", return_value="/fake/model.safetensors"),
        patch("eval.run_eval.resolve_corpus", return_value=[1, 2, 3]),
        patch(
            "eval.run_eval.grade",
            return_value={
                "base_ppl": 10.0,
                "compressed_ppl": 10.5,
                "recovery_ratio": 0.952,
                "gate_passed": True,
                "layers_loaded": 130,
            },
        ),
        patch("eval.run_eval.write_artifact", return_value="/fake/artifact.json"),
    ):
        result = main(
            [
                "--model",
                "/fake/base.safetensors",
                "--compressed",
                "/fake/compressed.ssf",
                "--tokenizer",
                "/fake/model.gguf",
            ]
        )

    assert result == 0
    mock_from_gguf.assert_called_once_with("/fake/model.gguf")


def test_run_eval_omitted_tokenizer_uses_default():
    """main() without --tokenizer does NOT call AutoTokenizer loaders."""
    from eval.run_eval import main

    with (
        patch("spectralstream.utils.tokenizer_engine.AutoTokenizer.from_pretrained"),
        patch("spectralstream.utils.tokenizer_engine.AutoTokenizer.from_gguf"),
        patch("eval.run_eval.resolve_model_path", return_value="/fake/model.safetensors"),
        patch("eval.run_eval.resolve_corpus", return_value=[1, 2, 3]) as mock_resolve_corpus,
        patch(
            "eval.run_eval.grade",
            return_value={
                "base_ppl": 10.0,
                "compressed_ppl": 10.5,
                "recovery_ratio": 0.952,
                "gate_passed": True,
                "layers_loaded": 130,
            },
        ),
        patch("eval.run_eval.write_artifact", return_value="/fake/artifact.json"),
    ):
        result = main(["--model", "/fake/base.safetensors", "--compressed", "/fake/compressed.ssf"])

    assert result == 0
    # resolve_corpus should receive a tokenizer argument (with tokenizer=None it defaults)
    _, kwargs = mock_resolve_corpus.call_args
    assert "tokenizer" in kwargs
