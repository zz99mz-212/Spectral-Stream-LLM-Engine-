"""
Eval grader: measures WikiText-2 perplexity on base vs compressed models.

Wraps ``InferencePipeline.measure_perplexity`` and orchestrates the
base→close→compressed→close lifecycle to avoid holding two FP32 models in
RAM (per D-04 / D-05 / must_have: "base model is closed before compressed
model is loaded").
"""

from __future__ import annotations

from typing import Any

from spectralstream.inference.pipeline import InferenceConfig, InferencePipeline

from eval.artifact import build_eval_artifact
from eval.constants import DEFAULT_SEQ_LEN, DEFAULT_STRIDE


def run_ppl(
    model_path: str,
    test_tokens: list[int],
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: int = DEFAULT_STRIDE,
) -> tuple[float, int]:
    """Run perplexity measurement on a single model.

    Opens an ``InferencePipeline`` as a context manager (auto-closed on
    return, per must-have: "base model is closed before compressed model
    is loaded"), calls ``measure_perplexity``, and records the number of
    loaded layers as a guard against silent partial-model loads.

    Parameters
    ----------
    model_path : str
        Path to the model file (``.safetensors`` or ``.ssf``).
    test_tokens : list[int]
        Token ids to evaluate perplexity on.
    seq_len : int
        Sliding-window sequence length (default 2048).
    stride : int
        Slide stride (default 512).

    Returns
    -------
    tuple[float, int]
        ``(perplexity, layers_loaded)``.
    """
    with InferencePipeline(
        model_path, config=InferenceConfig(), use_unified=False
    ) as pipe:
        ppl = pipe.measure_perplexity(
            test_tokens, stride=stride, max_seq_len=seq_len
        )
        layers_loaded = len(pipe.tensor_names)
    return (float(ppl), layers_loaded)


def grade(
    base_path: str,
    compressed_path: str,
    test_tokens: list[int],
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: int = DEFAULT_STRIDE,
    threshold: float = 0.95,
    tokenizer_name: str = "default",
    method_name: str = "compressed",
) -> dict[str, Any]:
    """Grade compression quality by comparing base vs compressed perplexity.

    Orchestrates the full evaluate-close-evaluate sequence:
    1. Measure perplexity on the **base** model.
    2. Close the base pipeline (freeing its FP32 weights from RAM).
    3. Measure perplexity on the **compressed** model.
    4. Close the compressed pipeline.
    5. Build and return the D-09 artifact with ``recovery_ratio`` and
       ``gate_passed``.

    Both measurements use the *same* ``test_tokens``, ``seq_len``, and
    ``stride`` so the comparison is fair.

    Parameters
    ----------
    base_path : str
        Path to the original (uncompressed) model.
    compressed_path : str
        Path to the compressed ``.ssf`` model.
    test_tokens : list[int]
        Token ids for perplexity evaluation (same list for both models).
    seq_len : int
        Sliding-window sequence length.
    stride : int
        Slide stride.
    threshold : float
        Recovery gate threshold.
    tokenizer_name : str
        Tokenizer name for the artifact.
    method_name : str
        Compression method name for the artifact.

    Returns
    -------
    dict
        D-09 eval artifact dict.
    """
    # ── Base model PPL ────────────────────────────────────────────
    base_ppl, base_layers = run_ppl(base_path, test_tokens, seq_len, stride)

    # ── Compressed model PPL ──────────────────────────────────────
    compressed_ppl, compressed_layers = run_ppl(
        compressed_path, test_tokens, seq_len, stride
    )

    # ── Layers-loaded guard: use the minimum so we don't hide a
    #    partial-load on either side.
    layers_loaded = min(base_layers, compressed_layers)

    return build_eval_artifact(
        model=base_path,
        method=method_name,
        tokenizer=tokenizer_name,
        base_ppl=base_ppl,
        compressed_ppl=compressed_ppl,
        seq_len=seq_len,
        stride=stride,
        n_tokens=len(test_tokens),
        layers_loaded=layers_loaded,
        threshold=threshold,
    )
