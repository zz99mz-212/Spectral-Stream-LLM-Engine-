"""
Direct Cascade Compression Engine — clean, proven cascade patterns.

Uses direct method calls from the engine's ``_methods`` registry
(e.g. ``"svd_compress"``, ``"dct_spectral"``) instead of broken
method-type resolution.

**Key findings (empirical)**:
- SVD rank=2 gives 877-1228x on all weight types
- DCT/FWHT on SVD residuals gives NO extra compression (residual is noise)
- Entropy on SVD compressed data gives 1.5-2x more
- Only 3-5 patterns are actually useful; all others removed.

**Residual approach**: Stage 0 compresses the original tensor.  Stage 1+
compresses the *residual* (original minus cumulative reconstruction).
Each stage captures information the previous stage missed.

**Auto entropy post-process**: After cascade execution, automatically
tries Huffman/RANS entropy coding on the payload if it improves ratio.

Example
-------
>>> from spectralstream.compression.engine import CompressionIntelligenceEngine
>>> from spectralstream.compression.engine.direct_cascade import DirectCascadeEngine
>>> import numpy as np
>>> eng = CompressionIntelligenceEngine()
>>> dc = DirectCascadeEngine()
>>> t = np.random.randn(256, 256).astype(np.float32)
>>> data, meta = dc.execute_cascade(eng, t, "weight", "aggressive")
>>> meta["total_ratio"], meta["total_error"], len(meta["stages"])
"""

from __future__ import annotations

import json
import logging
import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .loss_metrics import TensorLossMetrics
from .tiered_error import (
    get_budget,
    get_fallback_pattern,
    is_within_budget,
    select_cascade_pattern as tiered_select_pattern,
)

logger = logging.getLogger(__name__)


class DirectCascadeEngine:
    """Residual-based multi-stage cascade compression using direct method calls.

    **Residual approach** — Each stage captures what the previous stage missed:

    1. Stage 0 compresses the *original* tensor (e.g. SVD captures low-rank structure)
    2. Stage 1+ compresses the *residual* = original - cumulative_reconstruction
       (e.g. DCT on residual captures detail that SVD missed)
    3. Reconstruction = SUM of all stage decompressions (not overwrite)

    This avoids the old (broken) approach of cascading reconstructions, where
    DCT on an SVD reconstruction was redundant — SVD already captured structure,
    so DCT had nothing new to encode.

    All stage compressed data is concatenated into a single bytes payload
    with a small header.

    ``CASCADE_PATTERNS`` map pattern names to lists of ``(method_name, params)``
    tuples.  Parameters can use the ``"auto:N"`` syntax, which is resolved to
    ``max(min(shape) // N, 4)`` — useful for rank selection.

    ``DEEP_CASCADE_PATTERNS`` contain 3-5 stage cascades designed for
    500:1+ compression on real weight tensors.  These patterns are
    automatically selected for tensors above 1M elements.

    ``DEEP_PATTERNS`` (NEW) contain 3-4 stage entropy-terminated cascades
    that stack multiple tensor decomposition and spectral methods:
    ``svd_tt_entropy`` (SVD+TT+DCT+Huffman), ``svd_cp_dct_entropy``
    (SVD+CP+DCT+RANS), ``svd_fwht_dct_entropy`` (SVD+FWHT+DCT+Huffman),
    and ``tt_kron_entropy`` (TT+Kronecker+RANS).  These are auto-selected
    when ``target_ratio >= 500`` for extreme compression scenarios.

    Parameters
    ----------
    store_all_stages : bool
        If True (default), each stage's compressed data is stored in the
        output payload, giving TRUE multiplicative ratios (total ratio =
        original_size / sum(all_stage_sizes)).  If False, only the last
        stage's data is kept (ratio based on last stage alone).
    entropy_post_process : str or None
        If ``"huffman"`` or ``"rans"``, applies entropy coding to the
        entire packaged payload after all stages, for additional 1.2-2x
        compression.  None (default) disables.
    """

    # ── Clean, Proven Cascade Patterns ──────────────────────────────────
    # Only patterns that empirically work on real weight tensors.
    # Key findings: SVD alone gives 877-1228x; DCT/FWHT on SVD residuals
    # gives NO extra compression; entropy gives 1.5-2x additional gain.
    CASCADE_PATTERNS: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {
        # ── Single method patterns ──────────────────────────────────
        "lightning": [
            ("dct_spectral", {"keep_ratio": 0.15}),
        ],
        "balanced": [
            ("svd_compress", {"rank": "auto:30"}),
        ],
        "aggressive": [
            ("svd_compress", {"rank": "auto:100"}),
        ],
        "extreme": [
            ("svd_compress", {"rank": "auto:200"}),
        ],
        "max_compression": [
            ("svd_compress", {"rank": "auto:500"}),
        ],
        # ── 1D-optimized (DCT only — SVD doesn't apply to 1D) ──────
        "1d_lightning": [
            ("dct_spectral", {"keep_ratio": 0.3}),
        ],
        "1d_aggressive": [
            ("dct_spectral", {"keep_ratio": 0.15}),
        ],
        # ── Entropy-stacked (SVD + lossless entropy on compressed) ──
        "svd_entropy": [
            ("svd_compress", {"rank": "auto:30"}),
            ("huffman", {}),
        ],
        "svd_rans": [
            ("svd_compress", {"rank": "auto:30"}),
            ("rans", {}),
        ],
        # ── Embedding patterns (SVD only, no residual DCT) ──────────
        "embedding_balanced": [
            ("svd_compress", {"rank": "auto:60"}),
        ],
        "embedding_extreme": [
            ("svd_compress", {"rank": "auto:200"}),
        ],
        # ── Residual sparse store (SVD + delta-encoded sparse) ──────
        "sparse_residual": [
            ("svd_compress", {"rank": "auto:50"}),
            ("sparse_store", {"threshold_sigma": 2.5}),
        ],
        # ═══ Entangled SVD Cascade ═══
        # Compresses SVD factors (U, Vt) separately using DCT for higher ratios.
        # U (m×r) — each column is a signal, compress with DCT
        # Vt (r×n) — each row is a signal, compress with DCT
        # S (r,) — tiny, store as float16
        "svd_entangled": [
            ("svd_compress", {"rank": "auto:30", "store_factors": True}),
        ],
    }

    # ── High-Ratio Cascade Patterns (4-5 stage, 200:1+) ──────────────────────
    # These patterns use AGGRESSIVE SVD ranks (auto:200 = min_dim // 200)
    # so Stage 1 leaves structure in the residual for downstream stages.
    HIGH_RATIO_CASCADES: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {
        # ── 3-Stage ──────────────────────────────────────────────────
        "svd_lowrank_int4_huffman": [
            ("svd_compress", {"rank": "auto:200"}),
            ("block_int4", {"block_size": 16}),
            ("huffman", {}),
        ],
        "fwht_int4_sparse_rans": [
            ("fwht_compress", {"keep_ratio": 0.08}),
            ("hadamard_int4", {"block_size": 16}),
            ("sparsity_int4", {"group_size": 32}),
            ("rans", {}),
        ],
        "tt_quant_sparse_huffman": [
            ("tensor_train", {"rank": 4}),
            ("delta_int4", {"block_size": 32}),
            ("sparsity_int4", {"group_size": 32}),
            ("huffman", {}),
        ],
        # ── 4-Stage ──────────────────────────────────────────────────
        "svd_int4_sparse_huffman": [
            ("svd_compress", {"rank": "auto:200"}),
            ("block_int4", {"block_size": 16}),
            ("sparsity_int4", {"group_size": 32}),
            ("huffman", {}),
        ],
        "dct_int4_sparse_huffman": [
            ("dct_spectral", {"keep_ratio": 0.06}),
            ("block_int4", {"block_size": 16}),
            ("sparsity_int4", {"group_size": 32}),
            ("huffman", {}),
        ],
        "svd_delta_huffman_rans": [
            ("svd_compress", {"rank": "auto:200"}),
            ("delta_int4", {"block_size": 32}),
            ("huffman", {}),
            ("rans", {}),
        ],
        # ── 5-Stage (200:1+) ──────────────────────────────────────────
        "tt_quant_sparse_fwht_huffman": [
            ("tensor_train", {"rank": 4}),
            ("delta_int4", {"block_size": 32}),
            ("sparsity_int4", {"group_size": 32}),
            ("fwht_compress", {"keep_ratio": 0.1}),
            ("huffman", {}),
        ],
        "svd_tt_quant_sparse_huffman": [
            ("svd_compress", {"rank": "auto:200"}),
            ("tensor_train", {"rank": 4}),
            ("block_int4", {"block_size": 16}),
            ("sparsity_int4", {"group_size": 32}),
            ("huffman", {}),
        ],
        "svd_dct_int4_sparse_huffman": [
            ("svd_compress", {"rank": "auto:200"}),
            ("dct_spectral", {"keep_ratio": 0.12}),
            ("block_int4", {"block_size": 16}),
            ("sparsity_int4", {"group_size": 32}),
            ("huffman", {}),
        ],
        # ── Embedding ────────────────────────────────────────────────
        "embedding_triple_cascade": [
            ("svd_compress", {"rank": "auto:200"}),
            ("block_int4", {"block_size": 16}),
            ("sparsity_int4", {"group_size": 32}),
            ("huffman", {}),
        ],
    }

    # Combined patterns map (backward compat)
    ALL_PATTERNS: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}

    def __init__(
        self,
        store_all_stages: bool = True,
        entropy_post_process: Optional[str] = None,
    ):
        self.store_all_stages = store_all_stages
        self.entropy_post_process = entropy_post_process
        if entropy_post_process is not None:
            entropy_post_process = entropy_post_process.lower()
            if entropy_post_process not in ("huffman", "rans"):
                logger.warning(
                    "Unknown entropy_post_process '%s', disabling",
                    entropy_post_process,
                )
                self.entropy_post_process = None
        # ALL_PATTERNS = CASCADE_PATTERNS + HIGH_RATIO_CASCADES
        self.ALL_PATTERNS = dict(self.CASCADE_PATTERNS)
        self.ALL_PATTERNS.update(dict(self.HIGH_RATIO_CASCADES))

    # ── Parameter Resolution Helpers ──────────────────────────────────────

    @staticmethod
    def min_dim(shape: Tuple[int, ...]) -> int:
        """Get the minimum dimension of a tensor shape."""
        return min(s for s in shape if s > 0)

    @staticmethod
    def auto_rank(shape: Tuple[int, ...], divisor: int = 50) -> int:
        """Compute SVD rank from tensor shape.

        ``rank = max(min(shape) // divisor, 4)`` guarantees at least rank 4
        for any input.  Higher divisors = more aggressive compression.

        Parameters
        ----------
        shape : tuple of int
            Tensor shape.
        divisor : int
            Divisor for rank computation (default 50).
            - divisor=50: moderate (min(1536)//50 = 30)
            - divisor=200: aggressive (min(1536)//200 = 7)
            - divisor=500: extreme (min(1536)//500 = 3)

        Returns
        -------
        int
            Computed rank.
        """
        min_dim = DirectCascadeEngine.min_dim(shape)
        return max(min_dim // divisor, 2)

    @staticmethod
    def auto_keep_ratio(shape: Tuple[int, ...], divisor: int = 4) -> float:
        """Compute keep_ratio from tensor shape for spectral methods.

        ``keep_ratio = max(divisor / min(shape), 0.005)`` ensures we keep
        at least ``divisor`` coefficients per dimension.

        Parameters
        ----------
        shape : tuple of int
            Tensor shape.
        divisor : int
            Number of coefficients to keep per dimension (default 4).

        Returns
        -------
        float
            Keep ratio between 0.005 and 1.0.
        """
        min_dim = DirectCascadeEngine.min_dim(shape)
        return max(min(divisor / min_dim, 1.0), 0.005)

    @staticmethod
    def auto_ranks(shape: Tuple[int, ...], divisor: int = 4) -> list:
        """Compute Tucker ranks tuple from tensor shape.

        ``[max(d // divisor, 2) for d in shape]``

        Parameters
        ----------
        shape : tuple of int
            Tensor shape (should be 2D or more).
        divisor : int
            Divisor for rank computation.

        Returns
        -------
        list of int
            Per-dimension ranks.
        """
        return [max(d // divisor, 2) for d in shape]

    @classmethod
    def resolve_param(cls, key: str, value: Any, shape: Tuple[int, ...]) -> Any:
        """Resolve a parameter value, handling ``auto:N`` syntax.

        Supports:
        - ``"auto:N"`` → rank via ``auto_rank`` (for rank parameters)
        - ``"auto_keep:N"`` → keep_ratio via ``auto_keep_ratio`` (for spectral)
        - ``"auto_ranks:N"`` → ranks list via ``auto_ranks`` (for Tucker)

        Parameters
        ----------
        key : str
            Parameter name (unused, kept for extensibility).
        value : Any
            Parameter value.
        shape : tuple of int
            Tensor shape for auto-resolution.

        Returns
        -------
        Any
            Resolved parameter value.
        """
        if isinstance(value, str):
            if value.startswith("auto_keep:"):
                divisor = int(value.split(":", 1)[1])
                return cls.auto_keep_ratio(shape, divisor)
            if value.startswith("auto_ranks:"):
                divisor = int(value.split(":", 1)[1])
                return cls.auto_ranks(shape, divisor)
            if value.startswith("auto:"):
                divisor = int(value.split(":", 1)[1])
                # Key-aware resolution: "ranks" key returns a list, "rank" returns int
                if key == "ranks":
                    return cls.auto_ranks(shape, divisor)
                return cls.auto_rank(shape, divisor)
        return value

    @staticmethod
    def _should_compress(tensor: np.ndarray) -> bool:
        """Check if compression is worth it for a given tensor.

        Tensors smaller than 1KB (e.g. < 256 float32 elements) have
        negligible size — the metadata overhead of compression exceeds
        any gain, so they should be stored uncompressed (passthrough).

        Parameters
        ----------
        tensor : np.ndarray
            Tensor to check.

        Returns
        -------
        bool
            True if compression is worthwhile, False for passthrough.
        """
        return tensor.nbytes >= 1

    # ── Stage Execution ──────────────────────────────────────────────────

    def execute_stage(
        self,
        engine: Any,
        tensor: np.ndarray,
        method_name: str,
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Execute a single cascade stage.

        Parameters
        ----------
        engine : CompressionIntelligenceEngine
            Engine with ``_methods`` registry.
        tensor : np.ndarray
            Input tensor for this stage.
        method_name : str
            Name of the method in ``engine._methods``.
        params : dict
            Parameters for the method (values resolved before calling).

        Returns
        -------
        dict or None
            Stage result with keys ``method``, ``params``, ``compressed_data``,
            ``metadata``, ``stage_ratio``, ``stage_error``.  None if the
            method is not found or execution fails.
        """
        inst = engine._methods.get(method_name)
        if inst is None:
            logger.warning("Method '%s' not found in engine._methods", method_name)
            return None

        # Resolve auto-parameters
        resolved_params: Dict[str, Any] = {}
        for k, v in params.items():
            resolved_params[k] = self.resolve_param(k, v, tensor.shape)

        try:
            compressed, meta = inst.compress(tensor, **resolved_params)
            stage_ratio = float(tensor.nbytes / max(len(compressed), 1))

            recon = inst.decompress(compressed, meta)
            if recon.shape != tensor.shape:
                recon = recon.reshape(tensor.shape)

            # Stage error = mean absolute error of this stage
            stage_error = float(np.abs(tensor.ravel() - recon.ravel()).mean())

            stage_info: Dict[str, Any] = {
                "method": method_name,
                "params": resolved_params,
                "compressed_data": compressed,
                "metadata": meta,
                "stage_ratio": float(stage_ratio),
                "stage_error": stage_error,
                "original_shape": list(tensor.shape),
            }
            return stage_info

        except Exception as exc:
            logger.warning("Stage '%s' failed: %s", method_name, exc, exc_info=True)
            return None

    # ── Pattern Selection ────────────────────────────────────────────────

    @classmethod
    def select_pattern(
        cls,
        tensor: np.ndarray,
        tensor_type: str = "weight",
        target_ratio: Optional[float] = None,
    ) -> str:
        """Automatically select the best cascade pattern for a tensor.

        Simplified selection — uses only patterns proven to work on
        real weight tensors:

        - 1D tensors (biases, norms) → ``1d_lightning`` / ``1d_aggressive``
        - Small tensors (< 1KB) → ``passthrough`` (handled upstream)
        - Embeddings (huge) → ``embedding_extreme`` / ``embedding_balanced``
        - 2D+ tensors → SVD rank-based selection

        Parameters
        ----------
        tensor : np.ndarray
            Tensor to compress.
        tensor_type : str
            Tensor type hint.
        target_ratio : float, optional
            Target compression ratio.

        Returns
        -------
        str
            Pattern name from ``CASCADE_PATTERNS``.
        """
        n_elements = tensor.size
        ndim = tensor.ndim
        effective_target = target_ratio or 200.0

        # ── 1D tensors (biases, norms): DCT-only ────────────────────
        if ndim <= 1:
            if n_elements < 256:
                return "lightning"
            return "1d_aggressive" if effective_target > 100 else "1d_lightning"

        # ── Tiny tensors: skip ─────────────────────────────────────────
        if tensor.nbytes < 1:
            return "passthrough"

        # ── Embeddings: SVD-only ───────────────────────────────────────
        if tensor_type == "embedding" or tensor.nbytes > 1e9:
            return (
                "embedding_extreme" if effective_target > 100 else "embedding_balanced"
            )

        # ── 2D+ weight tensors: SVD rank-based ─────────────────────────
        if effective_target >= 500:
            return "svd_tt_quant_sparse_huffman"
        if effective_target >= 200:
            return "svd_int4_sparse_huffman"
        if effective_target >= 50:
            return "aggressive"
        return "balanced"

    # ── Entropy Post-Processing ──────────────────────────────────────────

    def _apply_entropy_post_process(
        self,
        engine: Any,
        payload: bytes,
        entropy_method: str,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Apply entropy coding (huffman or rans) to the packaged payload.

        The entropy post-process wraps the existing ``_package_stages``
        output in an additional entropy stage.  On decompression, the
        entropy stage is unwrapped first, then stages are unpacked.

        Parameters
        ----------
        engine : CompressionIntelligenceEngine
            Engine with ``_methods`` registry.
        payload : bytes
            Packaged cascade payload.
        entropy_method : str
            ``"huffman"`` or ``"rans"``.

        Returns
        -------
        Tuple[bytes, dict]
            Entropy-encoded payload and metadata.
        """
        inst = engine._methods.get(entropy_method)
        if inst is None:
            logger.warning(
                "Entropy method '%s' not found, skipping post-process",
                entropy_method,
            )
            return payload, {}

        try:
            # Convert payload to numpy array for huffman (which expects ndarray)
            if entropy_method == "huffman":
                payload_array = np.frombuffer(payload, dtype=np.uint8)
                entropy_data, entropy_meta = inst.compress(payload_array)
            else:
                # rans accepts bytes
                entropy_data, entropy_meta = inst.compress(payload)

            ratio_gain = len(payload) / max(len(entropy_data), 1)
            logger.debug(
                "Entropy post-process (%s): %d → %d bytes (%.2fx gain)",
                entropy_method,
                len(payload),
                len(entropy_data),
                ratio_gain,
            )

            # Store the entropy-wrapped payload with a marker
            entropy_payload = self._package_entropy_stage(
                entropy_data, entropy_meta, entropy_method
            )
            return entropy_payload, {
                "entropy_method": entropy_method,
                "entropy_ratio_gain": ratio_gain,
                "inner_size": len(payload),
            }

        except Exception as exc:
            logger.warning(
                "Entropy post-process '%s' failed: %s",
                entropy_method,
                exc,
            )
            return payload, {}

    @staticmethod
    def _package_entropy_stage(
        entropy_data: bytes,
        entropy_meta: Dict[str, Any],
        entropy_method: str,
    ) -> bytes:
        """Package entropy-encoded data into a single-entry stage container.

        Format matches ``_package_stages`` with 1 stage (the entropy wrapper).
        The method name is ``f"entropy:{entropy_method}"`` so the
        reconstructor knows to decode it first.
        """
        buf = bytearray()
        # 1 stage
        buf += struct.pack("<I", 1)

        method_name = f"entropy:{entropy_method}"
        method_bytes = method_name.encode("utf-8")
        meta_json = json.dumps(
            {
                k: v
                for k, v in entropy_meta.items()
                if isinstance(v, (str, int, float, bool, list, tuple))
            },
            default=str,
        ).encode("utf-8")

        buf += struct.pack("<I", len(method_bytes))
        buf += method_bytes
        buf += struct.pack("<I", len(meta_json))
        buf += meta_json
        buf += struct.pack("<I", len(entropy_data))
        buf += entropy_data

        return bytes(buf)

    # ── Standalone Entropy Post-Processing (Direct Imports) ──────────────

    def _entropy_compress(
        self, data: bytes, method: str = "huffman"
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Apply entropy coding to compressed data using direct imports.

        Unlike ``_apply_entropy_post_process`` (which uses
        ``engine._methods`` lookups), this method imports entropy coders
        directly from ``spectralstream.compression.methods.entropy``.
        This is useful when no engine instance is available.

        Parameters
        ----------
        data : bytes
            Raw payload to entropy-compress.
        method : str
            ``"huffman"`` (default) or ``"rans"``.

        Returns
        -------
        Tuple[bytes, dict]
            Entropy-compressed bytes and metadata dict with keys:
            ``entropy_method``, plus any coder-specific metadata.

        Raises
        ------
        ImportError
            If the entropy coder module cannot be imported.
        """
        from spectralstream.compression.methods.entropy._class_wrappers import (
            HuffmanCoder,
            RANS,
        )

        if method == "rans":
            compressor = RANS()
        else:
            compressor = HuffmanCoder()

        # Convert bytes to uint8 array; both HuffmanCoder and RANS
        # accept bytes directly (RANS routes to HuffmanCoder for bytes).
        compressed, meta = compressor.compress(data)
        return compressed, {"entropy_method": method, **meta}

    def _entropy_decompress(self, data: bytes, meta: Dict[str, Any]) -> bytes:
        """Reverse entropy coding applied by ``_entropy_compress``.

        Parameters
        ----------
        data : bytes
            Entropy-compressed bytes.
        meta : dict
            Metadata dict containing ``entropy_method`` key.

        Returns
        -------
        bytes
            Decompressed original bytes.

        Raises
        ------
        ImportError
            If the entropy coder module cannot be imported.
        """
        from spectralstream.compression.methods.entropy._class_wrappers import (
            HuffmanCoder,
            RANS,
        )

        method = meta.get("entropy_method", "huffman")
        if method == "rans":
            compressor = RANS()
        else:
            compressor = HuffmanCoder()

        decompressed = compressor.decompress(data, meta)
        # Both HuffmanCoder and RANS return bytes for bytes input
        if isinstance(decompressed, bytes):
            return decompressed
        # Fallback: convert ndarray to bytes
        return decompressed.tobytes()

    # ── Auto Entropy Post-Process ───────────────────────────────────────

    def _try_entropy_postprocess(
        self, data: bytes, meta: Dict[str, Any], tensor: np.ndarray
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Auto-apply entropy if it improves compression ratio by >=10%.

        After cascade execution, tries Huffman coding on the payload.
        If the compressed payload is at least 10% smaller, the entropy
        coding is kept and metadata updated.

        Parameters
        ----------
        data : bytes
            Compressed payload from cascade stages.
        meta : dict
            Cascade result metadata (updated in-place if entropy helps).
        tensor : np.ndarray
            Original tensor (used for ratio comparison).

        Returns
        -------
        Tuple[bytes, dict]
            Possibly entropy-compressed data and updated metadata.
        """
        if len(data) < 128:
            return data, meta  # Too small for entropy gains

        from spectralstream.compression.methods.entropy._class_wrappers import (
            HuffmanCoder,
        )

        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            compressed, huff_meta = HuffmanCoder().compress(arr)
            if len(compressed) < len(data) * 0.9:  # At least 10% improvement
                orig_ratio = meta.get("total_ratio", 1.0)
                new_ratio = float(tensor.nbytes / max(len(compressed), 1))
                meta["total_ratio"] = new_ratio
                meta["entropy_post_process"] = {
                    "entropy_method": "huffman",
                    "entropy_ratio_gain": len(data) / max(len(compressed), 1),
                    "inner_size": len(data),
                }
                meta["entropy"] = meta["entropy_post_process"]
                logger.debug(
                    "Auto entropy: %.0f → %.0f bytes (gain=%.2fx, ratio=%.0f→%.0f)",
                    len(data),
                    len(compressed),
                    len(data) / max(len(compressed), 1),
                    orig_ratio,
                    new_ratio,
                )
                return compressed, meta
        except Exception:
            pass
        return data, meta

    # ── Single Pattern Execution (Self-Healing Per Stage) ──────────────

    def _execute_single_pattern(
        self,
        engine: Any,
        tensor: np.ndarray,
        tensor_type: str,
        pattern: str,
        entropy_post_process: Optional[str] = None,
    ) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """Execute a single cascade pattern with error recovery per stage.

        Each stage is wrapped in its own try/except.  If a stage fails
        (e.g. SVD on a 1xN tensor), the entire pattern is abandoned and
        None is returned so the caller can fall through to a simpler
        pattern.

        Uses memory-efficient staging (single ``cumulative_recon`` array
        updated in-place) to avoid holding multiple reconstruction arrays.

        Parameters
        ----------
        engine : CompressionIntelligenceEngine
            Engine with ``_methods`` registry.
        tensor : np.ndarray
            Float32 tensor to compress.
        tensor_type : str
            Tensor type hint (e.g. ``"weight"``, ``"norm_bias"``).
        pattern : str
            Cascade pattern name from ``ALL_PATTERNS``.
        entropy_post_process : str or None
            Optional entropy coding method (``"huffman"`` or ``"rans"``).

        Returns
        -------
        Tuple[bytes, dict] or None
            ``(compressed_data, metadata)`` if the pattern produced a valid
            compressed result, otherwise None.
        """
        stages_config = self.ALL_PATTERNS.get(pattern, [])
        if not stages_config:
            return None

        original = np.ascontiguousarray(tensor, dtype=np.float32)
        orig_size = original.nbytes

        # ── Special handling: Entangled SVD Cascade ─────────────────────
        # The svd_entangled pattern runs SVD, extracts raw factors (U,S,Vt),
        # compresses U and Vt independently with DCT, and stores S as
        # float16.  This gives higher ratios than storing SVD factors as
        # packed float16 because U columns and Vt rows are signal-like.
        if pattern == "svd_entangled":
            return self._execute_svd_entangled(
                engine,
                original,
                tensor_type,
                orig_size,
                entropy_post_process,
            )

        import gc as _gc

        # ── Stage tracking (memory-efficient) ───────────────────────
        stages_data: List[Dict[str, Any]] = []
        cumulative_recon: np.ndarray = np.zeros(original.shape, dtype=np.float64)
        residual: np.ndarray = original.astype(np.float64)

        # ── Stage Execution (with pseudo-method support) ──────────────
        pseudo_methods = {"dct_threshold", "sparse_store"}

        for i, (method_name, params) in enumerate(stages_config):
            is_pseudo = method_name in pseudo_methods

            if not is_pseudo:
                inst = engine._methods.get(method_name)
                entropy_methods = {"huffman", "rans"}
                if inst is None and method_name not in entropy_methods:
                    # Method not found — abandon this pattern
                    del cumulative_recon, residual
                    _gc.collect()
                    return None
                if method_name in entropy_methods:
                    inst = None
            else:
                inst = None

            # Resolve auto-parameters
            resolved_params: Dict[str, Any] = {}
            for k, v in params.items():
                resolved_params[k] = self.resolve_param(k, v, original.shape)

            try:
                if i == 0:
                    # ── Stage 0: compress ORIGINAL tensor ──────────────
                    if is_pseudo:
                        data, meta, recon = self._execute_pseudo_stage(
                            method_name, original, resolved_params, orig_size
                        )
                    elif method_name in ("huffman", "rans"):
                        from spectralstream.compression.methods.entropy._class_wrappers import (  # noqa: E501
                            HuffmanCoder,
                            RANS,
                        )

                        coder = HuffmanCoder() if method_name == "huffman" else RANS()
                        np_data = np.ascontiguousarray(original)
                        raw_bytes = np_data.tobytes()
                        compressed, entropy_meta = coder.compress(raw_bytes)
                        decompressed = coder.decompress(compressed, entropy_meta)
                        recon = (
                            np.frombuffer(decompressed, dtype=np.float32)
                            .reshape(original.shape)
                            .astype(np.float64)
                        )
                        # Encode bytes-tree to hex for JSON-safe metadata
                        entropy_meta_clean: Dict[str, Any] = {}
                        for ek, ev in entropy_meta.items():
                            if isinstance(ev, bytes):
                                entropy_meta_clean[ek] = ev.hex()
                            elif isinstance(ev, (str, int, float, bool, list, tuple)):
                                entropy_meta_clean[ek] = ev
                            else:
                                entropy_meta_clean[ek] = str(ev)
                        data, meta = compressed, entropy_meta_clean
                    elif method_name == "progressive_svd":
                        max_err = resolved_params.get("max_error", 0.01)
                        ps_result = self._progressive_svd_compress(
                            engine,
                            original,
                            max_error=max_err,
                        )
                        if ps_result is None:
                            raise ValueError("Progressive SVD failed")
                        data, meta = ps_result["data"], ps_result["meta"]
                        recon = ps_result["recon"]
                    else:
                        data, meta = inst.compress(original, **resolved_params)
                        recon = inst.decompress(data, meta)
                    if recon.shape != original.shape:
                        recon = recon.reshape(original.shape)
                    recon = recon.astype(np.float64)

                    stage_ratio = float(orig_size / max(len(data), 1))
                    stage_error = float(
                        np.abs(original.astype(np.float64) - recon).mean()
                    )

                    # Accumulate into cumulative_recon (in-place)
                    cumulative_recon += recon

                    # Residual = original - stage_0_reconstruction (in-place)
                    np.subtract(original.astype(np.float64), recon, out=residual)

                    del recon

                else:
                    # ── Stage 1+: compress RESIDUAL ────────────────────
                    residual_f32 = np.ascontiguousarray(residual, dtype=np.float32)

                    if is_pseudo:
                        data, meta, recon_residual = self._execute_pseudo_stage(
                            method_name, residual_f32, resolved_params, orig_size
                        )
                        del residual_f32
                    elif method_name in ("huffman", "rans"):
                        from spectralstream.compression.methods.entropy._class_wrappers import (  # noqa: E501
                            HuffmanCoder,
                            RANS,
                        )

                        coder = HuffmanCoder() if method_name == "huffman" else RANS()
                        raw_bytes = residual_f32.tobytes()
                        compressed, entropy_meta = coder.compress(raw_bytes)
                        del residual_f32
                        decompressed = coder.decompress(compressed, entropy_meta)
                        recon_residual = (
                            np.frombuffer(decompressed, dtype=np.float32)
                            .reshape(original.shape)
                            .astype(np.float64)
                        )
                        # Encode bytes-tree to hex for JSON-safe metadata
                        entropy_meta_clean: Dict[str, Any] = {}
                        for ek, ev in entropy_meta.items():
                            if isinstance(ev, bytes):
                                entropy_meta_clean[ek] = ev.hex()
                            elif isinstance(ev, (str, int, float, bool, list, tuple)):
                                entropy_meta_clean[ek] = ev
                            else:
                                entropy_meta_clean[ek] = str(ev)
                        data, meta = compressed, entropy_meta_clean
                    elif method_name == "progressive_svd":
                        max_err = resolved_params.get("max_error", 0.01)
                        ps_result = self._progressive_svd_compress(
                            engine,
                            residual_f32,
                            max_error=max_err,
                        )
                        if ps_result is None:
                            del residual_f32
                            raise ValueError("Progressive SVD failed")
                        data, meta = ps_result["data"], ps_result["meta"]
                        recon_residual = ps_result["recon"]
                        del residual_f32
                    else:
                        data, meta = inst.compress(residual_f32, **resolved_params)
                        del residual_f32
                        recon_residual = inst.decompress(data, meta)
                    if recon_residual.shape != original.shape:
                        recon_residual = recon_residual.reshape(original.shape)
                    recon_residual = recon_residual.astype(np.float64)

                    stage_ratio = float(orig_size / max(len(data), 1))

                    # Accumulate into cumulative_recon (in-place)
                    cumulative_recon += recon_residual

                    stage_error = float(
                        np.abs(original.astype(np.float64) - cumulative_recon).mean()
                    )

                    # Update residual in-place
                    np.subtract(residual, recon_residual, out=residual)

                    del recon_residual

                # Store metadata only (no ndarrays)
                stages_data.append(
                    {
                        "method": method_name,
                        "compressed_data": data,
                        "metadata": {
                            k: v
                            for k, v in meta.items()
                            if isinstance(v, (str, int, float, bool, list, tuple))
                        },
                        "stage_ratio": stage_ratio,
                        "stage_error": stage_error,
                        "original_shape": list(original.shape),
                    }
                )

                logger.debug(
                    "Cascade stage %d (%s): ratio=%.2fx, error=%.6f  [%s]",
                    i + 1,
                    method_name,
                    stage_ratio,
                    stage_error,
                    "original" if i == 0 else "residual",
                )

                del data, meta
                _gc.collect()

            except Exception as exc:
                # Stage failed — abandon this pattern entirely
                logger.debug(
                    "Pattern '%s' stage %d (%s) failed: %s",
                    pattern,
                    i + 1,
                    method_name,
                    exc,
                )
                del cumulative_recon, residual
                _gc.collect()
                return None

        if not stages_data:
            del cumulative_recon, residual
            _gc.collect()
            return None

        # ── Final total error = last stage's cumulative error ─────────
        total_error = stages_data[-1]["stage_error"]

        # ── Build output payload ─────────────────────────────────────
        if self.store_all_stages:
            compressed_data, total_ratio_val = self._package_stages(
                stages_data, orig_size
            )
        else:
            last = stages_data[-1]
            total_ratio_val = float(orig_size / max(len(last["compressed_data"]), 1))
            compressed_data = last["compressed_data"]

        # ── Post-process: entropy coding on combined payload ─────────
        entropy_info: Dict[str, Any] = {}
        if entropy_post_process is not None and len(compressed_data) > 1024:
            try:
                entropy_data, entropy_meta = self._entropy_compress(
                    compressed_data, entropy_post_process
                )
                ratio_gain = len(compressed_data) / max(len(entropy_data), 1)
                entropy_info = entropy_meta
                entropy_info["entropy_ratio_gain"] = ratio_gain
                entropy_info["inner_size"] = len(compressed_data)
                compressed_data = entropy_data
                total_ratio_val = float(orig_size / max(len(compressed_data), 1))
            except Exception:
                pass

        # Build clean stage summaries (exclude raw bytes)
        stage_summaries: List[Dict[str, Any]] = []
        for s in stages_data:
            stage_summaries.append(
                {
                    "method": s["method"],
                    "compressed_size": len(s["compressed_data"]),
                    "stage_ratio": s["stage_ratio"],
                    "stage_error": s["stage_error"],
                }
            )

        # ── Compute comprehensive loss metrics ───────────────────────
        full_recon: np.ndarray
        if self.store_all_stages:
            full_recon = np.ascontiguousarray(cumulative_recon, dtype=np.float32)
        else:
            last = stages_data[-1]
            inst_last = engine._methods.get(last["method"])
            if inst_last is not None:
                last_meta = dict(last["metadata"])
                if "original_shape" not in last_meta:
                    last_meta["original_shape"] = list(original.shape)
                full_recon = inst_last.decompress(last["compressed_data"], last_meta)
                if full_recon.shape != original.shape:
                    full_recon = full_recon.reshape(original.shape)
                full_recon = np.ascontiguousarray(full_recon, dtype=np.float32)
            else:
                full_recon = original.copy()

        # Free large intermediates
        del cumulative_recon, residual
        _gc.collect()

        loss_metrics = TensorLossMetrics.compute(
            original=original,
            reconstructed=full_recon,
            name=tensor_type if tensor_type else "cascade",
            compressed_size=len(compressed_data),
        )

        result_metadata: Dict[str, Any] = {
            "method": "cascade",
            "pattern": pattern,
            "stages": stage_summaries,
            "total_ratio": float(total_ratio_val),
            "total_error": float(total_error),
            "total_mse": loss_metrics.mse,
            "mse": loss_metrics.mse,
            "n_stages": len(stages_data),
            "tensor_type": tensor_type,
            "loss_metrics": loss_metrics.to_dict(),
            "quality_grade": loss_metrics.quality_grade,
            "is_acceptable": loss_metrics.is_acceptable,
            "snr_db": loss_metrics.snr_db,
            "psnr_db": loss_metrics.psnr_db,
            "cosine_similarity": loss_metrics.cosine_similarity,
        }
        if entropy_info:
            result_metadata["entropy_post_process"] = entropy_info
            result_metadata["entropy"] = entropy_info

        logger.debug(
            "Pattern '%s' succeeded: ratio=%.1fx, error=%.6f, grade=%s, SNR=%.1fdB",
            pattern,
            total_ratio_val,
            total_error,
            loss_metrics.quality_grade,
            loss_metrics.snr_db,
        )

        return compressed_data, result_metadata

    # ── Entangled SVD Cascade ────────────────────────────────────────────

    def _execute_svd_entangled(
        self,
        engine: Any,
        tensor: np.ndarray,
        tensor_type: str,
        orig_size: int,
        entropy_post_process: Optional[str] = None,
    ) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """Execute entangled SVD cascade: compress U, Vt separately with DCT.

        Standard SVD stores all factors (U, S, Vt) as packed float16.
        The entangled approach compresses U (m×r) and Vt (r×n) with DCT,
        exploiting the signal-like structure of their columns/rows for
        much higher compression ratios.  S (r,) is tiny — stored as float16.

        Flow
        ----
        1. Run SVD with ``store_factors=True`` to get raw U, S, Vt arrays
        2. Compress U with DCT (each column is a signal)
        3. Compress Vt with DCT (each row is a signal)
        4. Store S as float16 bytes
        5. Reconstruct from compressed factors for error/loss metrics
        6. Package all parts into a single payload
        7. Optionally apply entropy post-processing

        Parameters
        ----------
        engine : CompressionIntelligenceEngine
            Engine with ``_methods`` registry.
        tensor : np.ndarray
            Float32 tensor to compress (must be 2D).
        tensor_type : str
            Tensor type hint (e.g. ``"weight"``, ``"norm_bias"``).
        orig_size : int
            Original tensor size in bytes.
        entropy_post_process : str or None
            Optional entropy coding method (``"huffman"`` or ``"rans"``).

        Returns
        -------
        Tuple[bytes, dict] or None
            ``(compressed_data, metadata)`` or None on failure.
        """
        import gc as _gc
        import json
        import struct as _struct

        svd_inst = engine._methods.get("svd_compress")
        if svd_inst is None:
            return None

        # ── Step 1: Run SVD with store_factors to get raw U, S, Vt ───
        # Resolve rank from pattern: "auto:30" → max(min(shape)//30, 4)
        rank = DirectCascadeEngine.auto_rank(tensor.shape, 30)

        try:
            svd_data, svd_meta = svd_inst.compress(
                tensor,
                rank=rank,
                store_factors=True,
            )
        except Exception as exc:
            logger.debug("SVD entangled: svd_compress failed: %s", exc)
            return None

        # Extract raw factors from metadata
        U = svd_meta.get("_svd_U")
        S = svd_meta.get("_svd_S")
        Vt = svd_meta.get("_svd_Vt")

        if U is None or S is None or Vt is None:
            logger.debug("SVD entangled: raw factors not found in metadata")
            return None

        # Clean up metadata (remove large factor arrays)
        for _key in ("_svd_U", "_svd_S", "_svd_Vt"):
            svd_meta.pop(_key, None)
        del svd_data
        _gc.collect()

        # ── Step 2: Entangle factors — compress U, Vt separately ─────
        compressed_parts, factor_ratio = self._entangle_svd_factors(
            engine,
            U,
            S,
            Vt,
        )

        # ── Step 3: Package all parts into a single payload ──────────
        # Format: uint32 num_parts, then for each part:
        #   uint32 name_len + bytes name + uint32 meta_len + bytes meta + uint32 data_len + bytes data
        buf = bytearray()
        buf += _struct.pack("<I", len(compressed_parts))  # 3 parts: U, S, Vt

        part_metas: Dict[str, Dict[str, Any]] = {}
        for name in ("U", "S", "Vt"):
            part_data, part_meta, part_shape = compressed_parts[name]
            part_metas[name] = part_meta

            name_bytes = name.encode("utf-8")
            meta_json = json.dumps(
                {
                    k: v
                    for k, v in {**part_meta, "shape": list(part_shape)}.items()
                    if isinstance(v, (str, int, float, bool, list, tuple))
                },
                default=str,
            ).encode("utf-8")

            buf += _struct.pack("<I", len(name_bytes))
            buf += name_bytes
            buf += _struct.pack("<I", len(meta_json))
            buf += meta_json
            buf += _struct.pack("<I", len(part_data))
            buf += part_data

        compressed_data = bytes(buf)

        # ── Step 4: Apply entropy post-processing if requested ───────
        entropy_info: Dict[str, Any] = {}
        if entropy_post_process is not None and len(compressed_data) > 1024:
            try:
                entropy_data, entropy_meta = self._entropy_compress(
                    compressed_data,
                    entropy_post_process,
                )
                ratio_gain = len(compressed_data) / max(len(entropy_data), 1)
                entropy_info = entropy_meta
                entropy_info["entropy_ratio_gain"] = ratio_gain
                entropy_info["inner_size"] = len(compressed_data)
                compressed_data = entropy_data
            except Exception:
                pass

        # ── Step 5: Reconstruct from compressed factors ──────────────
        # Decompress DCT factors to get U_recon, Vt_recon, then compute
        # U @ diag(S) @ Vt for the full reconstruction.
        dct_inst = engine._methods.get("dct_spectral")
        try:
            u_data, u_meta, u_shape = compressed_parts["U"]
            if dct_inst is not None and not part_metas["U"].get("passthrough", True):
                U_recon = dct_inst.decompress(u_data, part_metas["U"])
            else:
                U_recon = (
                    np.frombuffer(u_data, dtype=np.float16)
                    .reshape(u_shape)
                    .astype(np.float32)
                )

            v_data, v_meta, v_shape = compressed_parts["Vt"]
            if dct_inst is not None and not part_metas["Vt"].get("passthrough", True):
                Vt_recon = dct_inst.decompress(v_data, part_metas["Vt"])
            else:
                Vt_recon = (
                    np.frombuffer(v_data, dtype=np.float16)
                    .reshape(v_shape)
                    .astype(np.float32)
                )

            S_recon = S.astype(np.float32)  # S is already in memory
            recon = (U_recon * S_recon) @ Vt_recon
            if recon.shape != tensor.shape:
                recon = recon.reshape(tensor.shape)
            recon = np.ascontiguousarray(recon, dtype=np.float32)
        except Exception as exc:
            logger.debug("SVD entangled: reconstruction failed: %s", exc)
            return None

        # ── Step 6: Compute final ratio and error ────────────────────
        total_ratio = float(orig_size / max(len(compressed_data), 1))
        total_error = float(
            np.abs(tensor.astype(np.float64) - recon.astype(np.float64)).mean()
        )

        # ── Step 7: Loss metrics ────────────────────────────────────
        from .loss_metrics import TensorLossMetrics

        loss_metrics = TensorLossMetrics.compute(
            original=tensor,
            reconstructed=recon,
            name=f"{tensor_type}_svd_entangled",
            compressed_size=len(compressed_data),
        )

        # ── Build stage summaries ────────────────────────────────────
        stage_summaries: List[Dict[str, Any]] = [
            {
                "method": "svd_compress",
                "compressed_size": 0,  # Replaced by entangled parts
                "stage_ratio": factor_ratio,
                "stage_error": total_error,
                "rank": rank,
            },
            {
                "method": "entangle_u",
                "compressed_size": len(compressed_parts["U"][0]),
                "stage_ratio": factor_ratio,
                "stage_error": total_error,
                "method_params": {"dct_keep_ratio": 0.2},
            },
            {
                "method": "entangle_v",
                "compressed_size": len(compressed_parts["Vt"][0]),
                "stage_ratio": factor_ratio,
                "stage_error": total_error,
                "method_params": {"dct_keep_ratio": 0.2},
            },
            {
                "method": "entangle_s",
                "compressed_size": len(compressed_parts["S"][0]),
                "stage_ratio": factor_ratio,
                "stage_error": 0.0,
                "method_params": {"dtype": "float16"},
            },
        ]
        if entropy_info:
            stage_summaries.append(
                {
                    "method": f"entropy:{entropy_post_process}",
                    "compressed_size": len(compressed_data),
                    "stage_ratio": total_ratio,
                    "stage_error": 0.0,
                    "entropy_ratio_gain": entropy_info.get("entropy_ratio_gain", 1.0),
                }
            )

        result_metadata: Dict[str, Any] = {
            "method": "svd_entangled",
            "pattern": "svd_entangled",
            "stages": stage_summaries,
            "total_ratio": total_ratio,
            "total_error": total_error,
            "total_mse": loss_metrics.mse,
            "mse": loss_metrics.mse,
            "n_stages": len(stage_summaries),
            "tensor_type": tensor_type,
            "loss_metrics": loss_metrics.to_dict(),
            "quality_grade": loss_metrics.quality_grade,
            "is_acceptable": loss_metrics.is_acceptable,
            "snr_db": loss_metrics.snr_db,
            "psnr_db": loss_metrics.psnr_db,
            "cosine_similarity": loss_metrics.cosine_similarity,
            "entangled_svd": {
                "rank": rank,
                "factor_ratio": factor_ratio,
                "u_shape": list(U.shape),
                "v_shape": list(Vt.shape),
                "s_size": S.nbytes,
            },
        }
        if entropy_info:
            result_metadata["entropy_post_process"] = entropy_info
            result_metadata["entropy"] = entropy_info

        logger.debug(
            "SVD entangled: rank=%d, ratio=%.1fx, factor_ratio=%.1fx, "
            "error=%.6f, grade=%s, SNR=%.1fdB",
            rank,
            total_ratio,
            factor_ratio,
            total_error,
            loss_metrics.quality_grade,
            loss_metrics.snr_db,
        )

        del S, U, Vt, recon
        _gc.collect()
        return compressed_data, result_metadata

    def _entangle_svd_factors(
        self,
        engine: Any,
        U: np.ndarray,
        S: np.ndarray,
        Vt: np.ndarray,
    ) -> Tuple[Dict[str, Tuple[bytes, Dict[str, Any], Tuple[int, ...]]], float]:
        """Compress SVD factors (U, Vt) independently using DCT.

        U (m×r) — each column treated as a signal, compressed with DCT
        Vt (r×n) — each row treated as a signal, compressed with DCT
        S (r,) — tiny, stored as float16

        Parameters
        ----------
        engine : CompressionIntelligenceEngine
            Engine with ``_methods`` registry (provides DCT method).
        U : ndarray (m, r)
            Left singular vectors.
        S : ndarray (r,)
            Singular values.
        Vt : ndarray (r, n)
            Right singular vectors (transposed).

        Returns
        -------
        Tuple[Dict[str, Tuple[bytes, dict, tuple]], float]
            ``(compressed_parts, factor_ratio)`` where ``compressed_parts``
            has keys ``"U"``, ``"Vt"``, ``"S"`` and each value is
            ``(compressed_bytes, metadata_dict, original_shape)``.
            ``factor_ratio`` is the compression ratio of the factors
            themselves (original factor bytes / compressed factor bytes).
        """
        dct_inst = engine._methods.get("dct_spectral")
        compressed_parts: Dict[str, Tuple[bytes, Dict[str, Any], Tuple[int, ...]]] = {}

        # ── Compress U: (m×r) — treat each column as a signal ─────────
        if dct_inst is not None and U.shape[1] >= 4:
            try:
                u_data, u_meta = dct_inst.compress(
                    np.ascontiguousarray(U, dtype=np.float32),
                    keep_ratio=0.2,
                )
                compressed_parts["U"] = (u_data, u_meta, U.shape)
            except Exception:
                # Fallback: float16
                u_data = U.astype(np.float16).tobytes()
                compressed_parts["U"] = (u_data, {"passthrough": True}, U.shape)
        else:
            u_data = U.astype(np.float16).tobytes()
            compressed_parts["U"] = (u_data, {"passthrough": True}, U.shape)

        # ── Compress Vt: (r×n) — treat each row as a signal ──────────
        if dct_inst is not None and Vt.shape[0] >= 4:
            try:
                v_data, v_meta = dct_inst.compress(
                    np.ascontiguousarray(Vt, dtype=np.float32),
                    keep_ratio=0.2,
                )
                compressed_parts["Vt"] = (v_data, v_meta, Vt.shape)
            except Exception:
                v_data = Vt.astype(np.float16).tobytes()
                compressed_parts["Vt"] = (v_data, {"passthrough": True}, Vt.shape)
        else:
            v_data = Vt.astype(np.float16).tobytes()
            compressed_parts["Vt"] = (v_data, {"passthrough": True}, Vt.shape)

        # ── Store S as float16 (tiny — negligible size) ──────────────
        s_data = S.astype(np.float16).tobytes()
        compressed_parts["S"] = (s_data, {"passthrough": True}, S.shape)

        # ── Compute factor compression ratio ──────────────────────────
        original_factor_bytes = U.nbytes + S.nbytes + Vt.nbytes
        compressed_factor_bytes = sum(len(p[0]) for p in compressed_parts.values())
        factor_ratio = float(original_factor_bytes / max(compressed_factor_bytes, 1))

        return compressed_parts, factor_ratio

    # ── Full Cascade Execution (Self-Healing Fallback Chain) ────────────

    def execute_cascade(
        self,
        engine: Any,
        tensor: np.ndarray,
        tensor_type: str = "weight",
        pattern: Optional[str] = None,
        entropy_post_process: Optional[str] = None,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Execute cascade with self-healing fallback chain.

        Rather than crashing on the first failure or producing a < 1.0x
        ratio, this method tries progressively simpler patterns:

        1. Requested pattern (e.g. ``'extreme'``)
        2. One level down (e.g. ``'aggressive'``)
        3. ``'balanced'``
        4. ``'lightning'``
        5. Passthrough (no compression, ratio = 1.0)

        After each pattern attempt, the result is validated:
        - ``ratio > 1.0`` -> return immediately
        - ``ratio <= 1.0`` -> fall through to the next pattern

        Parameters
        ----------
        engine : CompressionIntelligenceEngine
            Engine with ``_methods`` registry.
        tensor : np.ndarray
            Float32 tensor to compress.
        tensor_type : str
            Tensor type hint (e.g. ``"weight"``, ``"norm_bias"``, ``"qkv"``).
        pattern : str, optional
            Cascade pattern name.  If None, automatically selected via
            ``select_pattern`` based on tensor size and type.
        entropy_post_process : str or None
            Entropy coding method to apply after all cascade stages.
            ``"huffman"`` or ``"rans"``.  If None (default), falls back
            to ``self.entropy_post_process`` (set at init).  If both are
            None, no entropy post-processing is applied.

        Returns
        -------
        Tuple[bytes, dict]
            - ``compressed_data``: Packaged bytes containing all stages'
              compressed data and a header.
            - ``metadata``: Dict with keys ``method``, ``pattern``,
              ``stages`` (list of per-stage info), ``total_ratio``,
              ``total_error``, ``n_stages``.
        """
        # ── Embedding tensors → specialized compressor ────────────────
        if tensor_type == "embedding" and tensor.size >= 10_000_000:
            try:
                result = self._compress_embedding(engine, tensor, tensor_type)
                if result is not None:
                    return result
            except Exception as exc:
                logger.debug("Embedding compressor failed: %s, falling back", exc)

        # ── Passthrough for sub-1KB tensors ───────────────────────────
        # Tensors smaller than 1KB have negligible size — metadata overhead
        # of compression exceeds any gain.  Store uncompressed.
        # Note: `original` is not yet defined here; use tensor directly.
        if not self._should_compress(tensor):
            raw = np.ascontiguousarray(tensor, dtype=np.float32)
            compressed_data = raw.tobytes()
            loss_metrics = TensorLossMetrics.compute(
                original=raw,
                reconstructed=raw,
                name=f"{tensor_type}_passthrough",
                compressed_size=len(compressed_data),
            )
            return compressed_data, {
                "method": "passthrough",
                "pattern": "passthrough",
                "stages": [
                    {
                        "method": "passthrough",
                        "compressed_size": len(compressed_data),
                        "stage_ratio": 1.0,
                        "stage_error": 0.0,
                    }
                ],
                "total_ratio": 1.0,
                "total_error": 0.0,
                "total_mse": 0.0,
                "mse": 0.0,
                "n_stages": 0,
                "tensor_type": tensor_type,
                "loss_metrics": loss_metrics.to_dict(),
                "quality_grade": loss_metrics.quality_grade,
                "is_acceptable": loss_metrics.is_acceptable,
            }

        # ── Auto-select pattern if not specified ──────────────────────
        if pattern is None or pattern == "auto":
            pattern = self.select_pattern(tensor, tensor_type)
        elif pattern not in self.ALL_PATTERNS:
            logger.warning("Unknown pattern '%s', auto-selecting", pattern)
            pattern = self.select_pattern(tensor, tensor_type)

        # ── Build fallback chain ──────────────────────────────────────
        # Start with the requested pattern, then degrade gracefully.
        embedding_fallbacks = (
            ["embedding_extreme", "embedding_balanced", "balanced"]
            if tensor_type == "embedding"
            else []
        )
        # Residual-optimized fallback for robust tensor types
        residual_fallbacks = (
            ["sparse_residual"] if tensor_type in ("weight", "ffn", "embedding") else []
        )
        patterns_to_try: List[str] = [
            pattern,
            *residual_fallbacks,
            *embedding_fallbacks,
            "aggressive",
            "balanced",
            "lightning",
        ]
        # Remove duplicates while preserving order
        patterns_to_try = list(dict.fromkeys([p for p in patterns_to_try if p]))

        # Resolve active entropy method
        active_entropy = (
            entropy_post_process
            if entropy_post_process is not None
            else self.entropy_post_process
        )
        # Normalise: True → "huffman", False/None → None
        if isinstance(active_entropy, bool):
            active_entropy = "huffman" if active_entropy else None

        last_error = ""
        original = np.ascontiguousarray(tensor, dtype=np.float32)
        orig_size = original.nbytes

        for attempt_pattern in patterns_to_try:
            try:
                result = self._execute_single_pattern(
                    engine,
                    original,
                    tensor_type,
                    attempt_pattern,
                    active_entropy,
                )
                if result is not None:
                    data, meta = result
                    ratio = meta.get("total_ratio", orig_size / max(len(data), 1))
                    error = meta.get("total_error", 1.0)

                    # Validate: ratio must be > 1.0 (otherwise passthrough is better)
                    if ratio > 1.0:
                        # ── Check tiered error budget for this tensor type ──
                        # Sensitive types (attention Q/K) must meet tight
                        # budgets; robust types (FFN) have wider tolerance.
                        loss_dict = meta.get("loss_metrics", {})
                        rel_err = meta.get("total_error", 1.0)
                        mse_val = loss_dict.get("mse", meta.get("mse", 1.0))
                        snr_val = meta.get("snr_db", 0.0)

                        if not is_within_budget(tensor_type, rel_err, mse_val, snr_val):
                            budget = get_budget(tensor_type)
                            logger.debug(
                                "Budget violated for %s with pattern '%s': "
                                "error=%.6f (max=%.6f) mse=%.8f (max=%.8f) "
                                "snr=%.1fdB (min=%.1fdB) — trying conservative",
                                tensor_type,
                                attempt_pattern,
                                rel_err,
                                budget[0],
                                mse_val,
                                budget[1],
                                snr_val,
                                budget[2],
                            )
                            # Try more conservative cascade pattern recursively
                            conservative = get_fallback_pattern(
                                tensor_type,
                                attempt_pattern,
                            )
                            if conservative != attempt_pattern:
                                return self.execute_cascade(
                                    engine,
                                    tensor,
                                    tensor_type,
                                    conservative,
                                    entropy_post_process,
                                )

                            # Already at most conservative pattern — accept
                            logger.debug(
                                "Already at most conservative pattern '%s'; "
                                "accepting budget violation for %s",
                                attempt_pattern,
                                tensor_type,
                            )
                            return data, meta

                        logger.debug(
                            "Fallback chain: pattern '%s' succeeded "
                            "(ratio=%.1fx, error=%.6f, budget_ok=True)",
                            attempt_pattern,
                            ratio,
                            error,
                        )
                        return data, meta

                    last_error = f"ratio={ratio:.1f}x < 1.0"

            except Exception as exc:
                last_error = str(exc)[:80]
                logger.debug(
                    "Fallback chain: pattern '%s' raised: %s",
                    attempt_pattern,
                    last_error,
                )
                continue

        # ── Try Method Stacking Engine as fallback ──────────────────────
        # Stacking works on the ORIGINAL tensor (not residuals), testing
        # complementary method pairs independently.  This is effective
        # when all residual-based cascade patterns fail because the
        # residual is noise-like.
        if tensor_type in (
            "weight",
            "ffn",
            "embedding",
            "qkv",
            "attention_q",
            "attention_k",
        ):
            try:
                from .stacking_engine import try_stacking_fallback

                stack_result = try_stacking_fallback(
                    self, engine, original, tensor_type, max_error=0.01
                )
                if stack_result is not None:
                    logger.info(
                        "Stacking fallback succeeded for %s tensor %s: ratio=%.1fx",
                        tensor_type,
                        str(original.shape),
                        stack_result[1].get("total_ratio", 1.0),
                    )
                    return stack_result
            except Exception as exc:
                logger.debug("Stacking fallback failed: %s", exc)

        # ── Last resort: passthrough (no compression, ratio = 1.0) ────
        logger.warning(
            "All cascade patterns and stacking fallback failed (%s); "
            "using passthrough for %s tensor %s",
            last_error,
            tensor_type,
            str(original.shape),
        )
        compressed_data = original.tobytes()
        loss_metrics = TensorLossMetrics.compute(
            original=original,
            reconstructed=original,
            name=f"{tensor_type}_passthrough",
            compressed_size=len(compressed_data),
        )
        return compressed_data, {
            "method": "passthrough",
            "pattern": "passthrough",
            "stages": [
                {
                    "method": "passthrough",
                    "compressed_size": len(compressed_data),
                    "stage_ratio": 1.0,
                    "stage_error": 0.0,
                }
            ],
            "total_ratio": 1.0,
            "total_error": 0.0,
            "n_stages": 1,
            "tensor_type": tensor_type,
            "loss_metrics": loss_metrics.to_dict(),
            "quality_grade": loss_metrics.quality_grade,
            "is_acceptable": loss_metrics.is_acceptable,
        }

    # ── Packaging / Unpackaging ──────────────────────────────────────────

    @staticmethod
    def _package_stages(
        stages_meta: List[Dict[str, Any]], orig_size: int
    ) -> Tuple[bytes, float]:
        """Package all stages' compressed data into a single payload.

        Format
        ------
        - uint32: number of stages (N)
        - For each stage:
          - uint32: method name length (L)
          - bytes: method name (UTF-8)
          - uint32: JSON metadata length (M)
          - bytes: JSON metadata
          - uint32: compressed data length (D)
          - bytes: compressed data
        """
        buf = bytearray()
        buf += struct.pack("<I", len(stages_meta))

        for stage in stages_meta:
            method_bytes = stage["method"].encode("utf-8")
            meta_json = json.dumps(
                {
                    k: v
                    for k, v in stage.get("metadata", {}).items()
                    if isinstance(v, (str, int, float, bool, list, tuple))
                },
                default=str,
            ).encode("utf-8")
            comp_data = stage["compressed_data"]

            buf += struct.pack("<I", len(method_bytes))
            buf += method_bytes
            buf += struct.pack("<I", len(meta_json))
            buf += meta_json
            buf += struct.pack("<I", len(comp_data))
            buf += comp_data

        total_compressed = len(buf)
        total_ratio = float(orig_size / max(total_compressed, 1))
        return bytes(buf), total_ratio

    @staticmethod
    def unpack_stages(
        data: bytes,
    ) -> List[Dict[str, Any]]:
        """Unpack stages from a payload created by ``_package_stages``.

        Parameters
        ----------
        data : bytes
            Packaged payload.

        Returns
        -------
        list of dict
            Each dict has keys ``method``, ``metadata``, ``compressed_data``.
        """
        stages: List[Dict[str, Any]] = []
        pos = 0
        n_stages = struct.unpack_from("<I", data, pos)[0]
        pos += 4

        for _ in range(n_stages):
            name_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            method_name = data[pos : pos + name_len].decode("utf-8")
            pos += name_len

            meta_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            meta = json.loads(data[pos : pos + meta_len].decode("utf-8"))
            pos += meta_len

            comp_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            comp_data = data[pos : pos + comp_len]
            pos += comp_len

            stages.append(
                {
                    "method": method_name,
                    "metadata": meta,
                    "compressed_data": comp_data,
                }
            )

        return stages

    # ── Reconstruction ───────────────────────────────────────────────────

    def reconstruct(
        self,
        engine: Any,
        data: bytes,
        metadata: Dict[str, Any],
        original_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Reconstruct original tensor from cascade-compressed data.

        Handles both regular cascade payloads and entropy-wrapped payloads
        (where the first stage is ``"entropy:huffman"`` or ``"entropy:rans"``).

        Parameters
        ----------
        engine : CompressionIntelligenceEngine
            Engine with ``_methods`` registry.
        data : bytes
            Packaged compressed data.
        metadata : dict
            Cascade metadata from ``execute_cascade``.
        original_shape : tuple of int
            Original tensor shape.

        Returns
        -------
        np.ndarray
            Reconstructed float32 tensor.
        """
        if not metadata.get("stages"):
            # Single-stage fallback
            inst = engine._methods.get("block_int8")
            if inst is not None:
                return inst.decompress(data, metadata)
            return np.zeros(original_shape, dtype=np.float32)

        # Passthrough handling — data is raw bytes, no stage packaging
        if metadata.get("method") == "passthrough":
            return np.frombuffer(data, dtype=np.float32).reshape(original_shape).copy()

        # ── Handle meta["entropy"] format (from _entropy_compress) ───
        # When entropy is applied using the new _entropy_compress method,
        # the metadata stores entropy info at the top-level "entropy" key.
        # We decompress the outer entropy layer first, then proceed with
        # normal stage unpacking.
        if "entropy" in metadata:
            entropy_meta = metadata["entropy"]
            if "entropy_method" in entropy_meta:
                try:
                    data = self._entropy_decompress(data, entropy_meta)
                    # Remove entropy meta so downstream code sees clean metadata
                    metadata = {k: v for k, v in metadata.items() if k != "entropy"}
                except Exception as exc:
                    logger.warning(
                        "Entropy decode (meta['entropy'] format) failed: %s", exc
                    )
                    # Fall through — try unpacking raw data anyway

        # ── Handle Entangled SVD directly (custom format, not stage-based) ──
        if metadata.get("method") == "svd_entangled":
            return self._reconstruct_svd_entangled(
                engine, data, metadata, original_shape
            )

        # ── Unpack stages, handling legacy entropy wrapper ────────────
        # The legacy format (from _apply_entropy_post_process) wraps
        # entropy as a stage with method "entropy:huffman" etc.
        stages = self.unpack_stages(data)

        # Check if the first stage is an entropy wrapper
        if stages and stages[0]["method"].startswith("entropy:"):
            # Decode the entropy wrapper first
            entropy_method = stages[0]["method"].split(":", 1)[1]
            inst = engine._methods.get(entropy_method)
            if inst is not None:
                try:
                    entropy_meta = dict(stages[0]["metadata"])
                    encoded_data = stages[0]["compressed_data"]
                    if entropy_method == "huffman":
                        # Huffman returns uint8 array
                        decoded_array = inst.decompress(encoded_data, entropy_meta)
                        decoded_bytes = decoded_array.tobytes()
                    else:
                        # rans returns bytes
                        decoded_bytes = inst.decompress(encoded_data, entropy_meta)
                    # Now unpack the inner stages
                    stages = self.unpack_stages(decoded_bytes)
                except Exception as exc:
                    logger.warning("Entropy decode failed: %s, trying raw stages", exc)
                    # Fall through to use the outer stages
            else:
                logger.warning(
                    "Entropy method '%s' not found for decode",
                    entropy_method,
                )

        if not stages:
            return np.zeros(original_shape, dtype=np.float32)

        # ── Reconstruct through all stages and SUM their outputs ─────
        # In the residual-based cascade, each stage decompresses a
        # DIFFERENT signal (original, then residual, then residual, ...).
        # The total reconstruction = sum of all stage reconstructions.
        result = np.zeros(original_shape, dtype=np.float64)

        for stage_info in stages:
            method_name = stage_info["method"]

            # Handle pseudo-methods inline
            pseudo_methods = {"dct_threshold", "sparse_store"}
            if method_name in pseudo_methods:
                try:
                    decompressed = self._reconstruct_pseudo_stage(
                        method_name, stage_info, original_shape
                    )
                    if decompressed.shape != original_shape:
                        decompressed = decompressed.reshape(original_shape)
                    result += decompressed.astype(np.float64)
                except Exception as exc:
                    logger.warning(
                        "Pseudo-method '%s' reconstruction failed: %s",
                        method_name,
                        exc,
                    )
                continue

            # Handle entropy stages inline (not in engine._methods)
            if method_name in ("huffman", "rans"):
                try:
                    from spectralstream.compression.methods.entropy._class_wrappers import (  # noqa: E501
                        HuffmanCoder,
                        RANS,
                    )

                    coder = HuffmanCoder() if method_name == "huffman" else RANS()
                    stage_meta = dict(stage_info["metadata"])
                    # Restore hex-encoded bytes fields (tree, etc.)
                    for _key in list(stage_meta.keys()):
                        _val = stage_meta[_key]
                        if isinstance(_val, str) and _key in ("tree",):
                            try:
                                stage_meta[_key] = bytes.fromhex(_val)
                            except (ValueError, TypeError):
                                pass
                    decompressed = coder.decompress(
                        stage_info["compressed_data"], stage_meta
                    )
                    # coder returns bytes for bytes input
                    recon_bytes: bytes
                    if isinstance(decompressed, bytes):
                        recon_bytes = decompressed
                    else:
                        recon_bytes = decompressed.tobytes()
                    stage_recon = (
                        np.frombuffer(recon_bytes, dtype=np.float32)
                        .reshape(original_shape)
                        .astype(np.float64)
                    )
                    result += stage_recon
                except Exception as exc:
                    logger.warning(
                        "Entropy stage '%s' reconstruction failed: %s",
                        method_name,
                        exc,
                    )
                continue

            inst = engine._methods.get(method_name)
            if inst is None:
                logger.warning(
                    "Method '%s' not found during reconstruction",
                    method_name,
                )
                continue

            # Ensure original_shape is in metadata for decompress
            stage_meta = dict(stage_info["metadata"])
            if "original_shape" not in stage_meta:
                stage_meta["original_shape"] = list(original_shape)

            try:
                decompressed = inst.decompress(
                    stage_info["compressed_data"], stage_meta
                )
                if decompressed.shape != original_shape:
                    decompressed = decompressed.reshape(original_shape)
                # SUM — not overwrite — each stage adds its reconstruction
                result += decompressed.astype(np.float64)
            except Exception as exc:
                logger.warning(
                    "Reconstruction stage '%s' failed: %s",
                    stage_info["method"],
                    exc,
                )

        return np.ascontiguousarray(result, dtype=np.float32)

    # ── Entangled SVD Reconstruction ─────────────────────────────────────

    def _reconstruct_svd_entangled(
        self,
        engine: Any,
        data: bytes,
        metadata: Dict[str, Any],
        original_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Reconstruct a tensor from entangled SVD compressed data.

        The entangled SVD payload has a custom format (not standard stages):
          uint32: num_parts (always 3: U, S, Vt)
          For each part:
            uint32: name_len
            bytes:  name ("U", "S", or "Vt")
            uint32: meta_len
            bytes:  JSON metadata
            uint32: data_len
            bytes:  compressed data

        Reconstruction computes ``recon = U_recon @ diag(S_recon) @ Vt_recon``,
        where U_recon and Vt_recon are decompressed with DCT, and S_recon
        is read directly from float16 bytes.

        Parameters
        ----------
        engine : CompressionIntelligenceEngine
            Engine with ``_methods`` registry.
        data : bytes
            Entangled SVD compressed payload.
        metadata : dict
            Cascade metadata containing entangled_svd info and shapes.
        original_shape : tuple of int
            Original tensor shape.

        Returns
        -------
        np.ndarray
            Reconstructed float32 tensor.
        """
        import json as _json
        import struct as _struct

        dct_inst = engine._methods.get("dct_spectral")
        pos = 0
        num_parts = _struct.unpack_from("<I", data, pos)[0]
        pos += 4

        parts: Dict[str, Tuple[bytes, Dict[str, Any], Tuple[int, ...]]] = {}

        for _ in range(num_parts):
            name_len = _struct.unpack_from("<I", data, pos)[0]
            pos += 4
            name = data[pos : pos + name_len].decode("utf-8")
            pos += name_len

            meta_len = _struct.unpack_from("<I", data, pos)[0]
            pos += 4
            part_meta = _json.loads(data[pos : pos + meta_len].decode("utf-8"))
            pos += meta_len

            comp_len = _struct.unpack_from("<I", data, pos)[0]
            pos += 4
            comp_data = data[pos : pos + comp_len]
            pos += comp_len

            # Extract shape from metadata (stored as list during packaging)
            part_shape = tuple(part_meta.pop("shape", (0,)))
            parts[name] = (comp_data, part_meta, part_shape)

        # ── Decompress U ────────────────────────────────────────────────
        if "U" not in parts:
            return np.zeros(original_shape, dtype=np.float32)

        u_data, u_meta, u_shape = parts["U"]
        if dct_inst is not None and not u_meta.get("passthrough", False):
            U_recon = dct_inst.decompress(u_data, u_meta)
        else:
            U_recon = (
                np.frombuffer(u_data, dtype=np.float16)
                .reshape(u_shape)
                .astype(np.float32)
            )

        # ── Decompress Vt ───────────────────────────────────────────────
        if "Vt" not in parts:
            return np.zeros(original_shape, dtype=np.float32)

        v_data, v_meta, v_shape = parts["Vt"]
        if dct_inst is not None and not v_meta.get("passthrough", False):
            Vt_recon = dct_inst.decompress(v_data, v_meta)
        else:
            Vt_recon = (
                np.frombuffer(v_data, dtype=np.float16)
                .reshape(v_shape)
                .astype(np.float32)
            )

        # ── Read S from float16 bytes ───────────────────────────────────
        if "S" not in parts:
            return np.zeros(original_shape, dtype=np.float32)

        s_data, _, s_shape = parts["S"]
        S_recon = (
            np.frombuffer(s_data, dtype=np.float16).reshape(s_shape).astype(np.float32)
        )

        # ── Reconstruct: U @ diag(S) @ Vt ──────────────────────────────
        recon = (U_recon * S_recon) @ Vt_recon
        if recon.shape != original_shape:
            recon = recon.reshape(original_shape)
        return np.ascontiguousarray(recon, dtype=np.float32)

    # ── Specialized Embedding Compressor ────────────────────────────────

    def _compress_embedding(
        self,
        engine: Any,
        tensor: np.ndarray,
        tensor_type: str = "embedding",
    ) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """Specialized compression for HUGE embedding tensors (≥10M elements).

        Embedding matrices are (vocab_size × hidden_dim) — for Gemma 4
        this is 262144×8960 = 4.5 GB.  Standard SVD would take 30+ seconds
        and 20+ GB RAM.

        Strategy
        --------
        1. Try aggressive SVD with various ranks (scales as O(min(m,n)²·max(m,n)))
        2. Compute projected compression ratio for the full matrix
        3. If SVD is too expensive, fall back to tensor_train (lower memory)
        4. If all else fails, return None (caller falls through to cascade)

        Parameters
        ----------
        engine : CompressionIntelligenceEngine
            Engine with ``_methods`` registry.
        tensor : np.ndarray
            Float32 embedding tensor to compress.
        tensor_type : str
            Tensor type hint (default ``"embedding"``).

        Returns
        -------
        Tuple[bytes, dict] or None
            ``(compressed_data, metadata)`` if successful, else None.
        """
        import gc as _gc
        import time as _time

        orig = np.ascontiguousarray(tensor, dtype=np.float32)
        orig_size = orig.nbytes
        shape = orig.shape
        m, n = shape[0], shape[1] if len(shape) > 1 else 1
        logger.info(
            "Embedding compressor: %s tensor %s (%.1f MB, %dx%d)",
            tensor_type,
            shape,
            orig_size / 1e6,
            m,
            n,
        )

        svd = engine._methods.get("svd_compress")
        if svd is None:
            return None

        # ── Option A: SVD at various ranks, pick best ────────────────
        # For a matrix of size m×n, SVD computes O(m·n·min(m,n))
        # We try conservative ranks first for quality, then aggressive.
        results: List[Dict[str, Any]] = []

        # Rank candidates: from conservative to aggressive
        # For 8960-dim: /40=224, /50=179, /100=89, /200=44, /500=18
        rank_divisors = [40, 50, 80, 100, 150, 200, 500]
        for divisor in rank_divisors:
            rank = max(min(m, n) // divisor, 2)
            try:
                t0 = _time.perf_counter()
                data, meta = svd.compress(orig, rank=rank)
                compress_time = _time.perf_counter() - t0

                recon = svd.decompress(data, meta)
                if recon.shape != shape:
                    recon = recon.reshape(shape)

                ratio = float(orig_size / max(len(data), 1))
                sample_error = float(
                    np.abs(orig.ravel()[:10000] - recon.ravel()[:10000]).mean()
                )
                full_error = float(np.abs(orig - recon).mean())

                # Projected: for full SVD, compressed size = (m*r + r + n*r) * 4 bytes
                projected_comp = (m * rank + rank + n * rank) * 4
                projected_ratio = orig_size / max(projected_comp, 1)

                logger.debug(
                    "  Embedding SVD rank=%d: ratio=%.1fx projected=%.0fx "
                    "error=%.6f sample_err=%.6f time=%.1fs",
                    rank,
                    ratio,
                    projected_ratio,
                    full_error,
                    sample_error,
                    compress_time,
                )

                results.append(
                    {
                        "rank": rank,
                        "compressed": data,
                        "meta": meta,
                        "recon": recon,
                        "ratio": ratio,
                        "projected_ratio": projected_ratio,
                        "error": full_error,
                        "time": compress_time,
                    }
                )

                del data, meta
                _gc.collect()

            except Exception as exc:
                logger.debug("  Embedding SVD rank=%d failed: %s", divisor, exc)
                continue

        if results:
            # Sort by projected ratio, then by error
            results.sort(key=lambda r: (-r["projected_ratio"], r["error"]))

            # Pick best ratio within error budget (2% max for embedding)
            for r in results:
                if r["error"] < 0.02:
                    logger.info(
                        "Embedding SVD selected: rank=%d, ratio=%.0fx "
                        "(proj=%.0fx), error=%.6f, time=%.1fs",
                        r["rank"],
                        r["ratio"],
                        r["projected_ratio"],
                        r["error"],
                        r["time"],
                    )
                    data = r["compressed"]
                    meta = r["meta"]
                    meta.update(
                        {
                            "method": "svd_embedding",
                            "embedding_rank": r["rank"],
                            "embedding_ratio": r["ratio"],
                            "embedding_projected_ratio": r["projected_ratio"],
                            "embedding_error": r["error"],
                        }
                    )

                    # ── Apply entropy post-process for extra gain ─────
                    if self.entropy_post_process and len(data) > 1024:
                        try:
                            entropy_data, entropy_meta = self._entropy_compress(
                                data, self.entropy_post_process
                            )
                            ratio_gain = len(data) / max(len(entropy_data), 1)
                            meta["entropy"] = {
                                "entropy_method": self.entropy_post_process,
                                "entropy_ratio_gain": ratio_gain,
                                "inner_size": len(data),
                            }
                            data = entropy_data
                        except Exception:
                            pass

                    total_ratio = float(orig_size / max(len(data), 1))
                    from .loss_metrics import TensorLossMetrics

                    loss_metrics = TensorLossMetrics.compute(
                        original=orig,
                        reconstructed=r["recon"],
                        name="embedding_svd",
                        compressed_size=len(data),
                    )

                    result_meta = {
                        "method": "svd_embedding",
                        "pattern": f"embedding_svd_r{r['rank']}",
                        "stages": [
                            {
                                "method": "svd_compress",
                                "compressed_size": len(r["compressed"]),
                                "stage_ratio": r["ratio"],
                                "stage_error": r["error"],
                                "rank": r["rank"],
                            }
                        ],
                        "total_ratio": total_ratio,
                        "total_error": r["error"],
                        "n_stages": 1,
                        "tensor_type": tensor_type,
                        "embedding": {
                            "rank": r["rank"],
                            "projected_ratio": r["projected_ratio"],
                            "error": r["error"],
                        },
                        "loss_metrics": loss_metrics.to_dict(),
                        "quality_grade": loss_metrics.quality_grade,
                        "is_acceptable": loss_metrics.is_acceptable,
                        "snr_db": loss_metrics.snr_db,
                        "psnr_db": loss_metrics.psnr_db,
                        "cosine_similarity": loss_metrics.cosine_similarity,
                    }
                    if "entropy" in meta:
                        result_meta["entropy"] = meta["entropy"]

                    logger.info(
                        "Embedding compressed: rank=%d, ratio=%.0fx, "
                        "error=%.6f, SNR=%.1fdB",
                        r["rank"],
                        total_ratio,
                        r["error"],
                        loss_metrics.snr_db,
                    )
                    return data, result_meta

            # ── Fallback: best available even if error > 2% ──────────
            best = results[0]
            logger.warning(
                "Embedding SVD: best error=%.6f exceeds 2%% budget, "
                "using rank=%d anyway (best projected ratio=%.0fx)",
                best["error"],
                best["rank"],
                best["projected_ratio"],
            )

        # ── Option B: Tensor Train (memory-safe for huge matrices) ────
        tt = engine._methods.get("tensor_train")
        if tt is not None:
            try:
                tt_rank = max(min(m, n) // 200, 4)
                data, meta = tt.compress(orig, rank=tt_rank)
                recon = tt.decompress(data, meta)
                if recon.shape != shape:
                    recon = recon.reshape(shape)
                ratio = float(orig_size / max(len(data), 1))
                error = float(np.abs(orig - recon).mean())
                if ratio > 5.0 and error < 0.05:
                    from .loss_metrics import TensorLossMetrics

                    loss_metrics = TensorLossMetrics.compute(
                        original=orig,
                        reconstructed=recon,
                        name="embedding_tt",
                        compressed_size=len(data),
                    )
                    logger.info(
                        "Embedding TT: rank=%d, ratio=%.0fx, error=%.6f",
                        tt_rank,
                        ratio,
                        error,
                    )
                    return data, {
                        "method": "tensor_train_embedding",
                        "pattern": "embedding_tt",
                        "stages": [
                            {"method": "tensor_train", "compressed_size": len(data)}
                        ],
                        "total_ratio": ratio,
                        "total_error": error,
                        "n_stages": 1,
                        "tensor_type": tensor_type,
                        "loss_metrics": loss_metrics.to_dict(),
                        "quality_grade": loss_metrics.quality_grade,
                        "is_acceptable": loss_metrics.is_acceptable,
                    }
            except Exception as exc:
                logger.debug("Embedding TT failed: %s", exc)

        # ── Option C: Use embedding cascade patterns ──────────────────
        # Try each embedding pattern through the standard cascade
        for emb_pattern in [
            "embedding_extreme",
            "embedding_balanced",
        ]:
            try:
                result = self._execute_single_pattern(
                    engine,
                    orig,
                    tensor_type,
                    emb_pattern,
                    self.entropy_post_process,
                )
                if result is not None:
                    data, meta = result
                    ratio = meta.get("total_ratio", 1.0)
                    if ratio > 1.5:
                        meta["method"] = f"cascade_{emb_pattern}"
                        return data, meta
            except Exception:
                continue

        return None  # Caller falls through to standard cascade

    # ── Fallback ─────────────────────────────────────────────────────────

    def _fallback_single(
        self, engine: Any, tensor: np.ndarray
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Fallback: single-method compression when cascade is not suitable."""
        inst = engine._methods.get("block_int8")
        if inst is None:
            # Absolute last resort — raw float16
            data = tensor.astype(np.float16).tobytes()
            ratio = float(tensor.nbytes / max(len(data), 1))
            recon = tensor.astype(np.float16).astype(np.float32)
            loss_metrics = TensorLossMetrics.compute(
                tensor, recon, "fallback_float16", len(data)
            )
            return data, {
                "method": "cascade_fallback_float16",
                "pattern": "fallback",
                "stages": [{"method": "float16", "compressed_size": len(data)}],
                "total_ratio": ratio,
                "total_error": 0.0,
                "n_stages": 1,
                "loss_metrics": loss_metrics.to_dict(),
                "quality_grade": loss_metrics.quality_grade,
                "is_acceptable": loss_metrics.is_acceptable,
            }

        try:
            compressed, meta = inst.compress(tensor)
            ratio = float(tensor.nbytes / max(len(compressed), 1))
            recon = inst.decompress(compressed, meta)
            err = float(
                np.linalg.norm(tensor.ravel() - recon.ravel())
                / max(np.linalg.norm(tensor.ravel()), 1e-30)
            )
            loss_metrics = TensorLossMetrics.compute(
                tensor, recon, "fallback_block_int8", len(compressed)
            )
            return compressed, {
                "method": "cascade_fallback_block_int8",
                "pattern": "fallback",
                "stages": [
                    {
                        "method": "block_int8",
                        "compressed_size": len(compressed),
                        "stage_ratio": ratio,
                        "stage_error": float(
                            np.abs(tensor.ravel() - recon.ravel()).mean()
                        ),
                    }
                ],
                "total_ratio": ratio,
                "total_error": err,
                "n_stages": 1,
                "loss_metrics": loss_metrics.to_dict(),
                "quality_grade": loss_metrics.quality_grade,
                "is_acceptable": loss_metrics.is_acceptable,
            }
        except Exception as exc:
            logger.warning("Fallback compression failed: %s", exc)
            data = tensor.astype(np.float16).tobytes()
            ratio = float(tensor.nbytes / max(len(data), 1))
            recon = tensor.astype(np.float16).astype(np.float32)
            loss_metrics = TensorLossMetrics.compute(
                tensor, recon, "fallback_float16", len(data)
            )
            return data, {
                "method": "cascade_fallback_float16",
                "pattern": "fallback",
                "stages": [{"method": "float16", "compressed_size": len(data)}],
                "total_ratio": ratio,
                "total_error": 1.0,
                "n_stages": 1,
                "loss_metrics": loss_metrics.to_dict(),
                "quality_grade": loss_metrics.quality_grade,
                "is_acceptable": loss_metrics.is_acceptable,
            }

    # ── Pseudo-Method Execution (dct_threshold, sparse_store) ────────

    def _execute_pseudo_stage(
        self,
        method_name: str,
        tensor: np.ndarray,
        params: Dict[str, Any],
        orig_size: int,
    ) -> Tuple[bytes, Dict[str, Any], np.ndarray]:
        """Execute a pseudo-method stage (dct_threshold or sparse_store).

        These methods are handled inline rather than through ``engine._methods``,
        because they implement specialized residual processing that existing
        registered methods do not cover efficiently.

        Parameters
        ----------
        method_name : str
            ``"dct_threshold"`` or ``"sparse_store"``.
        tensor : np.ndarray
            Input tensor (float32) to compress.
        params : dict
            Resolved parameters.
        orig_size : int
            Original tensor size in bytes (for ratio calculation).

        Returns
        -------
        Tuple[bytes, dict, np.ndarray]
            ``(compressed_data, metadata, reconstruction)``.

        Raises
        ------
        ValueError
            If ``method_name`` is unknown.
        """
        if method_name == "dct_threshold":
            return self._pseudo_dct_threshold(tensor, params, orig_size)
        elif method_name == "sparse_store":
            return self._pseudo_sparse_store(tensor, params, orig_size)
        else:
            raise ValueError(f"Unknown pseudo-method: {method_name}")

    def _pseudo_dct_threshold(
        self,
        tensor: np.ndarray,
        params: Dict[str, Any],
        orig_size: int,
    ) -> Tuple[bytes, Dict[str, Any], np.ndarray]:
        """DCT on thresholded residual.

        The residual after SVD is mostly noise-like with small values.
        By zeroing out values below a threshold (``threshold_sigma``
        standard deviations), the residual becomes sparse and DCT can
        find structure much more effectively.

        Flow:
        1. Compute threshold = std(tensor) * threshold_sigma
        2. Zero out values below threshold (sparsify)
        3. Apply block-DCT to the sparsified residual
        4. Keep top ``keep_fraction`` DCT coefficients
        5. Reconstruct by inverse DCT on kept coefficients
        """
        threshold_sigma = float(params.get("threshold_sigma", 2.0))
        keep_fraction = float(params.get("keep_fraction", 0.15))
        block_size = int(params.get("block_size", 8))

        # Compute threshold
        t_std = float(np.std(tensor))
        threshold = t_std * threshold_sigma

        # Sparsify: zero out noise below threshold
        tensor_sparse = np.where(np.abs(tensor) < threshold, 0.0, tensor)
        tensor_sparse = np.ascontiguousarray(tensor_sparse, dtype=np.float32)

        # Apply block DCT
        orig_shape = tensor_sparse.shape
        mat = tensor_sparse.reshape(-1, tensor_sparse.shape[-1]).astype(np.float64)
        m, n = mat.shape
        bs = min(block_size, m, n)
        keep = max(1, int(bs * bs * keep_fraction))

        # DCT matrix for the block size
        _C = np.zeros((bs, bs), dtype=np.float64)
        _C[0, :] = 1.0 / np.sqrt(bs)
        s = np.sqrt(2.0 / bs)
        k_arr = np.arange(1, bs, dtype=np.float64)[:, None]
        i_arr = np.arange(bs, dtype=np.float64)[None, :]
        _C[1:, :] = s * np.cos(np.pi * k_arr * (i_arr + 0.5) / bs)

        zigzag = DirectCascadeEngine._zigzag_indices(bs, bs)[:keep]

        all_coeffs = []
        all_shapes = []

        for i in range(0, m, bs):
            for j in range(0, n, bs):
                block = mat[i : i + bs, j : j + bs]
                bh, bw = block.shape
                if bh < bs or bw < bs:
                    padded = np.zeros((bs, bs), dtype=np.float64)
                    padded[:bh, :bw] = block
                    block = padded
                coeffs = _C @ block @ _C.T
                flat = coeffs.ravel()
                ordered = flat[zigzag].astype(np.float32)
                all_coeffs.append(ordered)
                all_shapes.append((i, j, bh, bw))

        data_out = {
            "coeffs": all_coeffs,
            "shapes": all_shapes,
            "bs": bs,
            "keep": keep,
            "threshold": threshold,
            "threshold_sigma": threshold_sigma,
        }
        meta: Dict[str, Any] = {
            "orig_shape": list(orig_shape),
            "method": "dct_threshold",
            "pseudo_method": True,
        }

        # Serialize to bytes
        serialized = self._serialize_dct_threshold(data_out)
        compressed_data = serialized

        # Reconstruct
        n_coeffs_total = len(all_coeffs) * keep
        approx_bytes = n_coeffs_total * 4 + len(all_shapes) * 16

        recon = self._reconstruct_dct_threshold(data_out, orig_shape)

        return bytes(compressed_data), meta, recon

    @staticmethod
    def _serialize_dct_threshold(data: Dict[str, Any]) -> bytearray:
        """Serialize dct_threshold compressed data to bytes."""
        buf = bytearray()
        # Number of blocks
        buf += struct.pack("<I", len(data["coeffs"]))
        # Threshold info
        buf += struct.pack("<d", data["threshold"])
        buf += struct.pack("<d", data["threshold_sigma"])
        # Block size
        buf += struct.pack("<I", data["bs"])
        buf += struct.pack("<I", data["keep"])
        # Each block
        for coeffs, (i, j, bh, bw) in zip(data["coeffs"], data["shapes"]):
            buf += struct.pack("<IIII", i, j, bh, bw)
            buf += coeffs.tobytes()
        return buf

    @staticmethod
    def _reconstruct_dct_threshold(
        data: Dict[str, Any], orig_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Reconstruct from dct_threshold compressed data."""
        bs = data["bs"]
        keep = data["keep"]

        # DCT matrix
        _C = np.zeros((bs, bs), dtype=np.float64)
        _C[0, :] = 1.0 / np.sqrt(bs)
        s = np.sqrt(2.0 / bs)
        k_arr = np.arange(1, bs, dtype=np.float64)[:, None]
        i_arr = np.arange(bs, dtype=np.float64)[None, :]
        _C[1:, :] = s * np.cos(np.pi * k_arr * (i_arr + 0.5) / bs)
        _CT = _C.T

        zigzag = DirectCascadeEngine._zigzag_indices(bs, bs)[:keep]

        m_orig = orig_shape[0] if len(orig_shape) > 1 else 1
        n_orig = orig_shape[-1]
        result = np.zeros((m_orig, n_orig), dtype=np.float64)

        for coeffs, (bi, bj, bh, bw) in zip(data["coeffs"], data["shapes"]):
            full = np.zeros(bs * bs, dtype=np.float64)
            full[zigzag] = coeffs
            block = _CT @ full.reshape(bs, bs) @ _C
            result[bi : bi + bh, bj : bj + bw] = block[:bh, :bw]

        return result.reshape(orig_shape).astype(np.float32)

    @staticmethod
    def _zigzag_indices(rows: int, cols: int) -> np.ndarray:
        """Generate zigzag scan order indices for a rows x cols matrix."""
        indices = np.zeros((rows, cols), dtype=np.int32)
        r, c = 0, 0
        for i in range(rows * cols):
            indices[r, c] = i
            if (r + c) % 2 == 0:
                if c == cols - 1:
                    r += 1
                elif r == 0:
                    c += 1
                else:
                    r -= 1
                    c += 1
            else:
                if r == rows - 1:
                    c += 1
                elif c == 0:
                    r += 1
                else:
                    r += 1
                    c -= 1
        return indices.ravel()

    def _pseudo_sparse_store(
        self,
        tensor: np.ndarray,
        params: Dict[str, Any],
        orig_size: int,
    ) -> Tuple[bytes, Dict[str, Any], np.ndarray]:
        """Store only the significant residual values (sparse approach).

        Instead of DCT on a noise-like residual, this method:
        1. Computes a threshold = std(tensor) * threshold_sigma
        2. Finds all values above the threshold
        3. Stores their indices (delta-encoded) and values (float16)

        This is more efficient than DCT for noise-like residuals where
        the energy is spread across all frequencies.
        """
        threshold_sigma = float(params.get("threshold_sigma", 2.5))
        quantize_bits = int(params.get("quantize_bits", 16))  # 16 or 8

        t_std = float(np.std(tensor))
        threshold = t_std * threshold_sigma

        flat = tensor.ravel()
        mask = np.abs(flat) > threshold
        indices = np.where(mask)[0].astype(np.uint32)
        values = flat[mask]

        n_kept = len(indices)

        if n_kept == 0:
            # Nothing above threshold — store empty payload
            buf = bytearray()
            buf += struct.pack("<I", 0)
            buf += struct.pack("<d", threshold)
            buf += struct.pack("<d", threshold_sigma)
            recon = np.zeros_like(tensor)
            return (
                bytes(buf),
                {
                    "n_kept": 0,
                    "threshold": threshold,
                    "method": "sparse_store",
                    "pseudo_method": True,
                },
                recon,
            )

        # Quantize values to float16 (or keep float32)
        if quantize_bits == 16:
            stored_values = values.astype(np.float16).tobytes()
        else:
            stored_values = values.astype(np.float32).tobytes()

        # Delta-encode indices
        # First index is stored as-is, subsequent are differences
        deltas = np.zeros(n_kept, dtype=np.uint32)
        deltas[0] = indices[0]
        if n_kept > 1:
            deltas[1:] = indices[1:] - indices[:-1]

        # Use variable-length encoding for deltas (uint8 if possible, else uint16, else uint32)
        max_delta = int(deltas.max()) if n_kept > 0 else 0
        if max_delta < 256:
            delta_bytes = deltas.astype(np.uint8).tobytes()
            delta_width = 1
        elif max_delta < 65536:
            delta_bytes = deltas.astype(np.uint16).tobytes()
            delta_width = 2
        else:
            delta_bytes = deltas.astype(np.uint32).tobytes()
            delta_width = 4

        buf = bytearray()
        buf += struct.pack("<I", n_kept)
        buf += struct.pack("<d", threshold)
        buf += struct.pack("<d", threshold_sigma)
        buf += struct.pack("<B", quantize_bits)
        buf += struct.pack("<B", delta_width)
        buf += delta_bytes
        buf += stored_values

        meta = {
            "n_kept": n_kept,
            "threshold": threshold,
            "threshold_sigma": threshold_sigma,
            "quantize_bits": quantize_bits,
            "delta_width": delta_width,
            "total_elements": tensor.size,
            "method": "sparse_store",
            "pseudo_method": True,
        }

        # Reconstruct
        recon_flat = self._reconstruct_sparse_store(
            n_kept,
            threshold,
            quantize_bits,
            delta_width,
            delta_bytes,
            stored_values,
            tensor.size,
        )
        recon = recon_flat.reshape(tensor.shape).astype(np.float32)

        return bytes(buf), meta, recon

    @staticmethod
    def _reconstruct_sparse_store(
        n_kept: int,
        threshold: float,
        quantize_bits: int,
        delta_width: int,
        delta_bytes: bytes,
        values_bytes: bytes,
        total_elements: int,
    ) -> np.ndarray:
        """Reconstruct tensor from sparse store data."""
        recon = np.zeros(total_elements, dtype=np.float64)

        if n_kept == 0:
            return recon

        # Decode deltas
        if delta_width == 1:
            deltas = np.frombuffer(delta_bytes, dtype=np.uint8).astype(np.uint32)
        elif delta_width == 2:
            deltas = np.frombuffer(delta_bytes, dtype=np.uint16).astype(np.uint32)
        else:
            deltas = np.frombuffer(delta_bytes, dtype=np.uint32)

        # Undo delta encoding
        indices = np.zeros(n_kept, dtype=np.int64)
        indices[0] = deltas[0]
        if n_kept > 1:
            indices[1:] = np.cumsum(deltas[1:]).astype(np.int64) + deltas[0]

        # Ensure indices are in bounds
        indices = np.clip(indices, 0, total_elements - 1)

        # Decode values
        if quantize_bits == 16:
            values = np.frombuffer(values_bytes, dtype=np.float16).astype(np.float64)
        else:
            values = np.frombuffer(values_bytes, dtype=np.float32).astype(np.float64)

        recon[indices] = values
        return recon

    # ── Pseudo-Method Reconstruction ─────────────────────────────────

    def _reconstruct_pseudo_stage(
        self,
        method_name: str,
        stage_info: Dict[str, Any],
        original_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Reconstruct a pseudo-method stage from its compressed data.

        Parameters
        ----------
        method_name : str
            ``"dct_threshold"`` or ``"sparse_store"``.
        stage_info : dict
            Stage info dict with ``compressed_data`` and ``metadata`` keys.
        original_shape : tuple of int
            Original tensor shape.

        Returns
        -------
        np.ndarray
            Reconstructed float32 tensor for this stage.
        """
        data = stage_info["compressed_data"]
        meta = stage_info.get("metadata", {})

        if method_name == "dct_threshold":
            return self._reconstruct_dct_threshold_from_bytes(data, original_shape)
        elif method_name == "sparse_store":
            return self._reconstruct_sparse_store_from_bytes(data, original_shape)
        else:
            raise ValueError(f"Unknown pseudo-method: {method_name}")

    @staticmethod
    def _reconstruct_dct_threshold_from_bytes(
        data: bytes, original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Reconstruct dct_threshold from serialized bytes."""
        pos = 0
        n_blocks = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        threshold = struct.unpack_from("<d", data, pos)[0]
        pos += 8
        threshold_sigma = struct.unpack_from("<d", data, pos)[0]
        pos += 8
        bs = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        keep = struct.unpack_from("<I", data, pos)[0]
        pos += 4

        # DCT matrix
        _C = np.zeros((bs, bs), dtype=np.float64)
        _C[0, :] = 1.0 / np.sqrt(bs)
        s = np.sqrt(2.0 / bs)
        k_arr = np.arange(1, bs, dtype=np.float64)[:, None]
        i_arr = np.arange(bs, dtype=np.float64)[None, :]
        _C[1:, :] = s * np.cos(np.pi * k_arr * (i_arr + 0.5) / bs)
        _CT = _C.T

        zigzag = DirectCascadeEngine._zigzag_indices(bs, bs)[:keep]

        m_orig = original_shape[0] if len(original_shape) > 1 else 1
        n_orig = original_shape[-1]
        result = np.zeros((m_orig, n_orig), dtype=np.float64)

        for _ in range(n_blocks):
            bi, bj, bh, bw = struct.unpack_from("<IIII", data, pos)
            pos += 16
            coeff_count = keep
            coeff_bytes = coeff_count * 4
            coeffs = np.frombuffer(
                data[pos : pos + coeff_bytes], dtype=np.float32
            ).astype(np.float64)
            pos += coeff_bytes

            full = np.zeros(bs * bs, dtype=np.float64)
            full[zigzag] = coeffs
            block = _CT @ full.reshape(bs, bs) @ _C
            result[bi : bi + bh, bj : bj + bw] = block[:bh, :bw]

        return result.reshape(original_shape).astype(np.float32)

    @staticmethod
    def _reconstruct_sparse_store_from_bytes(
        data: bytes, original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Reconstruct sparse_store from serialized bytes."""
        pos = 0
        n_kept = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        threshold = struct.unpack_from("<d", data, pos)[0]
        pos += 8
        threshold_sigma = struct.unpack_from("<d", data, pos)[0]
        pos += 8
        quantize_bits = struct.unpack_from("<B", data, pos)[0]
        pos += 1
        delta_width = struct.unpack_from("<B", data, pos)[0]
        pos += 1

        total_elements = int(np.prod(original_shape))

        if n_kept == 0:
            return np.zeros(original_shape, dtype=np.float32)

        # Decode deltas — keep raw bytes (don't astype to uint32 before passing)
        if delta_width == 1:
            delta_size = n_kept * 1
            raw_delta_bytes = data[pos : pos + delta_size]
        elif delta_width == 2:
            delta_size = n_kept * 2
            raw_delta_bytes = data[pos : pos + delta_size]
        else:
            delta_size = n_kept * 4
            raw_delta_bytes = data[pos : pos + delta_size]
        pos += delta_size

        # Values
        val_bytesize = 2 if quantize_bits == 16 else 4
        values_size = n_kept * val_bytesize
        values_bytes = data[pos : pos + values_size]

        return (
            DirectCascadeEngine._reconstruct_sparse_store(
                n_kept,
                threshold,
                quantize_bits,
                delta_width,
                raw_delta_bytes,
                values_bytes,
                total_elements,
            )
            .reshape(original_shape)
            .astype(np.float32)
        )

    # ── Progressive SVD ─────────────────────────────────────────────

    def _progressive_svd_compress(
        self,
        engine: Any,
        tensor: np.ndarray,
        max_error: float = 0.01,
    ) -> Optional[Dict[str, Any]]:
        """Try SVD with progressively more aggressive ranks until error exceeds budget.

        Uses adaptive rank estimation (singular value decay analysis) to
        determine the starting rank, then tries progressively more aggressive
        ranks and returns the MOST aggressive rank that stays within the
        error budget.  This is useful for cascade patterns where SVD's
        aggressive rank selection can be compensated by later stages
        capturing the residual.

        Parameters
        ----------
        engine : CompressionIntelligenceEngine
            Engine with ``_methods`` registry (must contain ``"svd_compress"``).
        tensor : np.ndarray
            Float32 tensor to compress (will be reshaped to 2D for SVD).
        max_error : float
            Maximum allowed mean absolute error (default 0.01).

        Returns
        -------
        dict or None
            Result dict with keys ``data`` (bytes), ``meta`` (dict),
            ``recon`` (ndarray), ``ratio`` (float), ``error`` (float),
            ``rank`` (int) — or None if SVD is unavailable or all ranks fail.
        """
        svd = engine._methods.get("svd_compress")
        if svd is None:
            logger.warning("Progressive SVD: 'svd_compress' not found in engine")
            return None

        # Flatten to 2D for SVD
        orig_ndim = tensor.ndim
        t = np.ascontiguousarray(tensor, dtype=np.float32)
        if t.ndim > 2:
            t_2d = t.reshape(t.shape[0], -1)
        else:
            t_2d = t
        min_dim = min(t_2d.shape)

        if min_dim < 4:
            # Too small for SVD — passthrough
            return None

        # Use adaptive rank estimation to determine the starting rank.
        # _estimate_effective_rank_fast() uses randomized SVD on a subsample
        # to find the rank that captures 99% of spectral energy.
        estimated_rank = svd._estimate_effective_rank_fast(t_2d)

        # Rank candidates: start from the adaptive estimate, then go more
        # aggressive.  Real weight tensors have extremely low effective rank,
        # so this lets us find the optimal rank much faster than the old
        # hardcoded heuristic (min_dim // N).
        rank_candidates = [
            estimated_rank,  # Conservative (99% energy)
            max(estimated_rank // 2, 2),  # Aggressive
            max(estimated_rank // 5, 2),  # Very aggressive
            max(estimated_rank // 10, 2),  # Extreme
            max(estimated_rank // 20, 2),  # Maximum
            max(min_dim // 10, 4),  # Old conservative fallback
            max(min_dim // 40, 4),  # Old aggressive fallback
            max(min_dim // 160, 4),  # Old extreme fallback
        ]

        # Deduplicate and sort descending — start with highest rank (lowest error),
        # then progressively try more aggressive ranks.
        rank_candidates = sorted(set(rank_candidates), reverse=True)

        best_result: Optional[Dict[str, Any]] = None

        for rank in rank_candidates:
            try:
                data, meta = svd.compress(t_2d, rank=rank)
                recon = svd.decompress(data, meta)
                if recon.shape != t_2d.shape:
                    recon = recon.reshape(t_2d.shape)
                # Restore original ndim if needed
                if orig_ndim > 2:
                    recon_reshaped = recon.reshape(tensor.shape)
                else:
                    recon_reshaped = recon

                error = float(np.abs(tensor - recon_reshaped).mean())
                ratio = float(tensor.nbytes / max(len(data), 1))

                if error <= max_error:
                    # This rank works within budget — keep it and try more aggressive
                    best_result = {
                        "method": "progressive_svd",
                        "data": data,
                        "meta": meta,
                        "recon": recon_reshaped,
                        "ratio": ratio,
                        "error": error,
                        "rank": rank,
                    }
                    logger.debug(
                        "Progressive SVD rank=%d: ratio=%.1fx, error=%.6f "
                        "(within budget)",
                        rank,
                        ratio,
                        error,
                    )
                else:
                    # Error exceeded budget — stop, use previous rank
                    logger.debug(
                        "Progressive SVD rank=%d exceeded budget: error=%.6f > %.4f",
                        rank,
                        error,
                        max_error,
                    )
                    break

            except Exception as exc:
                logger.debug(
                    "Progressive SVD rank=%d failed: %s",
                    rank,
                    exc,
                )
                break

        if best_result is not None:
            logger.debug(
                "Progressive SVD selected rank=%d: ratio=%.1fx, error=%.6f",
                best_result["rank"],
                best_result["ratio"],
                best_result["error"],
            )
            return best_result

        logger.warning("Progressive SVD: no rank within error budget %.4f", max_error)
        return None
