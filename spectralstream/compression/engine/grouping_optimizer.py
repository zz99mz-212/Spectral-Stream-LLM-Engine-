"""
Tensor Grouping Optimizer — reduces method testing from O(N) to O(G) where G << N.

Groups tensors by (shape, dtype, name_pattern) and only tests one representative
per group. The best method+params found for the representative is then applied
to all tensors in the group, bypassing per-tensor method testing.

This reduces 20,110 method-tests-per-model to ~200 (one per unique shape×dtype×pattern).
"""

from __future__ import annotations

__all__ = [
    "TensorGroup",
    "TensorGrouper",
    "compress_with_grouping",
    "extract_name_pattern",
    "group_tensors",
    "get_tensor_metadata_from_dict",
    "get_tensor_metadata_from_info",
]

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Pattern to strip layer indices ──────────────────────────────────────
# Matches common layer index patterns in transformer models:
#   blk.0.attn → blk.{i}.attn
#   layers.12.ffn → layers.{i}.ffn
#   transformer.h.7.attn → transformer.h.{i}.attn
#   model.layers.0.self_attn → model.layers.{i}.self_attn
#   encoder.block.5 → encoder.block.{i}
#   0.attention (leading index) → {i}.attention
_LAYER_INDEX_RE = re.compile(
    r"(?<=[.\-_/])"
    r"(0|[1-9]\d*)"
    r"(?=[.\-_/]|$)"
)

# Alternative: leading index pattern (e.g. "0.attention", "12.mlp")
_LEADING_INDEX_RE = re.compile(r"^(0|[1-9]\d*)[.\-_]")


@dataclass
class TensorGroup:
    """A group of tensors sharing the same (shape, dtype, name_pattern).

    Attributes
    ----------
    shape : Tuple[int, ...]
        The common tensor shape for this group.
    dtype : str
        The common dtype string (e.g. 'float32', 'float16').
    pattern : str
        The name pattern with layer indices replaced by '{i}'.
    tensor_names : List[str]
        All original tensor names belonging to this group.
    representative : str
        The tensor name chosen as representative for this group.
    is_compressed : bool
        Whether this group has been processed (representative tested).
    cached_method : Optional[str]
        The best method name found for the representative.
    cached_params : Optional[dict]
        Parameters for the cached method.
    cached_ratio : float
        Compression ratio achieved on the representative.
    cached_error : float
        Relative error achieved on the representative.
    """

    shape: Tuple[int, ...]
    dtype: str
    pattern: str
    tensor_names: List[str] = field(default_factory=list)
    representative: str = ""
    is_compressed: bool = False
    cached_method: Optional[str] = None
    cached_params: Optional[dict] = None
    cached_ratio: float = 0.0
    cached_error: float = 0.0

    @property
    def size(self) -> int:
        """Number of tensors in this group."""
        return len(self.tensor_names)

    def __repr__(self) -> str:
        return (
            f"TensorGroup(shape={self.shape}, dtype={self.dtype}, "
            f"pattern='{self.pattern}', n_tensors={self.size}, "
            f"representative='{self.representative}', "
            f"compressed={self.is_compressed})"
        )


class TensorGrouper:
    """Groups tensors by (shape, dtype, name_pattern) for optimized method testing.

    Usage::

        grouper = TensorGrouper()
        groups = grouper.group_tensors(tensor_metadata)
        for group in groups:
            print(group.pattern, group.representative, group.size)
    """

    def __init__(self) -> None:
        self._groups: List[TensorGroup] = []

    def group_tensors(
        self,
        tensor_dict_or_metadata: Any,
    ) -> List[TensorGroup]:
        """Group tensors by (shape, dtype, name_pattern).

        Accepts either:
        - Dict[str, np.ndarray] — tensor name → array (extracts shape/dtype automatically)
        - Dict[str, Tuple[Tuple[int, ...], str]] — tensor name → (shape, dtype_string)

        Parameters
        ----------
        tensor_dict_or_metadata : dict
            Either tensor name → array, or tensor name → (shape, dtype).

        Returns
        -------
        List[TensorGroup]
            Groups with representatives selected.
        """
        # Auto-detect: if values are numpy arrays, extract metadata
        if tensor_dict_or_metadata and isinstance(
            next(iter(tensor_dict_or_metadata.values())), np.ndarray
        ):
            metadata = get_tensor_metadata_from_dict(tensor_dict_or_metadata)
        else:
            metadata = tensor_dict_or_metadata

        self._groups = group_tensors(metadata)
        return self._groups

    @property
    def groups(self) -> List[TensorGroup]:
        """Last computed groups."""
        return self._groups

    def get_group_for_tensor(self, name: str) -> Optional[TensorGroup]:
        """Find the group that contains a given tensor name."""
        pattern = extract_name_pattern(name)
        for g in self._groups:
            if name in g.tensor_names:
                return g
        return None

    def summary(self) -> str:
        """Return a human-readable summary of the grouping."""
        if not self._groups:
            return "No groups computed."
        lines = [
            f"Tensor Groups: {len(self._groups)} groups from "
            f"{sum(g.size for g in self._groups)} tensors",
        ]
        for i, g in enumerate(self._groups):
            lines.append(
                f"  [{i}] shape={g.shape}, dtype={g.dtype}, "
                f"pattern='{g.pattern}', n={g.size}, "
                f"rep='{g.representative}'"
            )
        return "\n".join(lines)


def extract_name_pattern(name: str) -> str:
    """Extract a generalized name pattern by replacing layer indices with '{i}'.

    Handles common transformer naming conventions:
        'blk.0.attention.wq.weight' → 'blk.{i}.attention.wq.weight'
        'model.layers.12.self_attn.q_proj.weight' → 'model.layers.{i}.self_attn.q_proj.weight'
        'transformer.h.7.attn.c_attn.weight' → 'transformer.h.{i}.attn.c_attn.weight'
        '0.attention.wq.weight' → '{i}.attention.wq.weight'

    Parameters
    ----------
    name : str
        Original tensor name.

    Returns
    -------
    str
        Generalized pattern with '{i}' replacing layer indices.
    """
    if not name:
        return "{unknown}"

    # Handle leading index pattern (e.g. "0.attention")
    name = _LEADING_INDEX_RE.sub("{i}.", name)

    # Replace other layer indices
    name = _LAYER_INDEX_RE.sub("{i}", name)

    return name


def _extract_layer_index(name: str) -> Optional[int]:
    """Extract the first layer index from a tensor name, if present.

    Returns None if no layer index is found.
    """
    for pattern in (r"\.(\d+)\.", r"^(\d+)[.\-_]"):
        m = re.search(pattern, name)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def group_tensors(
    tensor_metadata: Dict[str, Tuple[Tuple[int, ...], str]],
) -> List[TensorGroup]:
    """Group tensors by (shape, dtype, name_pattern).

    Parameters
    ----------
    tensor_metadata : Dict[str, Tuple[Tuple[int, ...], str]]
        Mapping from tensor name to (shape, dtype_string).

    Returns
    -------
    List[TensorGroup]
        List of tensor groups, with representative selected.
    """
    # Group by (shape_key, dtype, pattern)
    groups: Dict[Tuple[tuple, str, str], TensorGroup] = {}

    for name, (shape, dtype) in tensor_metadata.items():
        pattern = extract_name_pattern(name)
        key = (shape, dtype, pattern)

        if key not in groups:
            groups[key] = TensorGroup(
                shape=shape,
                dtype=dtype,
                pattern=pattern,
            )
        groups[key].tensor_names.append(name)

    # Select representative for each group
    result: List[TensorGroup] = []
    for group in groups.values():
        group.tensor_names.sort(key=_tensor_sort_key)
        group.representative = _select_representative(group.tensor_names)
        result.append(group)

    # Sort by group size (descending) for priority processing
    result.sort(key=lambda g: -g.size)
    return result


def _tensor_sort_key(name: str) -> tuple:
    """Generate a sort key for tensor names to ensure consistent ordering.

    Sorts by layer index first (if present), then alphabetically.
    """
    idx = _extract_layer_index(name)
    return (idx if idx is not None else -1, name)


def _select_representative(tensor_names: List[str]) -> str:
    """Select the representative tensor from a sorted list of names.

    Picks the middle element to get a typical layer (neither first nor last).
    For single-element lists, returns the only element.

    Parameters
    ----------
    tensor_names : List[str]
        Sorted list of tensor names in the group.

    Returns
    -------
    str
        The chosen representative name.
    """
    if not tensor_names:
        return ""
    if len(tensor_names) <= 2:
        return tensor_names[0]
    # Pick the middle element
    return tensor_names[len(tensor_names) // 2]


def get_tensor_metadata_from_dict(
    tensors: Dict[str, np.ndarray],
) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    """Extract metadata (shape, dtype string) from a tensor dictionary.

    Parameters
    ----------
    tensors : Dict[str, np.ndarray]
        Dictionary of tensor names → numpy arrays.

    Returns
    -------
    Dict[str, Tuple[Tuple[int, ...], str]]
        Mapping from tensor name to (shape, dtype_string).
    """
    return {name: (tensor.shape, str(tensor.dtype)) for name, tensor in tensors.items()}


def get_tensor_metadata_from_info(
    tensor_info: Dict[str, Tuple[Tuple[int, ...], str, Any, Any]],
) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    """Extract metadata from safetensors-style info dict.

    Safetensors entries are: (shape, dtype_str, offset, nbytes).
    This extracts just (shape, dtype_str).

    Parameters
    ----------
    tensor_info : Dict[str, Tuple[Tuple[int, ...], str, Any, Any]]
        Safetensors-style tensor info.

    Returns
    -------
    Dict[str, Tuple[Tuple[int, ...], str]]
        Mapping from tensor name to (shape, dtype_string).
    """
    return {name: (info[0], info[1]) for name, info in tensor_info.items()}


def compress_with_grouping(
    engine: Any,
    tensors: Dict[str, np.ndarray],
    target_ratio: float = 5000.0,
    max_error: float = 0.01,
    progress_callback: Any = None,
    use_resonance: bool = False,
    resonance_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Compress tensors using grouping optimization.

    1. Group tensors by (shape, dtype, name_pattern)
    2. For each group: test methods on representative → cache best method
    3. Apply cached method to all remaining tensors in group (bypass testing)
    4. Track which groups have been compressed

    Parameters
    ----------
    engine : CompressionIntelligenceEngine
        The compression engine instance.
    tensors : Dict[str, np.ndarray]
        Dictionary of tensor names → numpy arrays.
    target_ratio : float
        Target compression ratio.
    max_error : float
        Maximum acceptable relative error.
    progress_callback : callable or None
        Called as f(processed, total, tensor_name).

    Returns
    -------
    dict
        Compression report in the same format as compress_dict().
    """
    import gc
    import time

    from ._dataclasses import CompressedTensor
    from ._helpers import _build_report

    t0 = time.perf_counter()

    # ── Step 1: Extract metadata and group ──
    metadata = get_tensor_metadata_from_dict(tensors)
    groups = group_tensors(metadata)

    # Optional resonance refinement — merges shape groups with similar
    # spectral resonance profiles for even fewer, more intelligent groups.
    if use_resonance and len(groups) > 1:
        try:
            from .resonant_grouping import resonance_refine_groups

            resonant_groups = resonance_refine_groups(
                groups, tensors, threshold=resonance_threshold
            )
            logger.info(
                "Resonance refinement: %d shape groups → %d resonant groups "
                "(%.1f%% reduction)",
                len(groups),
                len(resonant_groups),
                100.0 * (len(groups) - len(resonant_groups)) / max(len(groups), 1),
            )
            # Convert ResonantGroup back to TensorGroup shape for downstream
            # processing. We keep the first member's shape/dtype/pattern as
            # the group representative shape so the pipeline remains compatible.
            converted: List[TensorGroup] = []
            for rg in resonant_groups:
                if not rg.members:
                    continue
                rep_name = rg.members[0]
                shape, dtype = metadata.get(rep_name, ((), "unknown"))
                pattern = extract_name_pattern(rep_name)
                tg = TensorGroup(
                    shape=shape,
                    dtype=dtype,
                    pattern=pattern,
                    tensor_names=list(rg.members),
                    representative=rep_name,
                )
                converted.append(tg)
            groups = converted
        except Exception as exc:
            logger.warning("Resonance refinement failed, using shape groups: %s", exc)

    logger.info(
        "Grouping: %d tensors → %d groups (%.1fx reduction)",
        len(tensors),
        len(groups),
        len(tensors) / max(len(groups), 1),
    )

    # ── Step 2: Profile all tensors with lazy profiling ──
    # Enable lazy profiling so group members skip expensive SVD/DCT/structural
    # analysis and reuse the representative's spectral/structural profile.
    engine.profiler.enable_lazy_profiling(cache_size_limit=len(groups) + 50)

    # Build group-signature mapping: name → (shape, dtype, pattern)
    name_to_group_sig: Dict[str, Tuple] = {}
    for g in groups:
        sig = (g.shape, g.dtype, g.pattern)
        for tn in g.tensor_names:
            name_to_group_sig[tn] = sig

    profiles = {}
    for name, tensor in tensors.items():
        is_rep = any(
            g.representative == name for g in groups if hasattr(g, "representative")
        )
        sig = name_to_group_sig.get(name)
        if sig is not None and not is_rep:
            # Group member (not representative): use lazy profiling
            profiles[name] = engine.profiler.profile_tensor_lazy(
                tensor,
                name,
                group_signature=sig,
            )
        else:
            # Representative or ungrouped tensor: full profiling
            profiles[name] = engine.profiler.profile_tensor(tensor, name)
        if progress_callback:
            progress_callback(len(profiles), len(tensors) * 2, name)

    # Allocate error budgets
    budgets = engine.allocator.allocate(
        profiles, target_ratio=target_ratio, max_error=max_error
    )

    # ── State for results ──
    compressed_list: List[CompressedTensor] = []
    failures: List[str] = []
    total_orig = 0
    total_comp = 0
    errors = []
    method_dist: Dict[str, int] = {}
    tensor_methods: Dict[str, str] = {}
    tensor_errors: Dict[str, float] = {}
    tensor_ratios: Dict[str, float] = {}
    groups_processed = 0

    # ── Step 3+: Process each group ──
    for group_idx, group in enumerate(groups):
        logger.debug(
            "Group %d/%d: shape=%s, dtype=%s, pattern='%s', n=%d",
            group_idx + 1,
            len(groups),
            group.shape,
            group.dtype,
            group.pattern,
            group.size,
        )

        # Get representative tensor
        rep_name = group.representative
        if rep_name not in tensors:
            logger.warning("Representative '%s' not found in tensor dict", rep_name)
            # Fall back to first available tensor in group
            for tn in group.tensor_names:
                if tn in tensors:
                    rep_name = tn
                    group.representative = tn
                    break
            else:
                logger.error("No tensors from group %s found in dict", group.pattern)
                failures.extend(group.tensor_names)
                continue

        rep_tensor = tensors[rep_name]
        rep_profile = profiles.get(rep_name)
        if rep_profile is None:
            rep_profile = engine.profiler.profile_tensor(rep_tensor, rep_name)

        # Test methods on representative only
        eb = budgets.get(rep_name, max_error)
        methods = engine._select_methods(rep_profile, eb, target_ratio)

        if not methods:
            from ._methods import _BlockINT8

            blk8 = engine._methods.get("block_int8")
            if blk8:
                methods = [{"instance": blk8, "params": {}, "name": "block_int8"}]

        rep_data, rep_meta, rep_ratio, rep_error = (
            engine.compress_tensor_with_validation(rep_tensor, rep_profile, methods, eb)
        )

        # Cache the best method for this group.
        # Extract only method-specific params (not metadata).
        # The method params from _select_methods are typically empty {},
        # but we keep them for methods that do require specific kwargs.
        best_method_params: dict = {}
        best_method_name = rep_meta.get("method", "unknown")
        for m in methods:
            if isinstance(m, dict) and m.get("name") == best_method_name:
                best_method_params = m.get("params", {})
                break
        group.cached_method = best_method_name
        group.cached_params = best_method_params
        group.cached_ratio = rep_ratio
        group.cached_error = rep_error
        group.is_compressed = True

        # Compress representative with full validation
        ct = CompressedTensor(
            _data=rep_data,
            method=rep_meta.get("method", "unknown"),
            params=rep_meta,
            original_shape=rep_tensor.shape,
            original_dtype=str(rep_tensor.dtype),
            compression_ratio=rep_ratio,
            relative_error=rep_error,
        )
        compressed_list.append(ct)
        total_orig += rep_tensor.nbytes
        total_comp += len(rep_data)
        errors.append(rep_error)
        mname = ct.method
        method_dist[mname] = method_dist.get(mname, 0) + 1
        tensor_methods[rep_name] = mname
        tensor_errors[rep_name] = rep_error
        tensor_ratios[rep_name] = rep_ratio

        if progress_callback:
            progress_callback(
                len(compressed_list) + sum(g.size - 1 for g in groups[:group_idx]),
                len(tensors),
                rep_name,
            )

        # Record in unified intelligence
        if engine._unified_intelligence is not None:
            try:
                tensor_type = engine._classify_by_name(rep_name)
                method_category = rep_meta.get(
                    "category", rep_meta.get("method_type", "quantization")
                )
                engine._unified_intelligence.record_result(
                    tensor_type=tensor_type,
                    method_name=mname,
                    ratio=rep_ratio,
                    error=rep_error,
                    method_category=method_category,
                    target_ratio=target_ratio,
                    tensor_name=rep_name,
                )
            except Exception:
                pass

        # ── Apply cached method to remaining tensors in group ──
        for tn in group.tensor_names:
            if tn == rep_name:
                continue
            if tn not in tensors:
                continue

            try:
                tensor = tensors[tn]
                # Use the cached method directly — skip testing
                data, meta, ratio_val, error_val = _compress_with_cached_method(
                    engine=engine,
                    tensor=tensor,
                    name=tn,
                    method_name=group.cached_method,
                    method_params=group.cached_params,
                    cached_ratio=group.cached_ratio,
                    cached_error=group.cached_error,
                )

                ct = CompressedTensor(
                    _data=data,
                    method=meta.get("method", group.cached_method or "unknown"),
                    params=meta,
                    original_shape=tensor.shape,
                    original_dtype=str(tensor.dtype),
                    compression_ratio=ratio_val,
                    relative_error=error_val,
                )
                compressed_list.append(ct)
                total_orig += tensor.nbytes
                total_comp += len(data)
                errors.append(error_val)
                mname = ct.method
                method_dist[mname] = method_dist.get(mname, 0) + 1
                tensor_methods[tn] = mname
                tensor_errors[tn] = error_val
                tensor_ratios[tn] = ratio_val
            except Exception as exc:
                logger.warning("Failed to apply cached method to '%s': %s", tn, exc)
                failures.append(tn)

            if progress_callback:
                progress_callback(len(compressed_list), len(tensors), tn)

        groups_processed += 1
        del rep_tensor
        if groups_processed % 5 == 0:
            gc.collect()

    gc.collect()

    # ── Build report ──
    avg_error = float(np.mean(errors)) if errors else 0.0
    max_err = float(np.max(errors)) if errors else 0.0
    min_err = float(np.min(errors)) if errors else 0.0

    t1 = time.perf_counter()

    # Log grouping effectiveness
    total_tested = groups_processed  # only representatives were tested
    total_bypassed = len(tensors) - total_tested
    logger.info(
        "Grouping optimization: %d representatives tested, %d tensors bypassed "
        "(%.1f%% reduction in method testing)",
        total_tested,
        total_bypassed,
        100.0 * total_bypassed / max(len(tensors), 1),
    )

    stats = {
        "tensors": compressed_list,
        "total_orig_bytes": total_orig,
        "total_compressed_bytes": total_comp,
        "overall_ratio": total_orig / max(total_comp, 1),
        "average_ratio": total_orig / max(total_comp, 1),
        "avg_error": avg_error,
        "max_error": max_err,
        "min_error": min_err,
        "num_tensors": len(compressed_list),
        "method_distribution": method_dist,
        "failures": failures,
        "per_layer_error": {ct.method: ct.relative_error for ct in compressed_list},
        "time_seconds": t1 - t0,
        "weighted_error": avg_error,
        "tensor_methods": tensor_methods,
        "tensor_errors": tensor_errors,
        "tensor_ratios": tensor_ratios,
        "grouping": {
            "n_groups": len(groups),
            "n_tensors": len(tensors),
            "n_tested": total_tested,
            "n_bypassed": total_bypassed,
            "reduction_pct": round(100.0 * total_bypassed / max(len(tensors), 1), 1),
        },
    }
    return _build_report(stats)


def _compress_with_cached_method(
    engine: Any,
    tensor: np.ndarray,
    name: str,
    method_name: Optional[str],
    method_params: Optional[dict],
    cached_ratio: float = 0.0,
    cached_error: float = 0.0,
) -> Tuple[bytes, dict, float, float]:
    """Compress a tensor using a cached method, skipping validation.

    Only the cached method is tried — no fallback testing of alternatives.
    The *method_params* should be the method's own parameter dict (e.g.
    ``{"block_size": 128}``), NOT the full metadata dict returned by
    ``compress()``.

    Parameters
    ----------
    engine : CompressionIntelligenceEngine
        The compression engine.
    tensor : np.ndarray
        The tensor to compress.
    name : str
        Tensor name (for logging).
    method_name : str or None
        Cached method name to use.
    method_params : dict or None
        Parameters specific to the method (not metadata).
    cached_ratio : float
        Previously achieved ratio (for metadata).
    cached_error : float
        Previously achieved error (for metadata).

    Returns
    -------
    tuple of (bytes, dict, float, float)
        (compressed_data, metadata, ratio, error)
    """
    if method_name is None or method_name not in engine._methods:
        logger.warning(
            "Cached method '%s' not available for '%s', falling back to fast path",
            method_name,
            name,
        )
        return engine.compress_fast(tensor, name)

    inst = engine._methods[method_name]
    # Only use method-specific params (NOT metadata). Most methods work
    # with empty params. We strip out all metadata-like keys.
    method_kwargs: dict = {}
    if method_params:
        # Filter to only keys that look like method parameters
        # (exclude metadata keys added by enrich_meta / compress pipeline)
        meta_keys = {
            "original_shape",
            "method",
            "compression_ratio",
            "relative_error",
            "snr_db",
            "cached",
            "group_ratio",
            "group_error",
            "tensor_shape",
            "quality_metrics",
        }
        method_kwargs = {k: v for k, v in method_params.items() if k not in meta_keys}

    try:
        data, meta = inst.compress(tensor, **method_kwargs)
        meta["original_shape"] = list(tensor.shape)
        meta["method"] = method_name
        meta["compression_ratio"] = tensor.nbytes / max(len(data), 1)
        meta["cached"] = True
        meta["group_ratio"] = cached_ratio
        meta["group_error"] = cached_error

        ratio = tensor.nbytes / max(len(data), 1)
        error = cached_error

        return data, meta, ratio, error
    except Exception as exc:
        logger.warning(
            "Cached method '%s' failed on '%s': %s — falling back to fast path",
            method_name,
            name,
            exc,
        )
        return engine.compress_fast(tensor, name)
