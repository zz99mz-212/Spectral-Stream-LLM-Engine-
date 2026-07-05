"""Method variant generator — creates parameterized variants of all compression methods.

Expands ~397 base methods into ~1200+ total methods via parameter grids.
Engine methods get detailed multi-parameter grids.
Category-based methods get programmatic single-parameter grids.
Tier 5 (quantization) gets minimal grids.
"""

from __future__ import annotations


import inspect
from typing import Any, Dict, Optional, Tuple

import numpy as np

from spectralstream.compression.engine._methods import (
    _BlockINT8,
    _BlockINT4,
    _HadamardINT8,
    _HadamardINT4,
    _SparsityINT4,
    _DeltaINT4,
    _SVDCompress,
    _DCTSpectral,
    _TensorTrain,
    _FWHTCompress,
)


class MethodVariant:
    """Wrapper that binds fixed parameters to a compression method.

    The intelligence engine can freely pick from named variants —
    each wraps the same base algorithm with different parameterization.

    Each variant acts as a drop-in replacement for the base class:
    - Same .compress(tensor, **kw) / .decompress(data, metadata) interface
    - Fixed parameters are merged with any runtime kwargs
    - Category attribute is preserved for method selection logic
    """

    def __init__(
        self,
        base_cls: Any,
        name: str,
        category: Optional[str] = None,
        **fixed_params: Any,
    ) -> None:
        self.name = name
        self.category = category or getattr(base_cls, "category", "variant")
        self._base_cls = base_cls
        self._base = base_cls() if isinstance(base_cls, type) else base_cls
        self._params = fixed_params

    def __call__(self, **kw: Any) -> MethodVariant:
        """Allow use as drop-in replacement for base class in METHOD_CLASSES.

        Returns self so that 'cls()' patterns in tests and discovery work.
        """
        return self

    def compress(self, tensor: np.ndarray, **kw: Any) -> Tuple[bytes, dict]:
        merged = {**self._params, **kw}
        return self._base.compress(tensor, **merged)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return self._base.decompress(data, metadata)

    def __repr__(self) -> str:
        return f"<MethodVariant '{self.name}' (base={self._base_cls.__name__}) params={self._params}>"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _norm(v: str) -> str:
    return v.replace(".", "_").replace("-", "_").replace(" ", "_")


def _supports(cls_or_inst: Any, param: str) -> bool:
    try:
        sig = inspect.signature(cls_or_inst.compress)
        return param in sig.parameters
    except (TypeError, ValueError, AttributeError):
        return False


def _grid_variants(
    variants: Dict[str, MethodVariant],
    cls: Any,
    base_name: str,
    category: str,
    param: str,
    values: list,
) -> None:
    """Create one variant per param value if the method supports it."""
    if not _supports(cls, param):
        return
    for val in values:
        val_str = _norm(str(val))
        vname = _norm(f"{base_name}_{param}_{val_str}")
        if vname in variants:
            continue
        variants[vname] = MethodVariant(cls, vname, category=category, **{param: val})


# ── Engine method variants (70 total) ──────────────────────────────────────


def _engine_variants() -> Dict[str, MethodVariant]:
    v: Dict[str, MethodVariant] = {}

    # block_int8: 7 variants
    for bs in [16, 32, 64, 128, 256, 512, 1024]:
        v[f"block_int8_bs{bs}"] = MethodVariant(
            _BlockINT8, f"block_int8_bs{bs}", "quantization", block_size=bs
        )

    # block_int4: 5 variants
    for bs in [8, 16, 32, 64, 128]:
        v[f"block_int4_bs{bs}"] = MethodVariant(
            _BlockINT4, f"block_int4_bs{bs}", "quantization", block_size=bs
        )

    # hadamard_int8: 5 variants
    for bs in [32, 64, 128, 256, 512]:
        v[f"hadamard_int8_bs{bs}"] = MethodVariant(
            _HadamardINT8, f"hadamard_int8_bs{bs}", "transform_quant", block_size=bs
        )

    # hadamard_int4: 4 variants
    for bs in [8, 16, 32, 64]:
        v[f"hadamard_int4_bs{bs}"] = MethodVariant(
            _HadamardINT4, f"hadamard_int4_bs{bs}", "transform_quant", block_size=bs
        )

    # sparsity_int4: 3 variants
    for gs in [16, 32, 64]:
        v[f"sparsity_int4_gs{gs}"] = MethodVariant(
            _SparsityINT4, f"sparsity_int4_gs{gs}", "sparsity_quant", group_size=gs
        )

    # delta_int4: 3 variants
    for bs in [16, 32, 64]:
        v[f"delta_int4_bs{bs}"] = MethodVariant(
            _DeltaINT4, f"delta_int4_bs{bs}", "delta_quant", block_size=bs
        )

    # svd_compress: 20 variants (7 rank + 5 error_budget + 8 combined)
    for r in [4, 8, 16, 32, 64, 128, 256]:
        v[f"svd_rank{r}"] = MethodVariant(
            _SVDCompress, f"svd_rank{r}", "decomposition", rank=r
        )
    for eb in [0.001, 0.005, 0.01, 0.05, 0.1]:
        eb_s = _norm(str(eb))
        v[f"svd_eb{eb_s}"] = MethodVariant(
            _SVDCompress, f"svd_eb{eb_s}", "decomposition", error_budget=eb
        )
    for r, eb in [
        (4, 0.001),
        (8, 0.001),
        (16, 0.005),
        (32, 0.01),
        (64, 0.005),
        (128, 0.01),
        (256, 0.05),
        (16, 0.001),
    ]:
        eb_s = _norm(str(eb))
        v[f"svd_r{r}_eb{eb_s}"] = MethodVariant(
            _SVDCompress,
            f"svd_r{r}_eb{eb_s}",
            "decomposition",
            rank=r,
            error_budget=eb,
        )

    # dct_spectral: 10 variants
    for kr in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
        kr_s = _norm(str(kr))
        v[f"dct_kr{kr_s}"] = MethodVariant(
            _DCTSpectral, f"dct_kr{kr_s}", "spectral", keep_ratio=kr
        )

    # tensor_train: 6 variants
    for r in [4, 8, 16, 32, 64, 128]:
        v[f"tt_rank{r}"] = MethodVariant(
            _TensorTrain, f"tt_rank{r}", "tensor_network", rank=r
        )

    # fwht_compress: 7 variants
    for kr in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        kr_s = _norm(str(kr))
        v[f"fwht_kr{kr_s}"] = MethodVariant(
            _FWHTCompress, f"fwht_kr{kr_s}", "spectral", keep_ratio=kr
        )

    return v


# ── Universal & category parameter grids ─────────────────────────────────
# UNIVERSAL_PARAM_GRID is tried for ALL methods regardless of category.
# CATEGORY_PARAM_GRIDS are tried after, per category.
# The catch-all fallback handles methods with unrecognized param names.
# Max 5 variants per method to keep the set manageable for the engine.

_MAX_VARIANTS_PER_METHOD = 5

UNIVERSAL_PARAM_GRID: list = [
    ("rank", [4, 8, 16, 32, 64]),
    ("block_size", [16, 32, 64, 128]),
    ("threshold", [0.001, 0.01, 0.05, 0.1]),
    ("sparsity", [0.1, 0.3, 0.5, 0.7, 0.9]),
    ("keep_fraction", [0.01, 0.05, 0.1, 0.2, 0.5]),
    ("density", [0.1, 0.3, 0.5, 0.7]),
    ("n_components", [4, 8, 16, 32, 64]),
    ("n_bits", [4, 6, 8]),
    ("bond_dim", [4, 8, 16, 32]),
    ("target_energy", [0.9, 0.95, 0.99, 0.999]),
    ("eps", [0.001, 0.01, 0.05, 0.1]),
    ("n_blocks", [2, 4, 8]),
    ("temperature", [0.1, 0.5, 1.0, 2.0]),
    ("alpha", [0.1, 0.5, 1.0, 2.0]),
    ("beta", [0.1, 0.5, 1.0, 2.0]),
    ("gamma", [0.1, 0.5, 1.0]),
    ("n_layers", [2, 3, 4]),
    ("n_iter", [10, 20, 50]),
    ("n_iters", [10, 20, 50]),
    ("max_iters", [10, 20, 30]),
    ("prune_ratio", [0.1, 0.3, 0.5, 0.7, 0.9]),
    ("degree", [4, 8, 12, 16]),
    ("n_coeffs", [8, 16, 32, 64]),
    ("keep_ratio", [0.01, 0.05, 0.1, 0.2, 0.5]),
    ("keep_frac", [0.01, 0.05, 0.1, 0.2, 0.5]),
]

CATEGORY_PARAM_GRIDS: Dict[str, list] = {
    "decomposition": [
        ("rank", [4, 8, 16, 32, 64]),
        ("bond_dim", [4, 8, 16, 32]),
        ("block_size", [16, 32, 64]),
        ("n_blocks", [2, 4, 8]),
        ("energy_threshold", [0.8, 0.85, 0.9, 0.95, 0.99]),
        ("n_levels", [2, 3, 4]),
        ("n_features", [16, 32, 64]),
        ("max_iters", [10, 20, 30]),
        ("n_components", [8, 16, 32, 64]),
    ],
    "spectral": [
        ("keep_fraction", [0.005, 0.01, 0.05, 0.1, 0.2, 0.5]),
        ("keep_ratio", [0.005, 0.01, 0.05, 0.1, 0.2, 0.5]),
        ("target_energy", [0.9, 0.95, 0.99, 0.999]),
        ("threshold", [0.001, 0.005, 0.01, 0.05, 0.1]),
        ("block_size", [8, 16, 32, 64]),
        ("level", [1, 2, 3, 4]),
        ("n_coeffs", [8, 16, 32, 64]),
        ("degree", [4, 8, 12, 16]),
        ("n_components", [16, 32, 64]),
        ("keep_frac", [0.01, 0.05, 0.1, 0.2, 0.5]),
    ],
    "structural": [
        ("sparsity", [0.1, 0.3, 0.5, 0.7, 0.9]),
        ("density", [0.1, 0.3, 0.5, 0.7]),
        ("threshold", [0.0001, 0.001, 0.01, 0.05, 0.1]),
        ("block_size", [16, 32, 64, 128]),
        ("n_blocks", [2, 4, 8]),
        ("prune_ratio", [0.1, 0.3, 0.5, 0.7, 0.9]),
        ("dropout", [0.1, 0.3, 0.5]),
        ("reg", [0.0001, 0.001, 0.01]),
    ],
    "physics": [
        ("rank", [2, 4, 8, 16, 32]),
        ("keep_frac", [0.1, 0.2, 0.3, 0.5]),
        ("keep_fraction", [0.05, 0.1, 0.2, 0.5]),
        ("block_size", [16, 32, 64, 128]),
        ("n_bits", [2, 4, 6, 8]),
        ("n_components", [4, 8, 16, 32]),
        ("threshold", [0.001, 0.01, 0.05, 0.1]),
        ("bond_dim", [4, 8, 16]),
        ("temperature", [0.1, 0.5, 1.0, 2.0]),
    ],
    "novel": [
        ("rank", [2, 4, 8, 16, 32]),
        ("bond_dim", [4, 8, 16]),
        ("block_size", [16, 32, 64]),
        ("keep_fraction", [0.01, 0.05, 0.1, 0.2]),
        ("keep_frac", [0.05, 0.1, 0.2, 0.5]),
        ("threshold", [0.001, 0.01, 0.05, 0.1]),
        ("n_components", [8, 16, 32]),
        ("n_bits", [4, 6, 8]),
        ("chi", [4, 8, 16]),
        ("n_layers", [2, 3, 4]),
        ("density", [0.1, 0.3, 0.5, 0.7]),
        ("sparsity", [0.1, 0.3, 0.5, 0.7, 0.9]),
    ],
    "tensor_network": [
        ("rank", [4, 8, 16, 32]),
        ("bond_dim", [4, 8, 16]),
        ("chi", [4, 8, 16]),
        ("block_size", [16, 32, 64]),
    ],
    "entropy": [
        ("block_size", [32, 64, 128]),
        ("window_size", [16, 32, 64]),
    ],
    "hybrid": [
        ("block_size", [32, 64]),
        ("rank", [8, 16, 32]),
        ("keep_fraction", [0.05, 0.1, 0.2]),
    ],
    "lossless": [
        ("block_size", [64, 128, 256]),
        ("level", [1, 3, 6, 9]),
    ],
    "quantization": [
        ("block_size", [64, 128]),
        ("n_bits", [4, 8]),
    ],
    "transform_quant": [
        ("block_size", [64, 128]),
        ("n_bits", [4, 8]),
    ],
    "sparsity_quant": [
        ("group_size", [32, 64, 128]),
        ("sparsity", [0.3, 0.5, 0.7]),
    ],
    "delta_quant": [
        ("block_size", [32, 64]),
        ("n_bits", [4, 8]),
    ],
    "functional": [
        ("block_size", [64]),
    ],
}


def _get_first_param(cls: Any) -> str | None:
    """Get the first non-self, non-tensor parameter name from compress()."""
    try:
        sig = inspect.signature(cls.compress)
        for p in sig.parameters:
            if p not in ("self", "tensor", "args", "kwargs", "kw"):
                return p
    except (TypeError, ValueError, AttributeError):
        pass
    return None


def _infer_values(param) -> list:
    """Infer sensible variant values from a parameter's default value."""
    if param.default is inspect.Parameter.empty:
        return [4, 8, 16, 32, 64]
    d = param.default
    if isinstance(d, bool):
        return []
    if isinstance(d, int):
        if d <= 0:
            return []
        return [max(1, d // 2), d, d * 2]
    if isinstance(d, float):
        if d <= 0:
            return [0.001, 0.01, 0.1]
        return [d / 10, d / 2, d, d * 2]
    return [0.01, 0.1, 0.5]


# Engine method names to skip during category pass (already have detailed grids)
_SKIP_NAMES: set = {
    "block_int8",
    "block_int4",
    "hadamard_int8",
    "hadamard_int4",
    "sparsity_int4",
    "delta_int4",
    "svd_compress",
    "dct_spectral",
    "tensor_train",
    "fwht_compress",
}


# ── Public API ─────────────────────────────────────────────────────────────


def _apply_grid(
    variants: Dict[str, MethodVariant],
    cls: Any,
    name: str,
    cat: str,
    method_count: int,
    grid: list,
) -> int:
    """Apply a param grid to a method, generating variants. Returns new count."""
    for param_name, values in grid:
        if method_count >= _MAX_VARIANTS_PER_METHOD:
            break
        if not _supports(cls, param_name):
            continue
        for val in values:
            if method_count >= _MAX_VARIANTS_PER_METHOD:
                break
            val_str = _norm(str(val))
            vname = _norm(f"{name}_{param_name}_{val_str}")
            if vname not in variants:
                variants[vname] = MethodVariant(
                    cls, vname, category=cat, **{param_name: val}
                )
                method_count += 1
    return method_count


def get_method_variants(
    method_classes: Dict[str, Any],
) -> Dict[str, MethodVariant]:
    """Generate all parameterized method variants.

    Strategy:
    1. Engine methods get detailed multi-parameter grids (70 variants)
    2. All methods get variants from UNIVERSAL_PARAM_GRID
    3. Category-specific grids provide targeted variants
    4. Catch-all fallback creates variants for methods with unrecognized params
    5. Max 5 variants per method to keep the set manageable

    Args:
        method_classes: Dict of method classes to generate variants for.
            Must be passed explicitly to avoid circular imports.

    Returns:
        Dict[str, MethodVariant]: All generated variants keyed by name.
    """
    variants: Dict[str, MethodVariant] = {}

    # Engine method variants with detailed parameter grids
    variants.update(_engine_variants())

    # Category-based variants — iterate all registered methods
    for name, cls in method_classes.items():
        if name in _SKIP_NAMES:
            continue

        cat = getattr(cls, "category", None)
        if cat is None:
            continue

        method_count = 0

        # Stage 1: Universal param grid (applies to ALL methods)
        method_count = _apply_grid(
            variants, cls, name, cat, method_count, UNIVERSAL_PARAM_GRID
        )

        # Stage 2: Category-specific grid
        cat_grid = CATEGORY_PARAM_GRIDS.get(cat)
        if cat_grid is not None and method_count < _MAX_VARIANTS_PER_METHOD:
            method_count = _apply_grid(variants, cls, name, cat, method_count, cat_grid)

        # Stage 3: Catch-all — if method has params that none of our grids
        # recognized, create variants from its first parameter.
        if method_count > 0:
            continue

        first_param = _get_first_param(cls)
        if first_param is None:
            continue

        try:
            sig = inspect.signature(cls.compress)
            p_obj = sig.parameters.get(first_param)
            if p_obj is None:
                continue
            fallback_values = _infer_values(p_obj)
            for val in fallback_values:
                if method_count >= _MAX_VARIANTS_PER_METHOD:
                    break
                val_str = _norm(str(val))
                vname = _norm(f"{name}_{first_param}_{val_str}")
                if vname not in variants:
                    variants[vname] = MethodVariant(
                        cls, vname, category=cat, **{first_param: val}
                    )
                    method_count += 1
        except (TypeError, ValueError, AttributeError):
            pass

    return variants


# Count accessor (lazy, callable after full import)
def get_method_variant_count(
    method_classes: Dict[str, Any],
) -> int:
    """Return the total number of generated method variants.

    Args:
        method_classes: Dict of method classes to generate variants for.
            Must be passed explicitly to avoid circular imports.

    Returns:
        int: Total number of generated method variants.
    """
    try:
        return len(get_method_variants(method_classes))
    except Exception:
        return 0
