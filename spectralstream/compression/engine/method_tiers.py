"""Method Tier Assignment - CATEGORY BASED
=======================================
Every method's tier is determined by its CATEGORY, not manual mapping.
This ensures ALL methods (including new ones) get correct tier assignment.

Tier 1: Decomposition, Spectral, Tensor Network, Functional (score 10)
Tier 2: Structural, Physics (score 5)
Tier 3: Entropy, Lossless (score 2)
Tier 4: Hybrid, Cascade (score 1.5)
Tier 5: Quantization — LAST RESORT (score 0.3)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ._tier_common import (
    DEFAULT_TIER,
    CATEGORY_TIER_MAP,
    MANUAL_TIER_OVERRIDES,
    MethodTier,
    get_method_tier,
    get_tier,
    tier_score,
)


def get_tier_for_method(method_name: str) -> MethodTier:
    """Get the tier for a compression method by its NAME.

    This is the preferred way to get a method's tier when only its name
    is known. It looks up the method's category from METHOD_CLASSES (the
    global registry of all discoverable methods) and then maps the
    category to a tier.

    Resolution order:
    1. Manual overrides (for legacy methods with non-standard categories)
    2. Category-based lookup from METHOD_CLASSES
    3. Engine built-in methods (block_int8, svd_compress, etc.)
    4. Default (Tier 1 — Real Compression)

    Args:
        method_name: The name of the compression method (e.g., 'svd_compress',
            'block_int8', 'butterfly')

    Returns:
        The MethodTier for the method.

    Examples:
        >>> get_tier_for_method('svd_compress')
        <MethodTier.TIER1_REAL_COMPRESSION: 1>
        >>> get_tier_for_method('block_int8')
        <MethodTier.TIER5_QUANTIZATION: 5>
        >>> get_tier_for_method('dct_spectral')
        <MethodTier.TIER1_REAL_COMPRESSION: 1>
    """
    # 1. Check manual overrides first
    if method_name in MANUAL_TIER_OVERRIDES:
        return MANUAL_TIER_OVERRIDES[method_name]

    # 2. Look up from METHOD_CLASSES (global registry)
    try:
        from spectralstream.compression.methods import METHOD_CLASSES

        cls = METHOD_CLASSES.get(method_name)
        if cls is not None:
            category = getattr(cls, "category", "quantization")
            return get_tier(method_name, category)
    except ImportError:
        pass

    # 3. Check engine built-in methods
    try:
        from spectralstream.compression.engine._methods import (
            METHOD_REGISTRY as ENGINE_METHODS,
        )

        inst = ENGINE_METHODS.get(method_name)
        if inst is not None:
            category = getattr(inst, "category", "quantization")
            return get_tier(method_name, category)
    except ImportError:
        pass

    # 4. Fall back to default
    return DEFAULT_TIER


def validate_tier_gap() -> None:
    """Validate the tier gap between Tier 1 and Tier 5 is at least 10x.

    This ensures that compression methods (decomposition, spectral, tensor
    network) are STRICTLY prioritized over quantization (bit pruning).
    The score difference guarantees that the selector will prefer any
    Tier 1-4 method over a Tier 5 quantization method for the same
    expected ratio and error.

    Raises:
        AssertionError: If the gap between Tier 1 and Tier 5 scores
            is less than 10x.
    """
    t1_score = tier_score(MethodTier.TIER1_REAL_COMPRESSION)
    t5_score = tier_score(MethodTier.TIER5_QUANTIZATION)
    gap = t1_score / max(t5_score, 1e-30)
    assert gap >= 10.0, (
        f"Tier gap insufficient: Tier 1 score {t1_score} vs "
        f"Tier 5 score {t5_score} (gap = {gap:.1f}x, "
        f"requires at least 10x). "
        f"Compression methods would NOT be strictly prioritized "
        f"over quantization."
    )


def _build_method_tier_map() -> Dict[str, MethodTier]:
    """Build the full METHOD_TIER_MAP by scanning all discovered methods."""
    try:
        from .method_discovery import MethodDiscovery

        methods = MethodDiscovery.discover()

        tier_map: Dict[str, MethodTier] = {}
        tier_map.update(MANUAL_TIER_OVERRIDES)

        for name, info in methods.items():
            if name not in tier_map:
                cat = info.get("category", "")
                tier = get_method_tier(name, cat)
                tier_map[name] = tier

        return tier_map
    except Exception:
        return dict(MANUAL_TIER_OVERRIDES)


# ── Lazy METHOD_TIER_MAP ─────────────────────────────────────────────────────
# NOTE: This is intentionally lazy to avoid a static import cycle with
# method_discovery.py. At module level, _build_method_tier_map() would trigger
# method_discovery -> method_tiers at import time.
# Instead, we use a module-level __getattr__ (Python 3.7+) to defer population.
_METHOD_TIER_MAP_CACHE: Optional[Dict[str, MethodTier]] = None


def get_method_tier_map() -> Dict[str, MethodTier]:
    """Get the full METHOD_TIER_MAP, built lazily on first access.

    This avoids the import cycle: method_tiers → method_discovery → method_tiers.
    Loading is deferred until first call to avoid module-level circular imports.
    """
    global _METHOD_TIER_MAP_CACHE
    if _METHOD_TIER_MAP_CACHE is None:
        _METHOD_TIER_MAP_CACHE = _build_method_tier_map()
    return _METHOD_TIER_MAP_CACHE


def __getattr__(name: str) -> Dict[str, MethodTier]:
    """Module-level __getattr__ for backward-compatible lazy METHOD_TIER_MAP.

    When 'from method_tiers import METHOD_TIER_MAP' is used, this hook
    intercepts the attribute access and calls get_method_tier_map() lazily.
    This breaks the static import cycle with method_discovery.py.
    """
    if name == "METHOD_TIER_MAP":
        return get_method_tier_map()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def log_tier_distribution() -> None:
    """Log tier distribution for debugging."""
    counts: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for name, tier in get_method_tier_map().items():
        counts[tier] = counts.get(tier, 0) + 1

    logger = __import__("logging").getLogger(__name__)
    logger.info("Method Tier Distribution:")
    for tier in sorted(counts):
        logger.info(f"  Tier {tier}: {counts[tier]}")
