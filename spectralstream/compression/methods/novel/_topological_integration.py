"""Wrap ALL topological_biological and geometric_topological_manifold methods.

Dynamically discovers ~170 methods across two topological packages and wraps
them with a standard (bytes, dict) compress/decompress interface.

Lazy wrappers — no classes instantiated at import time.
get_topological_methods() returns adapter references, not instances.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_BIOLOGICAL_PKG = (
    "spectralstream.compression.methods.novel.topological.topological_biological"
)
_MANIFOLD_PKG = "spectralstream.compression.methods.novel.topological.geometric_topological_manifold"

# Category strings for registration
CATEGORY_BIOLOGICAL = "topological_biological"
CATEGORY_MANIFOLD = "geometric_topological_manifold"


class _TopologicalAdapter:
    """Lazy adapter — wraps a topological class, instantiates on compress/decompress."""

    def __init__(self, name: str, category: str, cls: type) -> None:
        self.name = name
        self.category = category
        self._cls = cls

    def compress(self, tensor: np.ndarray, **kw: Any) -> Tuple[bytes, dict]:
        inst = self._cls()
        return inst.compress(tensor, **kw)

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        inst = self._cls()
        return inst.decompress(data, metadata)


def _discover_classes(
    package_name: str,
    category: str,
) -> Dict[str, Tuple[str, Any]]:
    """Discover compression classes exported by a package's __init__.py.

    Returns dict of name -> (category, adapter) for every class in the
    package that has ``name``, ``compress``, and ``decompress`` attributes.
    """
    result: Dict[str, Tuple[str, Any]] = {}
    try:
        pkg = importlib.import_module(package_name)
    except ImportError:
        logger.warning(
            "Could not import %s — topological methods unavailable", package_name
        )
        return result

    for attr_name in dir(pkg):
        cls = getattr(pkg, attr_name)
        if not isinstance(cls, type):
            continue
        if not hasattr(cls, "compress") or not hasattr(cls, "decompress"):
            continue
        cls_name: str = getattr(cls, "name", None) or attr_name.lower()
        adapter = _TopologicalAdapter(cls_name, category, cls)
        result[cls_name] = (category, adapter)
    return result


# ── Cached discovery results ──────────────────────────────────────────────

_TOPOLOGICAL_DISCOVERED: bool = False
_TOPOLOGICAL_CACHE: Dict[str, Tuple[str, Any]] = {}


def _discover_all() -> Dict[str, Tuple[str, Any]]:
    global _TOPOLOGICAL_DISCOVERED, _TOPOLOGICAL_CACHE
    if _TOPOLOGICAL_DISCOVERED:
        return _TOPOLOGICAL_CACHE

    result: Dict[str, Tuple[str, Any]] = {}

    # 1. Topological biological (36+ neuroscience/math-bio methods)
    bio = _discover_classes(_BIOLOGICAL_PKG, CATEGORY_BIOLOGICAL)
    result.update(bio)
    if bio:
        logger.info("Discovered %d topological biological methods", len(bio))

    # 2. Geometric topological manifold (130+ manifold/Lie group/symmetric space methods)
    manifold = _discover_classes(_MANIFOLD_PKG, CATEGORY_MANIFOLD)
    result.update(manifold)
    if manifold:
        logger.info(
            "Discovered %d geometric topological manifold methods", len(manifold)
        )

    _TOPOLOGICAL_CACHE = result
    _TOPOLOGICAL_DISCOVERED = True
    return result


def get_topological_methods() -> Dict[str, Tuple[str, Any]]:
    """Return dict of name -> (category, lazy adapter) for all topological methods.

    No instantiation or testing — memory-safe for registration.
    """
    return _discover_all()
