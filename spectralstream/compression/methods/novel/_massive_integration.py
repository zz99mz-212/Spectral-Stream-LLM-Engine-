"""Massive Integration Bridge — auto-discovers and registers ALL unwired compression methods.

This bridge discovers classes with compress() methods from 15+ subdirectories
that were previously unwired from METHOD_CLASSES. Each adapter is lazy — no
classes instantiated at import time.

Discovered directories (total ~1931 methods):
  quantum_engine (27), quantum_inspired (28), unified_physics_quantum_2 (168)
  info_signal (36), information_theory_2 (166)
  revolutionary_gauge (206), revolutionary_topological (213)
  geometric_topological_manifold (160), topological_biological (36)
  chaotic_dynamical (204), fractal_holographic (175), plasma_chaos (32)
  quantization_massive (167), sparsity_transform_delta_massive (319)
  physics/hamiltonian_engine (4)
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, Tuple

import numpy as np


# ── Subdirectory module paths to scan ──────────────────────────────────────

MASSIVE_MODULES: Dict[str, str] = {
    "quantum.quantum_engine": "spectralstream.compression.methods.novel.quantum.quantum_engine",
    "quantum.quantum_inspired": "spectralstream.compression.methods.novel.quantum.quantum_inspired",
    "quantum.unified_physics_quantum_2": "spectralstream.compression.methods.novel.quantum.unified_physics_quantum_2",
    "entropy_info.info_signal": "spectralstream.compression.methods.novel.entropy_info.info_signal",
    "entropy_info.information_theory_2": "spectralstream.compression.methods.novel.entropy_info.information_theory_2",
    "revolutionary.revolutionary_gauge": "spectralstream.compression.methods.novel.revolutionary.revolutionary_gauge",
    "revolutionary.revolutionary_topological": "spectralstream.compression.methods.novel.revolutionary.revolutionary_topological",
    "topological.geometric_topological_manifold": "spectralstream.compression.methods.novel.topological.geometric_topological_manifold",
    "topological.topological_biological": "spectralstream.compression.methods.novel.topological.topological_biological",
    "fractal_chaos.chaotic_dynamical": "spectralstream.compression.methods.novel.fractal_chaos.chaotic_dynamical",
    "fractal_chaos.fractal_holographic": "spectralstream.compression.methods.novel.fractal_chaos.fractal_holographic",
    "fractal_chaos.plasma_chaos": "spectralstream.compression.methods.novel.fractal_chaos.plasma_chaos",
    "quantization_massive": "spectralstream.compression.methods.novel.quantization_massive",
    "structural.sparsity_transform_delta_massive": "spectralstream.compression.methods.novel.structural.sparsity_transform_delta_massive",
    "physics": "spectralstream.compression.methods.novel.physics",
}


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case, preserving acronyms."""
    result: list[str] = []
    for i, c in enumerate(name):
        if c.isupper() and i > 0 and name[i - 1].islower():
            result.append("_")
        elif (
            c.isupper()
            and i > 0
            and name[i - 1].isupper()
            and (i + 1 < len(name) and name[i + 1].islower())
        ):
            result.append("_")
        result.append(c.lower())
    return "".join(result)


def _discover_classes(mod) -> list[type]:
    """Discover all classes with compress() in a module."""
    classes: list[type] = []
    for name in dir(mod):
        obj = getattr(mod, name, None)
        if (
            isinstance(obj, type)
            and hasattr(obj, "compress")
            and inspect.isfunction(obj.compress)
        ):
            if name not in (
                "BlockINT8Wrapper",
                "SymplecticIntegrator",
                "_BlockINT8Wrapper",
            ):
                classes.append(obj)
    return classes


def _build_adapter(
    cls: type, category: str, name_override: str | None = None
) -> Tuple[str, str, Any]:
    """Build a lazy adapter for a class.

    Returns (name, category, adapter_instance).
    """
    name = name_override or _camel_to_snake(cls.__name__)
    category = getattr(cls, "category", category)

    class _LazyAdapter:
        _inst_cache: dict = {}

        def __init__(self, _cls=cls):
            self.name = name
            self.category = category
            self._cls = _cls

        def __call__(self):
            return self

        def _get_inst(self):
            cls_id = id(self._cls)
            if cls_id not in self._inst_cache:
                self._inst_cache[cls_id] = self._cls()
            return self._inst_cache[cls_id]

        def compress(self, tensor: np.ndarray, **kw: Any) -> Tuple[bytes, dict]:
            return self._get_inst().compress(tensor, **kw)

        def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
            return self._get_inst().decompress(data, metadata)

    return name, category, _LazyAdapter()


# ── Category assignment per subdirectory ───────────────────────────────────

SUBDIR_CATEGORIES: Dict[str, str] = {
    "quantum_engine": "quantum",
    "quantum_inspired": "quantum",
    "unified_physics_quantum_2": "quantum",
    "info_signal": "entropy",
    "information_theory_2": "entropy",
    "revolutionary_gauge": "revolutionary_gauge",
    "revolutionary_topological": "revolutionary_topological",
    "geometric_topological_manifold": "topological",
    "topological_biological": "topological",
    "chaotic_dynamical": "fractal_chaos",
    "fractal_holographic": "fractal_chaos",
    "plasma_chaos": "fractal_chaos",
    "quantization_massive": "quantization",
    "sparsity_transform_delta_massive": "structural",
    "physics": "physics",
}


def get_massive_methods() -> Dict[str, Tuple[str, Any]]:
    """Discover and return ALL unwired compression methods from massive subdirectories.

    Returns:
        Dict mapping method_name -> (category, adapter_instance).
    """
    result: Dict[str, Tuple[str, Any]] = {}

    for subdir_key, mod_path in MASSIVE_MODULES.items():
        try:
            mod = importlib.import_module(mod_path)
        except (ImportError, ModuleNotFoundError):
            continue

        classes = _discover_classes(mod)
        category = SUBDIR_CATEGORIES.get(subdir_key.split(".")[-1], "novel")

        for cls in classes:
            name = _camel_to_snake(cls.__name__)
            if name in result:
                name = f"{name}_{subdir_key.split('.')[-1]}"
            if name in result:
                name = f"{name}_{id(cls)}"
            _, cat, adapter = _build_adapter(cls, category, name)
            result[name] = (cat, adapter)

    return result
