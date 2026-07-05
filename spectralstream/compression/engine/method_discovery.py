"""MethodDiscovery — automatically finds ALL compression methods from METHOD_CLASSES.

Walks the methods/ package and discovers every class with compress/decompress.
Memory-efficient: uses lazy evaluation and single-tensor validation.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


def _auto_description(method_name: str, category: str) -> str:
    desc_map = {
        "block_int8": "8-bit block-wise quantization",
        "block_int4": "4-bit block-wise quantization",
        "hadamard_int8": "Hadamard transform + 8-bit quantization",
        "hadamard_int4": "Hadamard transform + 4-bit quantization",
        "sparsity_int4": "Sparsity-aware 4-bit quantization",
        "delta_int4": "Delta-encoded 4-bit quantization",
        "svd_compress": "SVD-based low-rank decomposition",
        "dct_spectral": "DCT spectral coefficient compression",
        "tensor_train": "Tensor train decomposition (TT-format)",
        "fwht_compress": "Fast Walsh-Hadamard transform compression",
    }
    if method_name in desc_map:
        return desc_map[method_name]
    name_parts = method_name.replace("_", " ").title()
    return f"{name_parts} ({category})"


class MethodDiscovery:
    """Automatically discovers and validates ALL compression methods."""

    _discovered: Optional[Dict[str, Dict[str, Any]]] = None

    def __init__(self, methods: Optional[Dict[str, Any]] = None) -> None:
        self._methods = methods if methods is not None else {}

    @classmethod
    def discover(cls) -> Dict[str, Dict[str, Any]]:
        """Discover all methods from METHOD_CLASSES in methods/__init__.py.

        Returns mapping: method_name -> {class, category, tier, file, instance, ...}
        """
        if cls._discovered is not None:
            return cls._discovered

        # Lazy imports to avoid circular dependency: method_discovery -> method_tiers
        from ._tier_common import get_tier, tier_score as _tier_score

        methods: Dict[str, Dict[str, Any]] = {}

        try:
            from spectralstream.compression.methods import METHOD_CLASSES
        except ImportError:
            logger.warning("Cannot import METHOD_CLASSES")
            # Fall back to scanning method directories
            return cls._discover_by_walk()

        # Also load engine built-in methods (block_int8, etc.) which may not be in METHOD_CLASSES
        try:
            from spectralstream.compression.engine._methods import (
                METHOD_REGISTRY as ENGINE_METHODS,
            )

            engine_methods = dict(ENGINE_METHODS)
        except ImportError:
            engine_methods = {}

        for method_name, method_cls in METHOD_CLASSES.items():
            try:
                if isinstance(method_cls, type):
                    cat = getattr(method_cls, "category", "quantization")
                    doc = (method_cls.__doc__ or "").strip()
                    # Find source file
                    source_file = ""
                    try:
                        mod = importlib.import_module(method_cls.__module__)
                        source_file = getattr(mod, "__file__", "")
                    except (ImportError, AttributeError, OSError):
                        pass
                    desc = (
                        doc.split("\n")[0]
                        if doc
                        else _auto_description(method_name, cat)
                    )
                else:
                    cat = getattr(method_cls, "category", "quantization")
                    doc = ""
                    source_file = ""
                    desc = _auto_description(method_name, cat)

                tier = get_tier(method_name, cat)

                methods[method_name] = {
                    "class": method_cls,
                    "instance": None,
                    "category": cat,
                    "tier": tier,
                    "tier_score": _tier_score(tier),
                    "file": source_file,
                    "name": method_name,
                    "validated": False,
                    "description": desc,
                }
            except Exception as e:
                logger.debug("Failed to load method '%s': %s", method_name, e)

        # Add engine built-in methods that may not be in METHOD_CLASSES
        for method_name, inst in engine_methods.items():
            if method_name not in methods:
                cat = getattr(inst, "category", "quantization")
                tier = get_tier(method_name, cat)
                doc = (type(inst).__doc__ or "").strip()
                desc = (
                    doc.split("\n")[0] if doc else _auto_description(method_name, cat)
                )
                methods[method_name] = {
                    "class": type(inst),
                    "instance": inst,
                    "category": cat,
                    "tier": tier,
                    "tier_score": _tier_score(tier),
                    "file": "_methods.py",
                    "name": method_name,
                    "validated": False,
                    "description": desc,
                }

        logger.info("Discovered %d methods", len(methods))
        cls._discovered = methods
        return methods

    @classmethod
    def _discover_by_walk(cls) -> Dict[str, Dict[str, Any]]:
        """Fallback: walk method directories to find compress/decompress classes."""
        # Lazy imports to avoid circular dependency: method_discovery -> method_tiers
        from ._tier_common import get_tier, tier_score as _tier_score

        methods: Dict[str, Dict[str, Any]] = {}
        base = Path(__file__).resolve().parent.parent / "methods"
        if not base.exists():
            return methods

        for py_file in sorted(base.rglob("*.py")):
            if py_file.name == "__init__.py":
                continue
            rel = py_file.relative_to(base.parent.parent)
            module_name = str(rel.with_suffix("")).replace("/", ".")
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for attr_name in dir(module):
                cls_obj = getattr(module, attr_name)
                if not (
                    isinstance(cls_obj, type)
                    and hasattr(cls_obj, "compress")
                    and hasattr(cls_obj, "decompress")
                ):
                    continue
                mname = getattr(cls_obj, "name", attr_name.lower().replace("_", ""))
                cat = getattr(cls_obj, "category", "quantization")
                tier = get_tier(mname, cat)
                try:
                    inst = cls_obj()
                except Exception:
                    continue
                methods[mname] = {
                    "class": cls_obj,
                    "instance": inst,
                    "category": cat,
                    "tier": tier,
                    "tier_score": _tier_score(tier),
                    "file": str(py_file),
                    "name": mname,
                    "validated": False,
                }
        logger.info("Walk-discovered %d methods", len(methods))
        return methods

    @classmethod
    def validate_method(
        cls, method_name: str, method_info: Dict[str, Any]
    ) -> Tuple[bool, float, float]:
        """Test a method on a tiny 16x16 tensor (256 bytes — OOM safe)."""
        inst = method_info.get("instance")
        method_cls = method_info.get("class")
        tensor = None
        if inst is None and method_cls is not None:
            try:
                inst = method_cls() if isinstance(method_cls, type) else method_cls
                method_info["instance"] = inst
            except Exception:
                return False, 0.0, 1.0
        if inst is None:
            return False, 0.0, 1.0
        try:
            tensor = np.random.RandomState(42).randn(16, 16).astype(np.float32)
            data, meta = inst.compress(tensor)
            recon = inst.decompress(data, meta)
            if recon.shape != tensor.shape:
                recon = recon.reshape(tensor.shape)
            ratio = max(tensor.nbytes / max(len(data), 1), 1.0)
            err = float(
                np.linalg.norm(recon.ravel() - tensor.ravel())
                / max(np.linalg.norm(tensor.ravel()), 1e-30)
            )
            del tensor, data, recon
            return True, ratio, err
        except Exception:
            if tensor is not None:
                try:
                    del tensor
                except Exception:
                    pass
            return False, 0.0, 1.0

    @classmethod
    def validate_all(cls, batch_size: int = 10) -> Dict[str, Tuple[bool, float, float]]:
        """Validate all methods in memory-safe batches of `batch_size`."""
        import gc

        methods = cls.discover()
        results: Dict[str, Tuple[bool, float, float]] = {}
        items = list(methods.items())
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            for mname, minfo in batch:
                try:
                    works, ratio, err = cls.validate_method(mname, minfo)
                    minfo["validated"] = works
                    minfo["validated_ratio"] = ratio
                    minfo["validated_error"] = err
                    results[mname] = (works, ratio, err)
                except Exception:
                    results[mname] = (False, 0.0, 1.0)
            gc.collect()
        working = sum(1 for v in results.values() if v[0])
        logger.info("Validated %d/%d methods working", working, len(results))
        return results

    @classmethod
    def validate_all_generator(
        cls,
    ) -> Generator[Tuple[str, bool, float, float], None, None]:
        """Validate methods lazily, yielding results one at a time.

        Memory-efficient: only one tensor and one method instance in memory at a time.
        Yields (method_name, works, ratio, error) tuples.
        """
        methods = cls.discover()
        for mname, minfo in methods.items():
            works, ratio, err = cls.validate_method(mname, minfo)
            minfo["validated"] = works
            minfo["validated_ratio"] = ratio
            minfo["validated_error"] = err
            yield mname, works, ratio, err

    @classmethod
    def get_methods_by_tier(cls, tier) -> Dict[str, Dict[str, Any]]:
        """Get all methods belonging to a specific tier."""
        methods = cls.discover()
        return {n: m for n, m in methods.items() if m["tier"] == tier}

    @classmethod
    def get_methods_by_category(cls, category: str) -> Dict[str, Dict[str, Any]]:
        """Get all methods in a specific category."""
        methods = cls.discover()
        return {n: m for n, m in methods.items() if m["category"] == category}

    @classmethod
    def get_compression_methods(cls) -> Dict[str, Dict[str, Any]]:
        """Get only real compression methods (Tiers 1-3, excluding quantization/entropy)."""
        methods = cls.discover()
        return {
            n: m
            for n, m in methods.items()
            if m["tier"] in (1, 2, 3)  # MethodTier enum values
        }

    @classmethod
    def get_quantization_methods(cls) -> Dict[str, Dict[str, Any]]:
        """Get only quantization methods (Tier 5)."""
        methods = cls.discover()
        return {n: m for n, m in methods.items() if m["tier"] == 5}

    @classmethod
    def get_method_stats(cls) -> Dict[str, int]:
        """Get counts of methods by tier and validation status."""
        methods = cls.discover()
        tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        validated = 0
        for m in methods.values():
            tier_counts[m["tier"].value] = tier_counts.get(m["tier"].value, 0) + 1
            if m.get("validated"):
                validated += 1
        return {
            "total": len(methods),
            "validated": validated,
            "tier1_real_compression": tier_counts.get(1, 0),
            "tier2_structural": tier_counts.get(2, 0),
            "tier3_physics": tier_counts.get(3, 0),
            "tier4_entropy": tier_counts.get(4, 0),
            "tier5_quantization": tier_counts.get(5, 0),
        }
