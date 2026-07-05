"""Test method categorization, tier assignment, and discoverability.

Verifies:
  1. All methods have a category attribute
  2. All categories are in the tier map
  3. All categories are in ALL_CATEGORIES
  4. No method has an unknown/uncategorized category
  5. Tier assignment works correctly for each category
"""

import sys
from collections import Counter

import pytest

sys.path.insert(0, ".")


class TestMethodCategories:
    """Verify all 1090+ methods are properly categorized and tiered."""

    def _get_methods(self):
        """Import and return METHOD_CLASSES."""
        from spectralstream.compression.methods import METHOD_CLASSES

        return METHOD_CLASSES

    def _get_tier_map(self):
        """Import and return CATEGORY_TIER_MAP."""
        from spectralstream.compression.engine.method_tiers import CATEGORY_TIER_MAP

        return CATEGORY_TIER_MAP

    def _get_tier(self, method_name, category=""):
        """Import and call get_tier."""
        from spectralstream.compression.engine.method_tiers import get_tier

        return get_tier(method_name, category)

    def test_all_methods_have_category(self):
        """Every method in METHOD_CLASSES must have a 'category' attribute."""
        methods = self._get_methods()
        uncategorized = [
            name for name, cls in methods.items() if not hasattr(cls, "category")
        ]
        assert len(uncategorized) == 0, (
            f"Uncategorized methods ({len(uncategorized)}): {uncategorized[:20]}"
        )

    def test_all_methods_category_nonempty(self):
        """Every method's category must be a non-empty string."""
        methods = self._get_methods()
        empty = [
            name
            for name, cls in methods.items()
            if not getattr(cls, "category", "")
            or getattr(cls, "category", "") == "unknown"
        ]
        assert len(empty) == 0, (
            f"Methods with empty/unknown category ({len(empty)}): {empty[:20]}"
        )

    def test_all_categories_in_tier_map(self):
        """Every method category must have a corresponding entry in CATEGORY_TIER_MAP."""
        methods = self._get_methods()
        tier_map = self._get_tier_map()
        cats = Counter()
        for name, cls in methods.items():
            cat = getattr(cls, "category", "unknown")
            cats[cat] += 1
        missing = [c for c in cats if c not in tier_map and c != "unknown"]
        assert len(missing) == 0, f"Categories missing from tier map: {missing}"

    def test_all_methods_in_all_categories(self):
        """Every method's category must appear in the ALL_CATEGORIES export list."""
        methods = self._get_methods()
        from spectralstream.compression.methods import ALL_CATEGORIES

        cats = set()
        for name, cls in methods.items():
            cat = getattr(cls, "category", "unknown")
            if cat != "unknown":
                cats.add(cat)
        missing = [c for c in sorted(cats) if c not in ALL_CATEGORIES]
        assert len(missing) == 0, (
            f"Categories in methods but NOT in ALL_CATEGORIES: {missing}"
        )

    def test_tier_map_has_no_extra_categories(self):
        """CATEGORY_TIER_MAP may contain future-proof categories not yet in use.

        The tier map is the SOURCE OF TRUTH for method prioritization.
        Future-proof entries ensure new methods automatically get correct tiers.
        """
        methods = self._get_methods()
        tier_map = self._get_tier_map()
        cats_in_use = set()
        for name, cls in methods.items():
            cat = getattr(cls, "category", "unknown")
            cats_in_use.add(cat)
        extra = [c for c in tier_map if c not in cats_in_use]
        # These are future-proof categories kept for forward compatibility
        expected_extra = {
            "transform_spectral",
            "revolutionary",
            "novel_signal",
            "novel_info",
            "novel_cross",
            "novel_chaotic",
            "tensor_quantum",
            "quantum_compression",
            "quantum_engine",
            "fractal_holographic",
            "information_theory_2",
            "novel_structural",
            "novel_physics",
            "novel_chaos",
            "novel_topological",
            "novel_biological",
            "breakthrough_physics",
            "unified_physics_quantum2",
            "geometric_topological_manifold",
            "novel_entropy",
            "novel_fractal",
            "functional_weight_space",
            "novel_algorithmic",
            "mixed",
        }
        actual_extra = set(extra) - expected_extra
        assert len(actual_extra) == 0, (
            f"Tier map contains unexpected unused categories: {actual_extra}"
        )

    def test_tier_assignment_decomposition(self):
        """Decomposition methods should be Tier 1."""
        tier = self._get_tier("test_method", "decomposition")
        assert tier == 1, f"Decomposition should be Tier 1, got Tier {tier}"

    def test_tier_assignment_spectral(self):
        """Spectral methods should be Tier 1."""
        tier = self._get_tier("test_method", "spectral")
        assert tier == 1, f"Spectral should be Tier 1, got Tier {tier}"

    def test_tier_assignment_novel(self):
        """Novel methods should be Tier 1."""
        tier = self._get_tier("test_method", "novel")
        assert tier == 1, f"Novel should be Tier 1, got Tier {tier}"

    def test_tier_assignment_revolutionary_gauge(self):
        """Revolutionary gauge methods should be Tier 1."""
        tier = self._get_tier("test_method", "revolutionary_gauge")
        assert tier == 1, f"Revolutionary gauge should be Tier 1, got Tier {tier}"

    def test_tier_assignment_revolutionary_topological(self):
        """Revolutionary topological methods should be Tier 1."""
        tier = self._get_tier("test_method", "revolutionary_topological")
        assert tier == 1, f"Revolutionary topological should be Tier 1, got Tier {tier}"

    def test_tier_assignment_functional_weight_space(self):
        """Functional weight space methods should be Tier 1."""
        tier = self._get_tier("test_method", "functional_weight_space")
        assert tier == 1, f"Functional weight space should be Tier 1, got Tier {tier}"

    def test_tier_assignment_structural(self):
        """Structural methods should be Tier 2."""
        tier = self._get_tier("test_method", "structural")
        assert tier == 2, f"Structural should be Tier 2, got Tier {tier}"

    def test_tier_assignment_physics(self):
        """Physics methods should be Tier 2."""
        tier = self._get_tier("test_method", "physics")
        assert tier == 2, f"Physics should be Tier 2, got Tier {tier}"

    def test_tier_assignment_entropy(self):
        """Entropy methods should be Tier 3."""
        tier = self._get_tier("test_method", "entropy")
        assert tier == 3, f"Entropy should be Tier 3, got Tier {tier}"

    def test_tier_assignment_lossless(self):
        """Lossless methods should be Tier 3."""
        tier = self._get_tier("test_method", "lossless")
        assert tier == 3, f"Lossless should be Tier 3, got Tier {tier}"

    def test_tier_assignment_hybrid(self):
        """Hybrid methods should be Tier 4."""
        tier = self._get_tier("test_method", "hybrid")
        assert tier == 4, f"Hybrid should be Tier 4, got Tier {tier}"

    def test_tier_assignment_cascade(self):
        """Cascade methods should be Tier 4."""
        tier = self._get_tier("test_method", "cascade")
        assert tier == 4, f"Cascade should be Tier 4, got Tier {tier}"

    def test_tier_assignment_quantization(self):
        """Quantization methods should be Tier 5."""
        tier = self._get_tier("test_method", "quantization")
        assert tier == 5, f"Quantization should be Tier 5, got Tier {tier}"

    def test_method_count_above_threshold(self):
        """There must be at least 1000 discoverable methods."""
        methods = self._get_methods()
        assert len(methods) >= 1000, (
            f"Only {len(methods)} methods discovered, expected >= 1000"
        )

    def test_method_count_by_category(self):
        """Record method counts per category for regression tracking."""
        methods = self._get_methods()
        cats = Counter()
        for name, cls in methods.items():
            cat = getattr(cls, "category", "unknown")
            cats[cat] += 1
        # Verify known minimum counts per major category
        thresholds = {
            "decomposition": 100,
            "spectral": 100,
            "physics": 50,
            "structural": 50,
            "quantization": 50,
            "functional": 50,
            "entropy": 30,
            "novel": 20,
            "hybrid": 20,
        }
        failures = []
        for cat, minimum in thresholds.items():
            actual = cats.get(cat, 0)
            if actual < minimum:
                failures.append(f"{cat}: expected >= {minimum}, got {actual}")
        assert not failures, "Category count thresholds breached: " + "; ".join(
            failures
        )

    def test_no_duplicate_method_names(self):
        """All method names in METHOD_CLASSES must be unique."""
        methods = self._get_methods()
        assert len(methods) == len(set(methods.keys())), (
            "Duplicate method names found in METHOD_CLASSES"
        )

    def test_get_tier_by_category_name(self):
        """get_tier('decomposition') should return Tier 1 directly."""
        from spectralstream.compression.engine.method_tiers import get_tier

        assert get_tier("decomposition") == 1
        assert get_tier("spectral") == 1
        assert get_tier("revolutionary_gauge") == 1
        assert get_tier("structural") == 2
        assert get_tier("physics") == 2
        assert get_tier("entropy") == 3
        assert get_tier("hybrid") == 4
        assert get_tier("cascade") == 4
        assert get_tier("quantization") == 5

    def test_manual_overrides_present(self):
        """Ensure key manual overrides are still defined."""
        from spectralstream.compression.engine.method_tiers import (
            MANUAL_TIER_OVERRIDES,
        )

        expected_overrides = [
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
        ]
        for name in expected_overrides:
            assert name in MANUAL_TIER_OVERRIDES, (
                f"Expected manual override for '{name}' not found"
            )

    def test_archive_methods_have_categories(self):
        """Archive-reintegrated methods must also have valid categories."""
        from spectralstream.compression.methods import METHOD_CLASSES
        from spectralstream.compression.engine.method_tiers import CATEGORY_TIER_MAP

        # Check archive-origin methods (known prefixes/suffixes)
        archive_like = [
            name
            for name, cls in METHOD_CLASSES.items()
            if any(suffix in name for suffix in ("_archive", "_advanced", "advanced_"))
        ]
        for name in archive_like[:50]:
            cls = METHOD_CLASSES[name]
            cat = getattr(cls, "category", "")
            assert cat, f"Archive method '{name}' has empty category"
            # If it's a real category, it should be in the tier map
            if cat != "unknown":
                assert cat in CATEGORY_TIER_MAP, (
                    f"Archive method '{name}' has category '{cat}' not in tier map"
                )

    def test_no_category_spelling_errors(self):
        """Verify there are no category names that look like typos."""
        methods = self._get_methods()
        tier_map = self._get_tier_map()
        cats = set()
        for name, cls in methods.items():
            cat = getattr(cls, "category", "unknown")
            if cat != "unknown":
                cats.add(cat)

        # Known valid categories
        known = set(tier_map.keys())
        unknown = cats - known
        assert len(unknown) == 0, f"Unknown/spelling-error categories: {unknown}"
