import sys

sys.path.insert(0, ".")
try:
    import numpy as np
    from spectralstream.kv_cache.core import KVCacheEntry
    from spectralstream.kv_cache.eviction import (
        EvictionPolicy,
        SpectralEviction,
        H2OEviction,
        SlidingWindowEviction,
        StreamingLLMEviction,
        EntropyEviction,
        ImportanceScoring,
        PredictiveEviction,
        StalenessAwareEviction,
        AccessPatternEviction,
        ResonanceEviction,
        EntropyGradient,
        ReinforcementLearningEviction,
        HybridEviction,
    )
except ImportError as e:
    print(f"Import error: {e}")
    raise


def _make_entry(position=0, layer_idx=0, score=1.0, dim=256):
    key = np.random.randn(dim).astype(np.float32)
    value = np.random.randn(dim).astype(np.float32)
    return KVCacheEntry(
        key=key, value=value, position=position, layer_idx=layer_idx, score=score
    )


class TestSpectralEviction:
    def test_select_eviction_returns_valid_index(self):
        entries = [_make_entry(i) for i in range(10)]
        policy = SpectralEviction()
        idx = policy.select_eviction(entries)
        assert 0 <= idx < len(entries)

    def test_select_eviction_empty(self):
        policy = SpectralEviction()
        assert policy.select_eviction([]) == -1

    def test_select_eviction_single(self):
        entries = [_make_entry(0)]
        policy = SpectralEviction()
        idx = policy.select_eviction(entries)
        assert idx == 0

    def test_select_eviction_reproducible(self):
        entries = [_make_entry(i) for i in range(10)]
        policy = SpectralEviction()
        results = [policy.select_eviction(entries) for _ in range(3)]
        assert all(0 <= r < len(entries) for r in results)


class TestH2OEviction:
    def test_select_eviction_basic(self):
        entries = [_make_entry(i, score=float(i)) for i in range(10)]
        policy = H2OEviction(heavy_hitter_frac=0.2)
        idx = policy.select_eviction(entries)
        assert 0 <= idx < len(entries)

    def test_select_eviction_empty(self):
        policy = H2OEviction()
        assert policy.select_eviction([]) == -1

    def test_select_eviction_low_score(self):
        entries = [_make_entry(i, score=0.1 if i == 3 else 1.0) for i in range(10)]
        policy = H2OEviction(heavy_hitter_frac=0.5)
        idx = policy.select_eviction(entries)
        assert idx >= 0


class TestSlidingWindowEviction:
    def test_select_oldest(self):
        entries = [_make_entry(i) for i in [5, 10, 3, 8, 1, 12]]
        policy = SlidingWindowEviction(window_size=4)
        idx = policy.select_eviction(entries)
        assert entries[idx].position == 1

    def test_select_eviction_empty(self):
        policy = SlidingWindowEviction()
        assert policy.select_eviction([]) == -1

    def test_select_eviction_sorted(self):
        entries = [_make_entry(i) for i in range(10)]
        policy = SlidingWindowEviction()
        idx = policy.select_eviction(entries)
        assert idx >= 0


class TestStreamingLLMEviction:
    def test_select_eviction_basic(self):
        entries = [_make_entry(i) for i in range(20)]
        policy = StreamingLLMEviction(sink_tokens=4, window_size=10)
        idx = policy.select_eviction(entries)
        assert idx >= 0

    def test_select_eviction_empty(self):
        policy = StreamingLLMEviction()
        assert policy.select_eviction([]) == -1

    def test_select_eviction_all_sink(self):
        entries = [_make_entry(i) for i in range(3)]
        policy = StreamingLLMEviction(sink_tokens=4)
        assert policy.select_eviction(entries) == -1


class TestEntropyEviction:
    def test_select_eviction_basic(self):
        entries = [_make_entry(i) for i in range(5)]
        policy = EntropyEviction()
        idx = policy.select_eviction(entries)
        assert 0 <= idx < len(entries)

    def test_select_eviction_empty(self):
        policy = EntropyEviction()
        assert policy.select_eviction([]) == -1


class TestImportanceScoring:
    def test_select_eviction_basic(self):
        entries = [_make_entry(i, score=float(10 - i)) for i in range(5)]
        policy = ImportanceScoring()
        idx = policy.select_eviction(entries)
        assert 0 <= idx < len(entries)

    def test_select_eviction_empty(self):
        policy = ImportanceScoring()
        assert policy.select_eviction([]) == -1


class TestPredictiveEviction:
    def test_select_eviction_basic(self):
        entries = [_make_entry(i) for i in range(5)]
        policy = PredictiveEviction()
        idx = policy.select_eviction(entries)
        assert 0 <= idx < len(entries)

    def test_select_eviction_empty(self):
        policy = PredictiveEviction()
        assert policy.select_eviction([]) == -1


class TestReinforcementLearningEviction:
    def test_select_eviction_basic(self):
        entries = [_make_entry(i) for i in range(5)]
        policy = ReinforcementLearningEviction(n_arms=4, epsilon=0.0)
        idx = policy.select_eviction(entries)
        assert 0 <= idx < len(entries)

    def test_update(self):
        policy = ReinforcementLearningEviction(n_arms=2)
        entries = [_make_entry(i) for i in range(3)]
        policy.select_eviction(entries)
        policy.update(hit=True)
        policy.update(hit=False)

    def test_select_eviction_empty(self):
        policy = ReinforcementLearningEviction()
        assert policy.select_eviction([]) == -1


class TestHybridEviction:
    def test_select_eviction_basic(self):
        entries = [_make_entry(i) for i in range(5)]
        policy = HybridEviction()
        idx = policy.select_eviction(entries)
        assert 0 <= idx < len(entries)

    def test_select_eviction_empty(self):
        policy = HybridEviction()
        assert policy.select_eviction([]) == -1

    def test_custom_policies(self):
        policies = [SpectralEviction(), H2OEviction()]
        policy = HybridEviction(policies=policies)
        entries = [_make_entry(i) for i in range(5)]
        idx = policy.select_eviction(entries)
        assert 0 <= idx < len(entries)


class TestStalenessAwareEviction:
    def test_select_eviction_basic(self):
        entries = [_make_entry(i) for i in range(5)]
        policy = StalenessAwareEviction()
        idx = policy.select_eviction(entries)
        assert 0 <= idx < len(entries)

    def test_select_eviction_empty(self):
        policy = StalenessAwareEviction()
        assert policy.select_eviction([]) == -1


class TestAccessPatternEviction:
    def test_select_eviction_basic(self):
        entries = [_make_entry(i) for i in range(5)]
        policy = AccessPatternEviction()
        idx = policy.select_eviction(entries)
        assert 0 <= idx < len(entries)

    def test_select_eviction_empty(self):
        policy = AccessPatternEviction()
        assert policy.select_eviction([]) == -1


class TestResonanceEviction:
    def test_select_eviction_basic(self):
        entries = [_make_entry(i) for i in range(5)]
        policy = ResonanceEviction()
        idx = policy.select_eviction(entries)
        assert 0 <= idx < len(entries)

    def test_select_eviction_empty(self):
        policy = ResonanceEviction()
        assert policy.select_eviction([]) == -1


class TestEntropyGradient:
    def test_select_eviction_basic(self):
        entries = [_make_entry(i) for i in range(5)]
        policy = EntropyGradient()
        idx = policy.select_eviction(entries)
        assert 0 <= idx < len(entries)

    def test_select_eviction_empty(self):
        policy = EntropyGradient()
        assert policy.select_eviction([]) == -1
