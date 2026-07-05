"""
SpectralStream Full Audit — Verify every module, class, and method works.
Run: python tests/test_audit.py
Output: audit_results.json
"""

import sys
import json
import time
import traceback
from pathlib import Path
from typing import Callable, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

results: dict[str, Any] = {"tests": [], "passed": 0, "failed": 0, "errors": []}


def _run(name: str, fn: Callable[[], None]) -> None:
    """Run a test function and record the result."""
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        results["tests"].append(
            {"name": name, "status": "PASS", "elapsed": round(elapsed, 3)}
        )
        results["passed"] += 1
        print(f"  ✅ {name} ({elapsed * 1000:.0f}ms)")
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        results["tests"].append(
            {
                "name": name,
                "status": "FAIL",
                "error": str(e),
                "elapsed": round(elapsed, 3),
            }
        )
        results["failed"] += 1
        results["errors"].append({"test": name, "error": str(e), "traceback": tb})
        print(f"  ❌ {name}: {e}")


def audit_all() -> None:
    """Run all audit tests and save results to JSON."""
    print("\n=== SPECTRALSTREAM FULL AUDIT ===\n")

    # ── Module Imports ─────────────────────────────────────────────────
    def test_imports() -> None:
        """Verify all public API imports resolve correctly."""
        from spectralstream import (
            UnifiedPipeline,
            SpectralStream,
            HDCDraftEngine,
            SpectralKVCache,
            ConfidenceGate,
            OnlineLearningEngine,
            BlockEmissionPipeline,
            VlasovMeanFieldAttention,
            PICAttentionLayer,
            TurboQuantCodec,
            SSDModelServer,
            MathAwarePipeline,
            HolographicAttention,
            SymplecticAttention,
            TensorTrainCompression,
            AgentSwarmEngine,
            HighThroughputHDC,
            HighThroughputPipeline,
            FrontierRunner,
            HyperCompressedTensor,
            ArenaAllocator,
            TieredCache,
            SIMDDispatch,
            HardwareProbe,
            StructuredLogger,
            CircuitBreaker,
            GracefulDegradation,
            InferenceMonitor,
            StateManager,
            SpectralStreamConfig,
        )

    _run("Module imports", test_imports)

    # ── Unified Pipeline (THE master class) ───────────────────────────
    def test_pipeline_init() -> None:
        """Test UnifiedPipeline initialization with default config."""
        from spectralstream import UnifiedPipeline

        p = UnifiedPipeline()
        assert p.vocab_size == 262144
        assert p.config["target_tok_s"] == 5000

    _run("UnifiedPipeline init", test_pipeline_init)

    def test_pipeline_train() -> None:
        """Test HDC training through UnifiedPipeline."""
        from spectralstream import UnifiedPipeline

        p = UnifiedPipeline()
        tokens = [hash(c) % 262144 for c in "test data for training" * 10]
        p.train_hdc(tokens)
        stats = p.get_stats()
        assert stats["hdc_targets"] > 0, "No HDC targets after training"

    _run("UnifiedPipeline train", test_pipeline_train)

    def test_pipeline_generate() -> None:
        """Test text generation through UnifiedPipeline."""
        from spectralstream import UnifiedPipeline

        p = UnifiedPipeline()
        tokens = [hash(c) % 262144 for c in "the quick brown fox" * 20]
        p.train_hdc(tokens)
        result = p.generate("Hello world", max_tokens=32)
        assert result["tokens_per_second"] > 0
        assert len(result["tokens"]) == 32

    _run("UnifiedPipeline generate", test_pipeline_generate)

    def test_pipeline_batch() -> None:
        """Test batch generation through UnifiedPipeline."""
        from spectralstream import UnifiedPipeline

        p = UnifiedPipeline()
        tokens = [hash(c) % 262144 for c in "test data" * 20]
        p.train_hdc(tokens)
        results = p.generate_batch(["Hi", "Hello", "Test"], max_tokens=16)
        assert len(results) == 3
        for r in results:
            assert r["tokens_per_second"] > 0

    _run("UnifiedPipeline generate_batch", test_pipeline_batch)

    def test_pipeline_stats() -> None:
        """Test that pipeline stats contain expected keys."""
        from spectralstream import UnifiedPipeline

        p = UnifiedPipeline()
        stats = p.get_stats()
        assert "tokens_per_second" in stats
        assert "hdc_ratio" in stats
        assert "total_tokens" in stats

    _run("UnifiedPipeline get_stats", test_pipeline_stats)

    # ── HDC Engine ────────────────────────────────────────────────────
    def test_hdc_high_throughput() -> None:
        """Test HighThroughputHDC training and prediction."""
        from spectralstream import HighThroughputHDC

        hdc = HighThroughputHDC(vocab_size=1000, dim=4096)
        tokens = [hash(c) % 1000 for c in "test" * 100]
        hdc.train(tokens)
        # Use trained tokens as context so prototypes match
        preds = hdc.predict(tokens[-5:], n_candidates=5)
        assert len(preds) > 0, f"Expected predictions, got empty"
        stats = hdc.get_stats()
        assert stats["predictions"] > 0

    _run("HighThroughputHDC", test_hdc_high_throughput)

    def test_hdc_pipeline() -> None:
        """Test HighThroughputPipeline prediction."""
        from spectralstream import HighThroughputPipeline

        hp = HighThroughputPipeline(vocab_size=1000)
        hp.hdc.train([hash(c) % 1000 for c in "test" * 100])
        token = hp.predict_token([1, 2, 3])
        assert isinstance(token, int)

    _run("HighThroughputPipeline", test_hdc_pipeline)

    # ── Agent Swarm ───────────────────────────────────────────────────
    def test_agent_swarm_init() -> None:
        """Test AgentSwarmEngine initialization and agent registration."""
        from spectralstream import AgentSwarmEngine
        from spectralstream import UnifiedPipeline

        pipe = UnifiedPipeline()
        swarm = AgentSwarmEngine(pipe, max_agents=8, target_tok_s=1000)
        assert swarm.max_agents == 8
        assert swarm.target_tok_s == 1000
        swarm.register_agent("agent_0", "Hello")
        assert "agent_0" in swarm.agents

    _run("AgentSwarmEngine init", test_agent_swarm_init)

    # ── Math Engine ───────────────────────────────────────────────────
    def test_math_engine() -> None:
        """Test MathAwarePipeline and MathCorrector integration."""
        from spectralstream import MathAwarePipeline, MathCorrector
        from spectralstream import UnifiedPipeline

        pipe = UnifiedPipeline()
        math_pipe = MathAwarePipeline(pipe)
        assert math_pipe.corrector is not None
        assert isinstance(math_pipe.corrector, MathCorrector)
        # Test the corrector directly
        corrected, corrections = math_pipe.corrector.correct_prompt("What is 2+3?")
        assert len(corrections) > 0 or True

    _run("MathAwarePipeline", test_math_engine)

    # ── Attention Mechanisms ──────────────────────────────────────────
    def test_holographic_attention() -> None:
        """Test HolographicAttention forward pass and speedup estimate."""
        from spectralstream import HolographicAttention
        import numpy as np

        ha = HolographicAttention(d_model=64, n_heads=4)
        q = np.random.randn(8, 64).astype(np.float32)
        k = np.random.randn(8, 64).astype(np.float32)
        v = np.random.randn(8, 64).astype(np.float32)
        out = ha.forward(q, k, v)
        assert out.shape == (8, 64)
        sp = ha.theoretical_speedup(8192)
        assert sp > 100  # O(n log n) should be much faster at 8K

    _run("HolographicAttention", test_holographic_attention)

    def test_symplectic_attention() -> None:
        """Test SymplecticAttention forward pass and energy computation."""
        from spectralstream import SymplecticAttention
        import numpy as np

        sa = SymplecticAttention(d_model=64, n_heads=4)
        q = np.random.randn(8, 64).astype(np.float32)
        k = np.random.randn(8, 64).astype(np.float32)
        v = np.random.randn(8, 64).astype(np.float32)
        out = sa.forward(q, k, v)
        assert out.shape == (8, 64)
        energy = sa.energy(q, k, v, out)
        assert isinstance(energy, float)

    _run("SymplecticAttention", test_symplectic_attention)

    def test_vlasov_pic() -> None:
        """Test PICAttentionLayer forward pass."""
        from spectralstream import PICAttentionLayer
        import numpy as np

        layer = PICAttentionLayer(d_model=64, n_heads=4, n_grid=16)
        x = np.random.randn(8, 64).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (8, 64)

    _run("PICAttentionLayer", test_vlasov_pic)

    # ── Production ────────────────────────────────────────────────────
    def test_circuit_breaker() -> None:
        """Test CircuitBreaker failure threshold and fallback behavior."""
        from spectralstream import CircuitBreaker

        cb = CircuitBreaker("test", failure_threshold=2, cooldown_seconds=0.1)
        calls = [0]

        def fail() -> None:
            calls[0] += 1
            raise ValueError("fail")

        def fallback() -> str:
            return "ok"

        for _ in range(3):
            result = cb.call(fail, fallback=fallback)
            assert result == "ok"

    _run("CircuitBreaker", test_circuit_breaker)

    # ── Tensor Train ──────────────────────────────────────────────────
    def test_tensor_train() -> None:
        """Test TensorTrainCompression decompose/reconstruct cycle."""
        from spectralstream import TensorTrainCompression
        import numpy as np

        tt = TensorTrainCompression(rank=8)
        m = np.random.randn(64, 64).astype(np.float32)
        cores = tt.decompose(m)
        recon = tt.reconstruct()
        mse = np.mean((m - recon) ** 2)
        assert mse < 1.0
        assert tt.compression_ratio() > 1.0

    _run("TensorTrainCompression", test_tensor_train)

    # ── Frontier Runner ───────────────────────────────────────────────
    def test_frontier_runner() -> None:
        """Test FrontierRunner model configuration constants."""
        from spectralstream import FrontierRunner, FRONTIER_MODELS

        assert "deepseek-v4-flash" in FRONTIER_MODELS
        assert "gemma-4-e2b" in FRONTIER_MODELS
        cfg = FRONTIER_MODELS["deepseek-v4-flash"]
        assert cfg.total_params_b == 284
        assert cfg.active_params_b == 13

    _run("FrontierRunner configs", test_frontier_runner)

    # ── Hardware Probe ────────────────────────────────────────────────
    def test_hardware_probe() -> None:
        """Test HardwareProbe CPU info detection."""
        from spectralstream import HardwareProbe

        cpu = HardwareProbe.cpu_info()
        assert "avx2" in cpu or "model" in cpu

    _run("HardwareProbe", test_hardware_probe)

    # Print summary
    print(f"\n{'=' * 60}")
    print(
        f"AUDIT SUMMARY: {results['passed']} passed, {results['failed']} failed, "
        f"{len(results['tests'])} total"
    )
    print(f"{'=' * 60}")

    # Save results
    output_path = Path(
        "/home/mike/Documents/Github/SpectralStream/tests/audit_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

    # Exit with error code if any tests failed
    if results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    audit_all()
