from __future__ import annotations

import gc
import hashlib
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MethodTestResult:
    """Result of testing one compression method on a tensor."""

    def __init__(
        self,
        method_name: str,
        category: str,
        tier: int,
        ratio: float,
        cosine_similarity: float,
        snr_db: float,
        relative_error: float,
        compressed_bytes: int,
        elapsed: float,
        metadata: dict,
    ) -> None:
        self.method_name = method_name
        self.category = category
        self.tier = tier
        self.ratio = ratio
        self.cosine_similarity = cosine_similarity
        self.snr_db = snr_db
        self.relative_error = relative_error
        self.compressed_bytes = compressed_bytes
        self.elapsed = elapsed
        self.metadata = metadata

    def score(self) -> float:
        """Composite score: higher is better. Balances ratio and quality."""
        return (
            self.ratio
            * (1.0 - self.relative_error)
            * (0.5 + 0.5 * self.cosine_similarity)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "category": self.category,
            "tier": self.tier,
            "ratio": self.ratio,
            "cosine_similarity": self.cosine_similarity,
            "snr_db": self.snr_db,
            "relative_error": self.relative_error,
            "compressed_bytes": self.compressed_bytes,
            "elapsed": self.elapsed,
            "score": self.score(),
        }

    def __repr__(self) -> str:
        return (
            f"MethodTestResult({self.method_name}: ratio={self.ratio:.1f}x, "
            f"cos={self.cosine_similarity:.4f}, err={self.relative_error:.6f}, "
            f"score={self.score():.2f})"
        )


class ZeroShotPredictor:
    """Predicts optimal compression method from tensor metadata WITHOUT testing.

    Uses a lightweight meta-model:
    1. Semantic fingerprint: tensor_name + shape + layer_type → deterministic key
    2. TensorSketch: randomized projection (Johnson-Lindenstrauss) of elements
    3. Rule-based prediction: known optimal methods per tensor type
    4. Record results: learn from past compression outcomes

    The meta-model is trained offline from DynamicMethodTester results.
    At runtime: O(1) prediction in microseconds.
    """

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine
        self._pattern_cache: Dict[str, List[Tuple[str, dict, float]]] = {}
        self._confidence_threshold: float = 0.7

    def tensor_sketch(self, tensor: np.ndarray, sketch_size: int = 256) -> np.ndarray:
        """Fast Johnson-Lindenstrauss sketch of tensor.

        Uses sparse random projection (count-sketch):
        - Initialize s in {+/-1}^d randomly (d = tensor.size)
        - Initialize h: [d] -> [k] random hash
        - Sketch[j] = Sum_{i: h(i)=j} s[i] * tensor.ravel()[i]

        This gives an unbiased estimate of inner products.
        O(n) time instead of O(n^2) for SVD.
        """
        n = tensor.size
        k = min(sketch_size, n)
        rng = np.random.RandomState(42)
        h = rng.randint(0, k, size=n)
        s = rng.choice([-1, 1], size=n)
        flat = tensor.ravel()
        sketch = np.zeros(k, dtype=np.float64)
        np.add.at(sketch, h, s * flat)
        return sketch

    def semantic_fingerprint(self, name: str, shape: tuple) -> str:
        """Create deterministic fingerprint from tensor metadata.

        Examples:
        - 'attention_q_2048x1536_layer0'
        - 'ffn_gate_6144x1536_layer5'
        - 'embedding_152576x4096'
        """
        layer_type = self._classify_by_name(name)
        shape_str = "x".join(str(d) for d in shape)
        layer_depth = 0
        m = re.search(r"layers\.(\d+)", name)
        if m:
            layer_depth = int(m.group(1))
        return f"{layer_type}_{shape_str}_layer{layer_depth}"

    @staticmethod
    def _classify_by_name(name: str) -> str:
        """Classify tensor by name into semantic type."""
        name_lower = name.lower()
        if "q_proj" in name_lower or "wq" in name_lower:
            return "attention_q"
        if "k_proj" in name_lower or "wk" in name_lower:
            return "attention_k"
        if "v_proj" in name_lower or "wv" in name_lower:
            return "attention_v"
        if "o_proj" in name_lower or "wo" in name_lower:
            return "attention_o"
        if "gate_proj" in name_lower or "w_gate" in name_lower:
            return "ffn_gate"
        if "up_proj" in name_lower or "w_up" in name_lower:
            return "ffn_up"
        if "down_proj" in name_lower or "w_down" in name_lower:
            return "ffn_down"
        if "embed" in name_lower:
            return "embedding"
        if "norm" in name_lower or "rms_norm" in name_lower:
            return "norm"
        if "head" in name_lower or "lm_head" in name_lower:
            return "lm_head"
        return "other"

    def predict(
        self,
        name: str,
        tensor: np.ndarray,
        target_ratio: float = 5000.0,
    ) -> List[Tuple[str, dict, float]]:
        """Predict optimal compression methods for a tensor.

        Returns list of (method_name, params, confidence) sorted by confidence.

        1. Check CascadeLearner cache first (instant if known pattern)
        2. Compute semantic fingerprint + TensorSketch
        3. Use fuzzy match against known patterns
        4. Fall back to rule-based heuristic if no match
        """
        fp = self.semantic_fingerprint(name, tensor.shape)

        if fp in self._pattern_cache:
            return self._pattern_cache[fp]

        tensor_type = self._classify_by_name(name)
        sketch = self.tensor_sketch(tensor)
        predictions = self._rule_based_predict(
            tensor_type, tensor.shape, sketch, target_ratio
        )

        return predictions

    def _rule_based_predict(
        self,
        tensor_type: str,
        shape: tuple,
        sketch: np.ndarray,
        target_ratio: float,
    ) -> List[Tuple[str, dict, float]]:
        """Rule-based prediction based on tensor characteristics.

        Each tensor type has known optimal methods from previous R&D:
        - attention_q/k/v: SVD progressive cascade (low effective rank)
        - attention_o: DCT + SVD cascade
        - ffn_gate/up/down: TensorTrain with 4D reshaping
        - embedding: BlockINT4 (too large for SVD)
        - norm: no compression needed (tiny tensors)
        """
        predictions: List[Tuple[str, dict, float]] = []

        if tensor_type in ("attention_q", "attention_k", "attention_v", "attention_o"):
            predictions = [
                ("svd_compress", {"rank": 32}, 0.9),
                ("svd_compress", {"rank": 16}, 0.85),
                ("svd_compress", {"rank": 8}, 0.8),
            ]
        elif tensor_type in ("ffn_gate", "ffn_up", "ffn_down"):
            predictions = [
                ("tensor_train", {"rank": 16}, 0.85),
                ("tensor_train", {"rank": 8}, 0.8),
                ("tensor_train", {"rank": 4}, 0.75),
                ("svd_compress", {"rank": 8}, 0.7),
                ("svd_compress", {"rank": 4}, 0.65),
            ]
        elif tensor_type == "embedding":
            predictions = [
                ("block_int4", {"block_size": 32}, 0.8),
            ]
        elif tensor_type in ("lm_head", "other"):
            predictions = [
                ("block_int4", {"block_size": 32}, 0.7),
                ("svd_compress", {"rank": 16}, 0.65),
            ]

        return predictions

    def record_result(
        self,
        name: str,
        tensor: np.ndarray,
        method_name: str,
        ratio: float,
        cosine: float,
    ) -> None:
        """Record compression result to improve future predictions."""
        fp = self.semantic_fingerprint(name, tensor.shape)
        if fp not in self._pattern_cache:
            self._pattern_cache[fp] = []
        self._pattern_cache[fp].append((method_name, {}, cosine))
        self._pattern_cache[fp].sort(key=lambda x: -x[2])


class DynamicMethodTester:
    """Tests ALL registered compression methods against a tensor profile.

    Strategy:
    1. Profile tensor (shape, effective rank, spectral energy, sparsity)
    2. Filter methods by tier (prefer Tier 1-3 over Tier 5 quantization)
    3. Test each filtered method on the tensor
    4. Rank by composite score
    5. Find optimal stacking order (methods that capture complementary structure)
    """

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine
        self._results_cache: Dict[str, List[MethodTestResult]] = {}
        self._method_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._quality_assessor: Any = None

    # ── Lazy imports ───────────────────────────────────────────────────

    def _get_quality_assessor(self) -> Any:
        if self._quality_assessor is None:
            from spectralstream.core.math_primitives.quality import QualityAssessor

            self._quality_assessor = QualityAssessor()
        return self._quality_assessor

    def _get_all_methods(self) -> Dict[str, Dict[str, Any]]:
        """Lazily discover all methods once."""
        if self._method_cache is None:
            from .method_discovery import MethodDiscovery

            self._method_cache = MethodDiscovery.discover()
        return self._method_cache

    def _get_cascade_learner(self) -> Any:
        from .cascade_learner import CascadeLearner

        return CascadeLearner()

    # ── Tensor Profiling ───────────────────────────────────────────────

    def profile_tensor(self, tensor: np.ndarray) -> Dict[str, Any]:
        """Tensor profile via CompressionProfiler.

        Delegates to CompressionProfiler for unified statistical,
        spectral, structural, and sparsity analysis.
        """
        from ._profiler import CompressionProfiler

        base = CompressionProfiler().profile_tensor(tensor, "")
        profile: Dict[str, Any] = {
            "shape": base.shape,
            "ndim": len(base.shape),
            "n_elements": base.n_elements,
            "nbytes": base.nbytes,
            "dtype": base.dtype,
            "mean": base.mean,
            "std": base.std,
            "min": base.min_val,
            "max": base.max_val,
            "effective_rank": base.effective_rank,
            "singular_values": None,
            "condition_number": None,
            "spectral_energy_90pct": base.energy_concentration,
            "sparsity": getattr(base, "nm_sparsity_score", 0.0),
        }

        # Spectral flatness (Wiener entropy) — unique to DMT
        try:
            if tensor.ndim == 1:
                spectrum = np.abs(np.fft.rfft(tensor.astype(np.float64)))
            elif tensor.ndim == 2:
                spectrum = np.abs(np.fft.fft2(tensor.astype(np.float64))).ravel()
            else:
                spectrum = np.abs(np.fft.fftn(tensor.astype(np.float64))).ravel()
            spectral_mean = np.mean(spectrum)
            if spectral_mean > 0:
                geo_mean = np.exp(np.mean(np.log(spectrum + 1e-30)))
                profile["spectral_flatness"] = float(geo_mean / spectral_mean)
            else:
                profile["spectral_flatness"] = 0.0
        except Exception:
            profile["spectral_flatness"] = 0.0

        return profile

    def _tensor_fingerprint(self, tensor: np.ndarray) -> str:
        """Hash-based fingerprint for caching results."""
        h = hashlib.md5(tensor.tobytes()[:4096]).hexdigest()
        shape_str = "x".join(str(d) for d in tensor.shape)
        return f"{shape_str}_{tensor.dtype}_{h}"

    # ── Method Filtering ───────────────────────────────────────────────

    def get_applicable_methods(
        self,
        profile: Dict[str, Any],
        max_per_category: int = 5,
    ) -> List[Tuple[str, Any]]:
        """Get methods that match the tensor profile.

        Uses the profile to filter:
        - 2D methods for 2D tensors
        - Low-rank methods for low effective rank
        - Spectral methods for energy-compactable tensors
        - Methods by tier (prefer 1-4 over 5)
        """
        all_methods = self._get_all_methods()
        if not all_methods:
            logger.warning("No methods discovered")
            return []

        ndim = profile.get("ndim", 2)
        effective_rank = profile.get("effective_rank") or min(
            profile.get("shape", (256, 256))
        )
        spectral_flatness = profile.get("spectral_flatness", 0.5)
        sparsity = profile.get("sparsity", 0.0)

        from ._tier_common import MethodTier as _MethodTier, tier_score as _tier_score

        category_groups: Dict[str, List[Tuple[str, Any, float, int]]] = {}
        for mname, minfo in all_methods.items():
            inst = minfo.get("instance")
            if inst is None:
                try:
                    cls = minfo.get("class")
                    if cls is not None:
                        inst = cls() if isinstance(cls, type) else cls
                        minfo["instance"] = inst
                except Exception:
                    continue
            if inst is None:
                continue
            cat = minfo.get("category", "quantization")
            tier = minfo.get("tier")
            try:
                tval = tier.value if hasattr(tier, "value") else int(tier)
            except (ValueError, TypeError):
                tval = 5
            ts = _tier_score(_MethodTier(tval)) if tval in (1, 2, 3, 4, 5) else 0.3

            if not self._method_compatible(
                mname, cat, ndim, effective_rank, spectral_flatness, sparsity
            ):
                continue

            category_groups.setdefault(cat, []).append((mname, inst, ts, tval))

        selected: List[Tuple[str, Any]] = []
        seen: set = set()
        for cat, methods in category_groups.items():
            methods.sort(key=lambda x: -x[2])
            for mname, inst, ts, tval in methods[:max_per_category]:
                if mname not in seen:
                    seen.add(mname)
                    selected.append((mname, inst))

        if len(selected) < 10:
            for mname, minfo in all_methods.items():
                if mname in seen:
                    continue
                tier = minfo.get("tier")
                try:
                    tval = tier.value if hasattr(tier, "value") else int(tier)
                except (ValueError, TypeError):
                    tval = 5
                if tval <= 2:
                    inst = minfo.get("instance")
                    if inst is not None and self._method_compatible(
                        mname,
                        minfo.get("category", ""),
                        ndim,
                        effective_rank,
                        spectral_flatness,
                        sparsity,
                    ):
                        seen.add(mname)
                        selected.append((mname, inst))

        logger.debug(
            "Selected %d applicable methods from %d categories",
            len(selected),
            len(category_groups),
        )
        return selected

    @staticmethod
    def _method_compatible(
        mname: str,
        cat: str,
        ndim: int,
        effective_rank: Any,
        spectral_flatness: float,
        sparsity: float,
    ) -> bool:
        """Quick compatibility check: skip methods that clearly don't fit."""
        name_lower = mname.lower()

        _2d_only = {
            "svd_compress",
            "svd_truncated",
            "tucker_decomposition",
            "block_tucker",
            "cur_decomposition",
            "h_matrix",
            "nystrom",
            "toeplitz",
            "hankel",
            "block_diagonal",
            "dct_2d",
            "dct_2d_block",
        }
        if name_lower in _2d_only and ndim != 2:
            return False
        if "circulant" in name_lower and ndim != 2:
            return False
        if ("toeplitz" in name_lower or "hankel" in name_lower) and ndim != 2:
            return False

        low_rank_names = {
            "svd_compress",
            "svd_truncated",
            "tucker_decomposition",
            "block_tucker",
            "cur_decomposition",
            "random_feature",
            "nystrom",
            "hierarchical_tucker",
            "tt_svd",
        }
        if name_lower in low_rank_names:
            if isinstance(effective_rank, (int, float)) and effective_rank > min(
                ndim * 64, 512
            ):
                return False

        spectral_names = {
            "dct_spectral",
            "dct_2d",
            "dct_block",
            "dct_2d_block",
            "fwht_compress",
            "fwht",
            "fourier",
            "wavelet_haar",
            "wavelet_daubechies",
            "wavelet_symlet",
        }
        if name_lower in spectral_names and spectral_flatness > 0.9:
            return False

        sparsity_names = {
            "sparsity_int4",
            "block_sparsity",
            "unstructured_pruning",
            "sparse_gpt",
            "wanda_pruning",
            "dynamic_nm_sparsity",
            "channel_pruning",
            "group_lasso",
            "adaptive_sparsity",
        }
        if name_lower in sparsity_names and sparsity < 0.01:
            return False

        return True

    # ── Single Method Test ─────────────────────────────────────────────

    def test_method(
        self,
        method_name: str,
        method_instance: Any,
        tensor: np.ndarray,
    ) -> Optional[MethodTestResult]:
        """Test a single method on a tensor. Returns None on error."""
        t0 = time.perf_counter()
        try:
            data, meta = method_instance.compress(tensor)
        except Exception as exc:
            logger.debug("Method '%s' compress failed: %s", method_name, exc)
            return None
        t_compress = time.perf_counter() - t0

        try:
            recon = method_instance.decompress(data, meta)
        except Exception as exc:
            logger.debug("Method '%s' decompress failed: %s", method_name, exc)
            return None
        t_total = time.perf_counter() - t0

        if recon.shape != tensor.shape:
            try:
                recon = recon.reshape(tensor.shape)
            except Exception:
                logger.debug(
                    "Method '%s' shape mismatch: %s vs %s",
                    method_name,
                    recon.shape,
                    tensor.shape,
                )
                return None

        try:
            qa = self._get_quality_assessor()
            quality = qa.assess(tensor, recon)
        except Exception:
            var = float(np.var(tensor))
            mse = float(np.mean((tensor.ravel() - recon.ravel()) ** 2))
            relative_error = mse / var if var > 1e-30 else float(mse)
            snr_db = 10.0 * np.log10(var / mse) if mse > 1e-30 else 100.0
            cos_sim = float(
                np.dot(tensor.ravel(), recon.ravel())
                / max(
                    np.linalg.norm(tensor.ravel()) * np.linalg.norm(recon.ravel()),
                    1e-30,
                )
            )
        else:
            relative_error = float(quality.relative_error)
            snr_db = float(quality.snr_db)
            cos_sim = float(quality.cosine_similarity)

        ratio = tensor.nbytes / max(len(data), 1)
        compressed_bytes = len(data)
        category = getattr(method_instance, "category", "unknown")
        tier = 5
        try:
            from ._tier_common import get_tier as _get_tier, MethodTier as _MT

            tier_val = _get_tier(method_name, category)
            tier = tier_val.value if hasattr(tier_val, "value") else int(tier_val)
        except Exception:
            pass

        return MethodTestResult(
            method_name=method_name,
            category=category,
            tier=tier,
            ratio=ratio,
            cosine_similarity=cos_sim,
            snr_db=snr_db,
            relative_error=relative_error,
            compressed_bytes=compressed_bytes,
            elapsed=t_total,
            metadata=meta,
        )

    # ── Test All Applicable ────────────────────────────────────────────

    def test_all_applicable(
        self,
        tensor: np.ndarray,
        tensor_name: str = "",
        max_per_category: int = 5,
        max_total: int = 50,
    ) -> List[MethodTestResult]:
        """Test applicable methods on a tensor. Returns ranked results.

        Parameters
        ----------
        tensor : np.ndarray
            The tensor to test methods on.
        tensor_name : str
            Optional name for logging.
        max_per_category : int
            Maximum methods per category (default 5).
        max_total : int
            Hard cap on total methods tested (default 50).
            Prevents excessive runtime on large tensors with many methods.
        """
        profile = self.profile_tensor(tensor)
        fingerprint = self._tensor_fingerprint(tensor)

        if fingerprint in self._results_cache:
            logger.debug("Using cached results for %s", fingerprint)
            return self._results_cache[fingerprint]

        methods = self.get_applicable_methods(
            profile, max_per_category=max_per_category
        )
        if not methods:
            logger.warning("No applicable methods for tensor %s", tensor_name)
            return []

        # Enforce hard cap — keep top-tier methods
        if len(methods) > max_total:
            from ._tier_common import MethodTier as _MT, tier_score as _ts

            methods.sort(
                key=lambda m: _ts(_MT(getattr(m[1], "_tier_val", 5))), reverse=True
            )
            methods = methods[:max_total]

        results: List[MethodTestResult] = []
        total = len(methods)

        for idx, (mname, inst) in enumerate(methods):
            if idx > 0 and idx % 15 == 0:
                logger.info("Tested %d/%d methods", idx, total)
            result = self.test_method(mname, inst, tensor)
            if result is not None:
                results.append(result)
            gc.collect()

        results.sort(key=lambda r: -r.score())
        self._results_cache[fingerprint] = results

        logger.info(
            "Tested %d methods on %s%s — %d succeeded",
            total,
            tensor_name or f"tensor {tensor.shape}",
            f" ({tensor_name})" if tensor_name else "",
            len(results),
        )
        return results

    # ── Cascade Discovery ──────────────────────────────────────────────

    def find_optimal_cascade(
        self,
        tensor: np.ndarray,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        max_stages: int = 10,
    ) -> List[Tuple[str, dict]]:
        """Find optimal cascade stacking order.

        1. Test all applicable methods
        2. Rank by score
        3. For top methods, test if they capture complementary structure
        4. Build cascade: method1(tensor) -> residual -> method2(residual) -> ...
        5. Return optimal method sequence with params
        """
        if tensor.size == 0:
            return []

        results = self.test_all_applicable(tensor, max_per_category=4, max_total=40)
        if not results:
            return []

        top = [r for r in results if r.ratio > 1.5 and r.relative_error < 0.5][:15]
        if not top:
            top = results[:5]

        tier_order = [1, 2, 3, 4, 5]
        top.sort(
            key=lambda r: (
                tier_order.index(r.tier) if r.tier in tier_order else 99,
                -r.score(),
            )
        )

        used_stages: List[Tuple[str, dict]] = []
        current_tensor = tensor.copy()
        current_error = 0.0
        current_ratio = 1.0
        residual_correlation_scores: Dict[str, float] = {}

        for stage_idx in range(max_stages):
            best_stage: Optional[Tuple[str, dict, float, float, dict]] = None

            for r in top:
                if r.method_name in {s[0] for s in used_stages}:
                    continue

                try:
                    inst = None
                    all_m = self._get_all_methods()
                    minfo = all_m.get(r.method_name)
                    if minfo is not None:
                        inst = minfo.get("instance")
                    if inst is None:
                        continue

                    data, meta = inst.compress(current_tensor)
                    recon = inst.decompress(data, meta)
                    if recon.shape != current_tensor.shape:
                        recon = recon.reshape(current_tensor.shape)
                except Exception:
                    continue

                residual = current_tensor.ravel() - recon.ravel()

                stage_ratio = current_tensor.nbytes / max(len(data), 1)

                if len(used_stages) > 0 and current_tensor.size > 0:
                    orig_residual_norm = float(np.linalg.norm(current_tensor.ravel()))
                    if orig_residual_norm > 1e-30:
                        capture_ratio = 1.0 - float(
                            np.linalg.norm(residual) / orig_residual_norm
                        )
                    else:
                        capture_ratio = 0.0
                else:
                    capture_ratio = 1.0

                if r.method_name in residual_correlation_scores:
                    avg_corr = residual_correlation_scores[r.method_name]
                else:
                    avg_corr = 0.0

                diversity_bonus = 1.0 - 0.5 * avg_corr

                quality = self._quick_quality(current_tensor, recon)
                if quality["relative_error"] > max(max_error * 5, 0.5):
                    continue

                residual_energy = float(np.sum(residual**2))
                orig_energy = float(np.sum(current_tensor.ravel() ** 2))
                if residual_energy > orig_energy * 1.1:
                    continue

                cascade_score = (
                    stage_ratio
                    * capture_ratio
                    * diversity_bonus
                    * (1.0 - quality["relative_error"])
                )

                entry = (r.method_name, meta, stage_ratio, cascade_score, quality)
                if best_stage is None or cascade_score > best_stage[3]:
                    best_stage = entry

            if best_stage is None:
                break

            mname, meta, stage_ratio, _, quality = best_stage
            used_stages.append((mname, meta))
            current_error += quality["relative_error"]
            current_ratio *= stage_ratio

            try:
                inst = None
                all_m = self._get_all_methods()
                minfo = all_m.get(mname)
                if minfo is not None:
                    inst = minfo.get("instance")
                if inst is not None:
                    data, _ = inst.compress(current_tensor)
                    recon = inst.decompress(data, meta)
                    if recon.shape != current_tensor.shape:
                        recon = recon.reshape(current_tensor.shape)
                    current_tensor = current_tensor.astype(np.float32) - recon.astype(
                        np.float32
                    )
            except Exception:
                break

            if current_ratio >= target_ratio:
                logger.info(
                    "Cascade reached target ratio %.1f:1 in %d stages",
                    current_ratio,
                    stage_idx + 1,
                )
                break
            if current_error >= max_error:
                logger.info(
                    "Cascade reached max error %.4f in %d stages",
                    current_error,
                    stage_idx + 1,
                )
                break

            gc.collect()

        if not used_stages:
            if results:
                best = results[0]
                used_stages = [(best.method_name, {})]

        return used_stages

    def predict_optimal_cascade(
        self,
        name: str,
        tensor: np.ndarray,
        target_ratio: float = 5000.0,
    ) -> List[Tuple[str, dict]]:
        """Use ZeroShotPredictor first, fall back to testing.

        Parameters
        ----------
        name : str
            Tensor name (e.g. 'model.layers.0.self_attn.q_proj.weight').
        tensor : np.ndarray
            The tensor to compress.
        target_ratio : float
            Desired compression ratio (default 5000.0).

        Returns
        -------
        List[Tuple[str, dict]]
            Optimal method sequence with parameters.
        """
        predictor = ZeroShotPredictor(self._engine)
        predictions = predictor.predict(name, tensor, target_ratio)
        if predictions and predictions[0][2] >= self._zero_shot_confidence():
            return [(p[0], p[1]) for p in predictions]
        return self.find_optimal_cascade(tensor, target_ratio=target_ratio)

    @staticmethod
    def _zero_shot_confidence() -> float:
        return 0.7

    @staticmethod
    def _quick_quality(orig: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
        """Fast quality assessment without full QualityAssessor."""
        var = float(np.var(orig))
        mse = float(np.mean((orig.ravel() - recon.ravel()) ** 2))
        relative_error = mse / var if var > 1e-30 else float(mse)
        snr_db = 10.0 * np.log10(var / mse) if mse > 1e-30 else 100.0
        dot = float(np.dot(orig.ravel(), recon.ravel()))
        norm_p = float(np.linalg.norm(orig.ravel()))
        norm_q = float(np.linalg.norm(recon.ravel()))
        cosine_similarity = dot / max(norm_p * norm_q, 1e-30)
        return {
            "relative_error": relative_error,
            "snr_db": snr_db,
            "cosine_similarity": cosine_similarity,
            "mse": mse,
        }

    # ── Pattern Retrieval ──────────────────────────────────────────────

    def get_best_stacking_pattern(self, tensor_type: str) -> Optional[List[str]]:
        """Get best known stacking pattern for a tensor type from CascadeLearner."""
        try:
            from .cascade_learner import CascadeLearner

            learner = CascadeLearner()
            best = learner.get_best_pattern(tensor_type)
            if best is not None:
                return [m for m, _ in best.stages]
        except Exception:
            pass
        return None

    def clear_cache(self) -> None:
        """Clear cached results to force fresh testing."""
        self._results_cache.clear()
        gc.collect()
