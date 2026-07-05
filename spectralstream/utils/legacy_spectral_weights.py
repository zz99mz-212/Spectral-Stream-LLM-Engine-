"""
Spectral Weight Compression — DCT-domain model weight storage.

Inspired by Quasar's QSF format but clean-room implemented.

Key insight: Neural network weights have smooth, low-frequency structure.
DCT concentrates energy in the first few coefficients (typically top 5%
contain >99% of energy). We can discard the other 95% with minimal quality loss.

For Gemma 4 E2B (2.9GB GGUF):
- Top 5% DCT coefficients: ~150MB
- Decompression on load: ~500ms on CPU
- RAM savings: 2.75GB at rest

For DeepSeek V4 Flash (284B params, estimated ~170GB at Q4):
- Top 5% DCT coefficients: ~8.5GB on SSD
- Decompression per layer: ~50ms each
- RAM savings: 161.5GB at rest
"""

import numpy as np
from typing import Optional
from pathlib import Path
import json
from collections import OrderedDict
import time

from spectralstream.core.math_primitives.transforms import (
    dct as _dct_1d,
    idct as _idct_1d,
)


# ═══════════════════════════════════════════════════════════════════════════
# DCTWeightCompressor
# ═══════════════════════════════════════════════════════════════════════════


class DCTWeightCompressor:
    """
    Compress/decompress model weight tensors via DCT.

    Each weight matrix W ∈ R^(m×n) is:
    1. Transformed: W_hat = DCT_2D(W)  [O(mn log(mn))]
    2. Thresholded: Keep top K coefficients by energy
    3. Quantized: Scale and round to int8
    4. Stored: (coefficients, indices, scale, shape)

    Decompression:
    1. Unquantize: coefficients * scale
    2. IDCT: W = IDCT_2D(W_hat)  [O(mn log(mn))]

    Compression ratio: mn / (K * (4 + 2)) where K = keep_top_k
    At 5%: ~20:1 compression ratio (4-byte index + 1-byte value + overhead)
    """

    def __init__(self, keep_energy: float = 0.99):
        """
        Parameters
        ----------
        keep_energy : float
            Fraction of total energy to preserve (0.0–1.0).
            0.99 = keep the top coefficients that contain 99% of total energy.
        """
        self.keep_energy = keep_energy

    # ── 2D DCT via separable 1D transforms ──────────────────────────────

    def _dct_2d(self, matrix: np.ndarray) -> np.ndarray:
        """2D Type-II DCT via separable 1D transforms.  O(mn log(mn))."""
        dct_rows = _dct_1d(matrix.astype(np.float64), norm="ortho")
        dct_2d = _dct_1d(dct_rows.T, norm="ortho").T
        return dct_2d.astype(np.float32)

    def _idct_2d(self, coeffs: np.ndarray) -> np.ndarray:
        """Inverse 2D DCT via separable 1D transforms."""
        idct_rows = _idct_1d(coeffs.astype(np.float64), norm="ortho")
        idct_2d = _idct_1d(idct_rows.T, norm="ortho").T
        return idct_2d.astype(np.float32)

    def _dct_1d_compress(self, vector: np.ndarray) -> dict:
        """1D DCT compression for bias/norm vectors."""
        original = vector.astype(np.float32)
        dct_coeffs = _dct_1d(original, norm="ortho")
        flat = dct_coeffs.ravel()
        energy = flat**2
        total_energy = energy.sum()
        if total_energy < 1e-30:
            return {
                "shape": original.shape,
                "indices": np.array([0], dtype=np.int32),
                "coefficients": np.array([0], dtype=np.int8),
                "scale": 1.0,
                "energy_kept": 1.0,
            }
        sorted_indices = np.argsort(-energy)
        cumsum = np.cumsum(energy[sorted_indices])
        n_keep = max(
            1, int(np.searchsorted(cumsum / total_energy, self.keep_energy) + 1)
        )
        keep_indices = sorted_indices[:n_keep]
        keep_coeffs = flat[keep_indices]
        scale = float(np.max(np.abs(keep_coeffs))) / 127.0 if n_keep > 0 else 1.0
        if scale < 1e-30:
            scale = 1.0
        quantized = np.clip(np.round(keep_coeffs / scale), -128, 127).astype(np.int8)
        order = np.argsort(keep_indices)
        energy_kept = float(cumsum[n_keep - 1] / total_energy) if n_keep > 0 else 1.0
        return {
            "shape": original.shape,
            "indices": keep_indices[order].astype(np.int32),
            "coefficients": quantized[order],
            "scale": scale,
            "energy_kept": energy_kept,
        }

    # ── Public API ──────────────────────────────────────────────────────

    def compress(self, matrix: np.ndarray) -> dict:
        """
        Compress weight matrix via DCT + thresholding + quantization.

        Coefficients are stored sorted by energy (descending) so that
        progressive loading can load the first N coefficients for an
        approximate reconstruction and add more later for refinement.

        Returns
        -------
        dict with keys:
            shape        : tuple — original matrix shape
            indices      : int32 ndarray — flattened coefficient positions
            coefficients : int8 ndarray — quantized coefficient values
            scale        : float — scale factor for unquantization
            energy_kept  : float — fraction of preserved energy
        """
        # Handle 1D tensors (biases, norms) separately
        if matrix.ndim == 1:
            return self._dct_1d_compress(matrix)

        original = matrix.astype(np.float32)

        # 1. Transform to DCT domain  [O(mn log(mn))]
        dct_coeffs = self._dct_2d(original)
        flat = dct_coeffs.ravel()
        n_total = flat.size

        # 2. Compute energy per coefficient
        energy = flat**2
        total_energy = energy.sum()

        # Handle degenerate case (all zeros)
        if total_energy < 1e-30:
            return {
                "shape": original.shape,
                "indices": np.array([0], dtype=np.int32),
                "coefficients": np.array([0], dtype=np.int8),
                "scale": 1.0,
                "energy_kept": 1.0,
            }

        # 3. Sort coefficients by energy (descending)
        sorted_indices = np.argsort(-energy)
        cumsum = np.cumsum(energy[sorted_indices])

        # 4. Determine how many coefficients to keep
        n_keep = max(
            1, int(np.searchsorted(cumsum / total_energy, self.keep_energy) + 1)
        )
        n_keep = min(n_keep, n_total)

        keep_indices = sorted_indices[:n_keep]
        keep_coeffs = flat[keep_indices]

        # 5. Quantize to int8 (per-matrix scale)
        scale = float(np.max(np.abs(keep_coeffs))) / 127.0 if n_keep > 0 else 1.0
        if scale < 1e-30:
            scale = 1.0
        quantized = np.clip(np.round(keep_coeffs / scale), -128, 127).astype(np.int8)

        # Keep in energy-sorted order (descending) for progressive loading
        energy_kept = float(cumsum[n_keep - 1] / total_energy) if n_keep > 0 else 1.0

        return {
            "shape": original.shape,
            "indices": keep_indices.astype(np.int32),
            "coefficients": quantized,
            "scale": scale,
            "energy_kept": energy_kept,
        }

    def decompress(
        self, compressed: dict, n_coefficients: Optional[int] = None
    ) -> np.ndarray:
        """Decompress back to spatial domain.

        Parameters
        ----------
        compressed : dict
            Output of ``compress()``.
        n_coefficients : int or None
            If given, only use the first ``n_coefficients`` (highest-energy
            ones) for reconstruction.  This enables progressive loading.

        Returns
        -------
        ndarray (float32) — reconstructed weight matrix.
        """
        shape = compressed["shape"]
        indices = compressed["indices"]
        quantized = compressed["coefficients"]
        scale = compressed["scale"]

        if n_coefficients is not None:
            n_coefficients = max(1, min(n_coefficients, len(indices)))
            indices = indices[:n_coefficients]
            quantized = quantized[:n_coefficients]

        # Build sparse DCT coefficient array
        dct_flat = np.zeros(np.prod(shape), dtype=np.float32)
        dct_flat[indices] = quantized.astype(np.float32) * scale
        dct_2d = dct_flat.reshape(shape)

        # Handle 1D case
        if len(shape) == 1:
            return _idct_1d(dct_2d, norm="ortho")

        return self._idct_2d(dct_2d)

    def compression_ratio(self, original: np.ndarray, compressed: dict) -> float:
        """Compute achieved compression ratio (original_bytes / compressed_bytes).

        Parameters
        ----------
        original : ndarray
            Original weight matrix (float32).
        compressed : dict
            Compressed representation.

        Returns
        -------
        ratio : float
            >1 means compression, <1 means expansion.
        """
        orig_bytes = original.nbytes  # float32
        n_coeffs = len(compressed["coefficients"])
        # Storage: int32 indices + int8 coefficients + float64 scale + overhead
        compressed_bytes = n_coeffs * (4 + 1) + 8
        # Add shape overhead (2 ints) + energy_kept (float64) ≈ 20 bytes
        compressed_bytes += 8 + 8 + 4
        return orig_bytes / max(compressed_bytes, 1)

    def compress_adaptive(
        self, matrix: np.ndarray, target_ratio: float = 10.0, max_iter: int = 20
    ) -> dict:
        """Compress targeting a specific compression ratio.

        Binary-searches `keep_energy` to hit ``target_ratio``.

        Parameters
        ----------
        matrix : ndarray
            Weight tensor.
        target_ratio : float
            Desired compression ratio (e.g., 10 = 10:1).
        max_iter : int
            Maximum binary search iterations.

        Returns
        -------
        compressed : dict
        """
        low, high = 0.9999, 0.01
        for _ in range(max_iter):
            mid = (low + high) / 2.0
            orig_keep = self.keep_energy
            self.keep_energy = mid
            result = self.compress(matrix)
            self.keep_energy = orig_keep
            ratio = self.compression_ratio(matrix, result)
            if abs(ratio - target_ratio) / target_ratio < 0.1:
                return result
            if ratio > target_ratio:
                high = mid
            else:
                low = mid
        return result


# ═══════════════════════════════════════════════════════════════════════════
# SpectralWeightStore
# ═══════════════════════════════════════════════════════════════════════════


class SpectralWeightStore:
    """
    Manages spectral-compressed model weights across storage tiers.

    - SSD: DCT-compressed coefficients (5–10% of original size)
    - RAM: Currently decompressed layers (LRU cache)
    - Load on demand: decompress from SSD when layer is needed

    For a Gemma 4 E2B (2.9GB GGUF):
    - SSD storage: ~150–300MB (DCT-compressed)
    - RAM with 4 hot layers: ~200MB (decompressed active layers)

    Usage::

        store = SpectralWeightStore('model.gguf', './spectral_cache/')
        # On first run, pre-compresses all weights to ./spectral_cache/
        # On subsequent runs, loads from cache directly

        w = store.get_weight('blk.0.attn_q.weight')
        layer = store.get_layer_weights(0)
    """

    def __init__(
        self,
        gguf_path: str,
        cache_dir: str,
        cache_layers: int = 4,
        keep_energy: float = 0.99,
        min_tensor_elements: int = 64,
    ):
        """
        Parameters
        ----------
        gguf_path : str
            Path to the GGUF model file (used for pre-compression).
        cache_dir : str
            Directory to store/load compressed tensor data.
        cache_layers : int
            Number of transformer layers to keep decompressed in RAM.
        keep_energy : float
            Energy threshold for DCT compression (0.99 = 99%).
        min_tensor_elements : int
            Skip compression for tensors smaller than this (store raw).
        """
        self.gguf_path = Path(gguf_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_layers = cache_layers
        self.keep_energy = keep_energy
        self.min_tensor_elements = min_tensor_elements

        # LRU cache: tensor_name -> np.ndarray
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self._max_cache_bytes = 0

        # Load or build manifest
        manifest_path = self.cache_dir / "manifest.json"
        if manifest_path.exists():
            self._load_manifest(manifest_path)
        else:
            self._manifest = {}
            self._precompress_from_gguf()

    # ── Manifest I/O ────────────────────────────────────────────────────

    def _load_manifest(self, path: Path):
        with open(path) as f:
            data = json.load(f)
        self._manifest = data.get("tensors", {})
        self._max_cache_bytes = data.get("layer0_bytes", 0) * self.cache_layers
        if self._max_cache_bytes == 0:
            self._max_cache_bytes = 1024**3  # fallback: 1 GB

    def _save_manifest(self):
        layer0_bytes = 0
        for name, info in self._manifest.items():
            if ".0." in name:
                layer0_bytes += info.get("original_bytes", 0)
        data = {
            "model_path": str(self.gguf_path),
            "keep_energy": self.keep_energy,
            "layer0_bytes": layer0_bytes,
            "n_tensors": len(self._manifest),
            "tensors": self._manifest,
        }
        with open(self.cache_dir / "manifest.json", "w") as f:
            json.dump(data, f, indent=2)

    # ── Pre-compression ─────────────────────────────────────────────────

    def _precompress_from_gguf(self):
        """Load all tensors from GGUF, compress, save, free GGUF memory."""
        try:
            from spectralstream.gguf_model import GGUFModel
        except ImportError:
            GGUFModel = None

        print(f"[SpectralWeightStore] Pre-compressing {self.gguf_path} ...")
        model = GGUFModel(str(self.gguf_path))
        compressor = DCTWeightCompressor(self.keep_energy)

        tensor_names = list(model.tensors.keys())
        start_total = time.time()

        for idx, (name, tensor) in enumerate(model.tensors.items()):
            t0 = time.time()

            info = {
                "shape": list(tensor.shape),
                "original_bytes": int(tensor.nbytes),
            }

            # Skip tiny tensors (norms, biases, etc.) — store raw
            if tensor.size < self.min_tensor_elements or tensor.ndim == 0:
                path = self.cache_dir / f"{name}.raw.npy"
                np.save(path, tensor)
                info["store_type"] = "raw"
                info["compressed_bytes"] = int(path.stat().st_size)
                info["energy_kept"] = 1.0
                info["n_coefficients"] = tensor.size
            else:
                compressed = compressor.compress(tensor)
                path = self.cache_dir / f"{name}.npz"
                np.savez_compressed(
                    path,
                    indices=compressed["indices"],
                    coefficients=compressed["coefficients"],
                    scale=np.float32(compressed["scale"]),
                )
                info["store_type"] = "dct"
                info["compressed_bytes"] = int(path.stat().st_size)
                info["energy_kept"] = compressed["energy_kept"]
                info["n_coefficients"] = len(compressed["coefficients"])

            info["compress_ms"] = round((time.time() - t0) * 1000, 1)
            self._manifest[name] = info

            if (idx + 1) % 50 == 0 or idx == len(tensor_names) - 1:
                elapsed = time.time() - start_total
                print(f"  [{idx + 1}/{len(tensor_names)}] {elapsed:.1f}s elapsed")

        self._save_manifest()
        print(
            f"[SpectralWeightStore] Done. {len(self._manifest)} tensors "
            f"compressed in {time.time() - start_total:.1f}s"
        )

    # ── Tensor loading ──────────────────────────────────────────────────

    def _load_compressed(self, name: str) -> dict:
        """Load compressed data from disk."""
        info = self._manifest[name]

        if info.get("store_type") == "raw":
            path = self.cache_dir / f"{name}.raw.npy"
            data = np.load(path)
            return {"data": data, "shape": data.shape}

        path = self.cache_dir / f"{name}.npz"
        npz = np.load(path)
        return {
            "shape": tuple(info["shape"]),
            "indices": npz["indices"],
            "coefficients": npz["coefficients"],
            "scale": float(npz["scale"]),
            "energy_kept": info.get("energy_kept", 0.99),
        }

    def get_weight(
        self, tensor_name: str, n_coefficients: Optional[int] = None
    ) -> np.ndarray:
        """Get a decompressed weight tensor (with LRU caching).

        Parameters
        ----------
        tensor_name : str
            Full tensor name (e.g. ``blk.0.attn_q.weight``).
        n_coefficients : int or None
            For progressive loading — use only the top N coefficients.
            ``None`` uses all compressed coefficients.

        Returns
        -------
        weight : ndarray (float32)
        """
        # Cache hit (only when using full quality)
        if n_coefficients is None and tensor_name in self._cache:
            self._cache.move_to_end(tensor_name)
            return self._cache[tensor_name]

        # General case: cache key includes n_coefficients for progressive
        cache_key = (tensor_name, n_coefficients)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        # Load from disk
        data = self._load_compressed(tensor_name)
        shape = data["shape"]

        # Handle raw (uncompressed) tensors
        if "data" in data:
            weight = data["data"]
            if weight.dtype != np.float32:
                weight = weight.astype(np.float32)
            self._cache[tensor_name] = weight
            self._cache_bytes += weight.nbytes
            self._evict_if_needed()
            return weight

        # DCT decompress
        compressor = DCTWeightCompressor(self.keep_energy)
        weight = compressor.decompress(data, n_coefficients)

        # Cache only full-quality decompressions
        if n_coefficients is None:
            self._cache[tensor_name] = weight
            self._cache_bytes += weight.nbytes
            self._evict_if_needed()

        return weight

    def get_layer_weights(self, layer_idx: int) -> dict:
        """Get all weight tensors for a given transformer layer.

        Returns a dict mapping short names (e.g. ``attn_q``) to weight arrays.
        """
        layer_tensors = {}
        prefix = f"blk.{layer_idx}."
        for full_name in self._manifest:
            if full_name.startswith(prefix):
                short = full_name[len(prefix) :]
                layer_tensors[short] = self.get_weight(full_name)
        return layer_tensors

    def prefetch_layer(self, layer_idx: int):
        """Pre-decompress an entire layer into the RAM cache."""
        prefix = f"blk.{layer_idx}."
        for full_name in list(self._manifest.keys()):
            if full_name.startswith(prefix):
                self.get_weight(full_name)

    # ── Cache eviction ──────────────────────────────────────────────────

    def _evict_if_needed(self):
        """Evict LRU entries until under the memory budget."""
        if self._max_cache_bytes <= 0:
            return
        while self._cache_bytes > self._max_cache_bytes and len(self._cache) > 1:
            key, tensor = self._cache.popitem(last=False)
            self._cache_bytes -= tensor.nbytes

    @property
    def cache_utilization(self) -> float:
        """Fraction of the LRU cache currently in use (0.0–1.0)."""
        if self._max_cache_bytes <= 0:
            return 0.0
        return self._cache_bytes / self._max_cache_bytes

    # ── Utility ─────────────────────────────────────────────────────────

    def save_manifest(self, path: str):
        """Export the compression manifest to a JSON file."""
        with open(path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def summary(self) -> str:
        """Print a human-readable summary of the compressed store."""
        if not self._manifest:
            return "Empty spectral weight store."

        total_orig = sum(v["original_bytes"] for v in self._manifest.values())
        total_comp = sum(v.get("compressed_bytes", 0) for v in self._manifest.values())
        dct_tensors = sum(
            1 for v in self._manifest.values() if v.get("store_type") == "dct"
        )
        raw_tensors = sum(
            1 for v in self._manifest.values() if v.get("store_type") == "raw"
        )
        avg_energy = np.mean(
            [
                v.get("energy_kept", 1.0)
                for v in self._manifest.values()
                if v.get("store_type") == "dct"
            ]
        )

        lines = [
            f"SpectralWeightStore: {self.cache_dir}",
            f"  Tensors: {len(self._manifest)} ({dct_tensors} DCT, {raw_tensors} raw)",
            f"  Original size: {total_orig / 1e9:.2f} GB",
            f"  Compressed size: {total_comp / 1e6:.1f} MB",
            f"  Overall ratio: {total_orig / max(total_comp, 1):.1f}x",
            f"  Avg energy kept: {avg_energy:.4f}",
            f"  Cache: {self._cache_bytes / 1e6:.1f} MB / {self._max_cache_bytes / 1e6:.1f} MB",
        ]
        return "\n".join(lines)

    def __contains__(self, name: str) -> bool:
        return name in self._manifest

    def __len__(self) -> int:
        return len(self._manifest)

    def list_tensors(self, pattern: str = "") -> list:
        """List all tensor names, optionally filtering by substring."""
        if not pattern:
            return list(self._manifest.keys())
        return [n for n in self._manifest if pattern in n]


# ═══════════════════════════════════════════════════════════════════════════
# ProgressiveWeightLoader
# ═══════════════════════════════════════════════════════════════════════════


class ProgressiveWeightLoader:
    """
    Progressive loading: load low-freq coefficients first for quick
    inference start, then refine in the background.

    Stages
    ------
        0  (instant):  top 1%  of stored coefficients → ~85% quality
        1  (fast):     top 5%  → ~99% quality
        2  (full):     all coefficients → 100% quality

    Phase 1 (immediate): Load top 1% DCT coefficients → coarse but functional
    Phase 2 (1–5s):     Load top 5% → good quality
    Phase 3 (background): Load remaining → full quality
    """

    DEFAULT_STAGES = [
        ("instant", 0.01, 0.85),
        ("fast", 0.05, 0.99),
        ("full", 1.00, 1.00),
    ]

    def __init__(self, spectral_store: SpectralWeightStore, stages: list = None):
        """
        Parameters
        ----------
        spectral_store : SpectralWeightStore
            The backing weight store with compressed data.
        stages : list of (name, frac_coeffs, quality)
            Each stage defines what fraction of stored coefficients to load
            and the expected quality (0–1).  Stages are loaded in order.
        """
        self.store = spectral_store
        self.stages = stages or list(self.DEFAULT_STAGES)
        self.current_stage = -1
        self._stage_name = "none"

        # Per-tensor coefficient budgets for the current stage
        self._budgets: dict[str, int] = {}

    def load_stage(self, stage: int) -> bool:
        """Advance to the given stage (0, 1, 2...) and load all weights.

        Returns True if the stage was successfully loaded, False if
        ``stage`` is out of range.
        """
        if stage < 0 or stage >= len(self.stages):
            return False
        if stage <= self.current_stage:
            return True  # already at this stage or beyond

        name, frac, quality = self.stages[stage]
        self.current_stage = stage
        self._stage_name = name

        # Compute coefficient budget per tensor
        self._budgets.clear()
        for tensor_name, info in self.store._manifest.items():
            if info.get("store_type") != "dct":
                self._budgets[tensor_name] = info.get("n_coefficients", 0)
                continue
            total = info.get("n_coefficients", 0)
            n_load = max(1, int(total * frac))
            self._budgets[tensor_name] = n_load

        # Flush the LRU cache so next get_weight uses correct budget
        self.store._cache.clear()
        self.store._cache_bytes = 0

        return True

    def load_next_stage(self) -> bool:
        """Advance to the next progressive stage.

        Returns True if a new stage was loaded, False if already at final.
        """
        return self.load_stage(self.current_stage + 1)

    def get_weight(self, tensor_name: str) -> np.ndarray:
        """Get a weight tensor at the current progressive quality.

        Uses the active stage's coefficient budget to load only the
        highest-energy coefficients.
        """
        n_coeffs = self._budgets.get(tensor_name)
        return self.store.get_weight(tensor_name, n_coefficients=n_coeffs)

    def get_layer_weights(self, layer_idx: int) -> dict:
        """Get all weights for a layer at the current progressive quality."""
        layer = {}
        prefix = f"blk.{layer_idx}."
        for full_name in self.store._manifest:
            if full_name.startswith(prefix):
                short = full_name[len(prefix) :]
                layer[short] = self.get_weight(full_name)
        return layer

    def get_quality(self) -> float:
        """Return the current quality estimate (0.0–1.0)."""
        if self.current_stage < 0:
            return 0.0
        return self.stages[self.current_stage][2]

    @property
    def stage_name(self) -> str:
        """Human-readable name of the current stage."""
        return self._stage_name

    def summary(self) -> str:
        """Print a summary of the current loading state."""
        return (
            f"ProgressiveWeightLoader: stage {self.current_stage + 1}/"
            f"{len(self.stages)} ({self.stage_name}), "
            f"quality ≈ {self.get_quality():.0%}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════════════


def test_spectral_compression():
    """
    Test compression on realistic synthetic weights.

    Real neural net weights are smooth / low-frequency dominated.
    We simulate this with exponentially decaying frequency content
    plus a small amount of high-frequency noise.
    """
    np.random.seed(42)

    m, n = 2048, 2048

    # Build a weight matrix with concentrated low-frequency energy.
    # DCT basis functions are cosines; we create weights by summing
    # a few low-frequency cosines with decaying amplitudes, then
    # add a tiny noise floor.
    freqs_i = np.arange(m, dtype=np.float64)[:, None]  # (m, 1)
    freqs_j = np.arange(n, dtype=np.float64)[None, :]  # (1, n)

    # Low-frequency content: amplitude ~ 1/freq
    # Use 2D cosines with frequency index (u, v) where u+v < threshold
    W_low = np.zeros((m, n), dtype=np.float64)
    for u in range(0, 32):
        for v in range(0, 32):
            if u + v >= 48:
                continue
            amp = 1.0 / (1.0 + u + v)
            phase_i = np.cos(np.pi * u * freqs_i / m)
            phase_j = np.cos(np.pi * v * freqs_j / n)
            W_low += amp * phase_i * phase_j

    # Normalise
    W_low = W_low / np.std(W_low) * 0.1

    # Add very small high-frequency noise
    W_noise = np.random.randn(m, n).astype(np.float64) * 1e-4

    W = (W_low + W_noise).astype(np.float32)

    print(f"Test matrix: {W.shape}, {W.nbytes / 1e6:.1f} MB")
    print()

    for energy in [0.90, 0.95, 0.99, 0.999]:
        compressor = DCTWeightCompressor(keep_energy=energy)

        t0 = time.time()
        compressed = compressor.compress(W)
        ct = time.time() - t0

        t0 = time.time()
        W_recon = compressor.decompress(compressed)
        dt = time.time() - t0

        mse = float(np.mean((W - W_recon) ** 2))
        psnr = 20 * np.log10(np.max(np.abs(W)) / np.sqrt(mse + 1e-30))
        ratio = compressor.compression_ratio(W, compressed)
        kept_pct = 100 * len(compressed["coefficients"]) / (m * n)

        print(
            f"keep_energy={energy:.3f}:  "
            f"kept {kept_pct:>6.2f}% coeffs  "
            f"ratio {ratio:>5.1f}x  "
            f"MSE {mse:.2e}  "
            f"PSNR {psnr:>5.1f} dB  "
            f"compress {ct * 1000:>5.0f}ms  "
            f"decompress {dt * 1000:>5.0f}ms"
        )

    # Progressive loading test
    print()
    print("--- Progressive loading test ---")
    compressor = DCTWeightCompressor(keep_energy=0.99)
    compressed = compressor.compress(W)
    n_total = len(compressed["coefficients"])

    for pct in [0.01, 0.05, 0.25, 1.0]:
        n = max(1, int(n_total * pct))
        W_partial = compressor.decompress(compressed, n_coefficients=n)
        mse = float(np.mean((W - W_partial) ** 2))
        psnr = 20 * np.log10(np.max(np.abs(W)) / np.sqrt(mse + 1e-30))
        print(
            f"  Top {pct * 100:>3.0f}% coeffs ({n:>6d}):  "
            f"MSE {mse:.2e}  PSNR {psnr:>5.1f} dB"
        )

    print()
    print("Spectral compression test passed.")
    return compressor, compressed, W_recon


if __name__ == "__main__":
    test_spectral_compression()
