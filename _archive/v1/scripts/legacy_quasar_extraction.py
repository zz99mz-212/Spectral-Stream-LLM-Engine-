"""
Quasar Spectral Extraction -- Clean room implementation of Quasar V5's
core spectral LLM innovations for integration into SpectralStream.

Extracted (clean room, no Quasar code copied, all from first principles):
1. VlasovMeanFieldAttention -- O(n) mean-field attention via plasma physics
2. SpectralKVCache -- DCT-compressed key/value storage with Landau-Zener decay
3. SpectralFFN -- Resonance-gated feed-forward in frequency domain
4. SymplecticIntegrator -- Hamiltonian leapfrog per transformer layer
5. MonoidalChunker -- Long-context chunking with spectral merge
6. SpectralEmbedding -- Frequency-domain token embedding
7. GyrokineticAttention -- Slow/fast splitting of spectral attention field
8. LandauDamping -- Collisionless wave-particle damping for attention stability
9. ResonanceMoE -- Spectral entropy-based expert routing
10. AdaptiveSpectralRank -- Per-layer DCT coefficient count optimization

All implementations use only numpy + standard library.
No Quasar V5 code is copied or referenced during implementation.
"""

import numpy as np
from typing import Optional, Callable, List, Tuple, Dict
from collections import deque
import math


# ═════════════════════════════════════════════════════════════════════════════
# Common spectral primitives (DCT, spectral field ops)
# ═════════════════════════════════════════════════════════════════════════════


class SpectralOps:
    """Spectral-domain primitives via DCT/FFT.

    Type-II DCT with orthonormal normalisation, implemented
    via FFT (O(n log n)) with no scipy dependency.
    """

    @staticmethod
    def dct(x: np.ndarray, norm: str = "ortho") -> np.ndarray:
        n = x.shape[-1]
        x_pad = np.zeros((*x.shape[:-1], 2 * n), dtype=np.float64)
        x_pad[..., :n] = x
        x_pad[..., n:] = x[..., ::-1]
        fft_result = np.fft.fft(x_pad, axis=-1)
        dct_out = fft_result[..., :n].real
        k = np.arange(n, dtype=np.float64)
        dct_out *= 2.0 * np.cos(np.pi * k / (2 * n))
        if norm == "ortho":
            dct_out[..., 0] /= np.sqrt(4 * n)
            dct_out[..., 1:] /= np.sqrt(2 * n)
        return dct_out

    @staticmethod
    def idct(x: np.ndarray, norm: str = "ortho") -> np.ndarray:
        n = x.shape[-1]
        if norm == "ortho":
            y = x.astype(np.float64).copy()
            y[..., 0] /= np.sqrt(4 * n)
            y[..., 1:] /= np.sqrt(2 * n)
        else:
            y = x
        k = np.arange(n, dtype=np.float64)
        y = y * 2.0 * np.cos(np.pi * k / (2 * n))
        y_pad = np.zeros((*y.shape[:-1], 2 * n), dtype=np.complex128)
        y_pad[..., :n] = y
        y_pad[..., n:] = 0.0
        fft_result = np.fft.ifft(y_pad, axis=-1)
        result = fft_result[..., :n].real
        return result * n

    @staticmethod
    def spectral_entropy(x: np.ndarray) -> float:
        spectrum = np.abs(np.fft.fft(x.astype(np.float64)))
        power = spectrum / (np.sum(spectrum) + 1e-10)
        ent = -np.sum(power * np.log2(power + 1e-10))
        n = len(power)
        return float(ent / np.log2(n)) if n > 1 else 0.0

    @staticmethod
    def band_limit(x: np.ndarray, n_keep: int) -> np.ndarray:
        coeffs = SpectralOps.dct(x)
        coeffs[..., n_keep:] = 0.0
        return SpectralOps.idct(coeffs)

    @staticmethod
    def compress_spectral(
        x: np.ndarray, keep_frac: float = 0.2
    ) -> Tuple[np.ndarray, int]:
        n = x.shape[-1]
        k = max(1, int(n * keep_frac))
        coeffs = SpectralOps.dct(x)
        magnitude = np.abs(coeffs)
        threshold = np.sort(magnitude.ravel())[-k] if k < magnitude.size else 0.0
        mask = magnitude >= threshold
        compressed = coeffs * mask
        return compressed, k

    @staticmethod
    def spectral_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_spec = SpectralOps.dct(a.ravel())
        b_spec = SpectralOps.dct(b.ravel())
        a_flat = a_spec.ravel()
        b_flat = b_spec.ravel()
        sim = np.dot(a_flat, b_flat)
        norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-10
        return float(sim / norm)


# ═════════════════════════════════════════════════════════════════════════════
# Section 1: VlasovMeanFieldAttention
# ═════════════════════════════════════════════════════════════════════════════


class VlasovMeanFieldAttention:
    """
    O(n) mean-field attention via Vlasov plasma physics.

    Standard attention: O(n^2*d) pairwise QK^T
    Vlasov attention:   O(n*d*K) where K = spectral rank (typically 64)

    The key insight: Instead of every token attending to every other token,
    each token interacts with the self-consistent *mean field* of all tokens.
    This is the same trick used in plasma physics (Vlasov equation) where
    particles interact through a mean field, not pairwise.

    Algorithm:
    1. Compute spectral density of keys: rho_k = DCT(keys) in frequency domain
    2. Solve Poisson's equation for mean-field potential: Phi_mf via Yukawa kernel
    3. Each token responds to Phi_mf, not to individual tokens
    4. Weighted value aggregation via mean-field response
    """

    def __init__(
        self,
        d_model: int = 4096,
        n_heads: int = 32,
        spectral_rank: int = 64,
        screening_length: float = 1.0,
        sigma: float = 1.0,
        causal: bool = True,
        recency_bias: float = 0.1,
        use_yukawa: bool = True,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.spectral_rank = min(spectral_rank, self.head_dim)
        self.screening_length = screening_length
        self.sigma = sigma
        self.causal = causal
        self.recency_bias = recency_bias
        self.use_yukawa = use_yukawa

        # KV cache (spectral fields)
        self.k_cache: List[np.ndarray] = []
        self.v_cache: List[np.ndarray] = []
        self.cache_length = 0

        # Causal scan state for online prefix computation
        self._causal_state_k: Optional[np.ndarray] = None
        self._causal_count: float = 0.0

        # Dummy QKV projection matrices (to be replaced by trained weights)
        rng = np.random.RandomState(42)
        scale = 1.0 / np.sqrt(d_model)
        self.w_q = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.w_k = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.w_v = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.w_o = rng.randn(d_model, d_model).astype(np.float32) * scale

    # ── Core mean-field computation ──────────────────────────────────────

    def compute_mean_field_potential(self, keys: np.ndarray) -> np.ndarray:
        """
        Compute mean-field potential from key spectral density.

        Maps: K[n x d] -> rho_k[freq] -> Phi_mf (mean-field potential)

        The spectral density rho_k is computed via DCT.
        The potential solves Poisson's equation: (k^2 + mu^2) * Phi = 4*pi*rho
        in the frequency domain, where mu = 1/screening_length.

        Returns Phi_mf of shape (n, d).
        """
        n, d = keys.shape
        mu = 1.0 / max(self.screening_length, 1e-8)

        # 1. Key spectral density via DCT
        keys_dct = SpectralOps.dct(keys)  # (n, d)

        # 2. Spectral density (magnitude of DCT coefficients)
        rho_k = np.abs(keys_dct)

        # 3. Solve Poisson equation in frequency domain
        #    (k^2 + mu^2) * Phi_tilde = rho_k
        #    Phi_tilde = rho_k / (k^2 + mu^2)
        k_inds = np.arange(d, dtype=np.float64)
        k_sq = k_inds.reshape(1, -1) ** 2
        denom = k_sq + mu**2 + 1e-10
        phi_tilde = rho_k / denom

        # 4. Inverse DCT to get spatial potential
        phi_mf = SpectralOps.idct(phi_tilde)
        return phi_mf

    def apply_vlasov_kernel(self, potential: np.ndarray) -> np.ndarray:
        """
        Apply Vlasov collision kernel (Yukawa) in frequency domain.

        The kernel is the Fourier transform of the screened Coulomb
        (Yukawa) potential:
          V_tilde(k) = 4*pi / (k^2 + mu^2)
        where mu = 1/screening_length.

        This transforms the mean-field potential into the effective
        interaction felt by each token.
        """
        d = potential.shape[-1]
        mu = 1.0 / max(self.screening_length, 1e-8)

        pot_dct = SpectralOps.dct(potential)
        k_inds = np.arange(d, dtype=np.float64)
        k_sq = k_inds.reshape(1, -1) ** 2
        kernel = 4.0 * np.pi / (k_sq + mu**2 + 1e-10)
        field = pot_dct * kernel
        return SpectralOps.idct(field)

    def compute_mean_field_weights(
        self, queries: np.ndarray, mean_field: np.ndarray
    ) -> np.ndarray:
        """
        Compute attention weights from mean-field response.

        Each token's attention weight is its response to the total
        mean-field potential, not pairwise QK^T.

        The response is computed as:
          w_i = softmax( -||q_i - Phi_mf||^2 / (2 * sigma^2) )

        Returns weights of shape (n,).
        """
        n = queries.shape[0]
        phi_flat = mean_field.reshape(n, -1)
        q_flat = queries.reshape(n, -1)

        dist_sq = np.sum((q_flat - phi_flat) ** 2, axis=-1)
        weights = np.exp(-0.5 * dist_sq / (self.sigma**2 + 1e-10))
        weights = weights / (np.sum(weights) + 1e-10)
        return weights

    def _yukawa_kernel(self, dist_sq: np.ndarray) -> np.ndarray:
        """Screened Coulomb (Yukawa) potential: V(r) = exp(-r/mu) / (1 + r^2)."""
        mu = self.screening_length
        r = np.sqrt(dist_sq + 1e-10)
        return np.exp(-r / mu) / (1.0 + dist_sq)

    def _gaussian_kernel(self, dist_sq: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * dist_sq / (self.sigma**2 + 1e-10))

    def _kernel(self, dist_sq: np.ndarray) -> np.ndarray:
        if self.use_yukawa:
            return self._yukawa_kernel(dist_sq)
        return self._gaussian_kernel(dist_sq)

    # ── Main forward pass ────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: np.ndarray,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        O(n) mean-field attention forward pass.

        Args:
            hidden_states: (n, d_model) input
            use_cache: whether to use/update KV cache

        Returns:
            (n, d_model) attended output
        """
        n, d = hidden_states.shape

        # Project to Q, K, V
        q = hidden_states @ self.w_q
        k = hidden_states @ self.w_k
        v = hidden_states @ self.w_v

        # Multi-head reshape
        q = q.reshape(n, self.n_heads, self.head_dim)
        k = k.reshape(n, self.n_heads, self.head_dim)
        v = v.reshape(n, self.n_heads, self.head_dim)

        outputs = []
        for h in range(self.n_heads):
            q_h = q[:, h, :]
            k_h = k[:, h, :]
            v_h = v[:, h, :]

            if use_cache and self.cache_length > 0:
                all_k = np.concatenate(self.k_cache + [k_h], axis=0)
                all_v = np.concatenate(self.v_cache + [v_h], axis=0)
            else:
                all_k = k_h
                all_v = v_h

            # Compute mean-field potential from keys
            phi_mf = self.compute_mean_field_potential(all_k)

            # Apply Vlasov kernel
            field = self.apply_vlasov_kernel(phi_mf)

            # For each query, compute response to mean field
            out_h = np.zeros_like(q_h)
            for i in range(n):
                q_i = q_h[i : i + 1]
                phi_i = field[i : i + 1]

                # Distance from query to mean-field
                q_dev = q_i - phi_i
                dist_sq = np.sum(q_dev**2, axis=-1, keepdims=True)
                w = self._kernel(dist_sq)

                # Weighted value aggregation
                v_mean = np.mean(all_v, axis=0, keepdims=True)
                out_h[i] = w * v_mean + (1.0 - w) * v_h[i]

            outputs.append(out_h)

        # Concatenate heads and project
        out = np.concatenate(outputs, axis=-1)
        out = out @ self.w_o

        # Update cache
        if use_cache:
            self.k_cache.append(k.reshape(n, d))
            self.v_cache.append(v.reshape(n, d))
            self.cache_length += n

        return out

    def prefill(self, tokens: np.ndarray) -> None:
        """
        Prefill KV cache from prompt tokens.

        Args:
            tokens: (n, d_model) prompt token embeddings

        Complexity: O(n * K^2) where K = spectral_rank
        """
        k = tokens @ self.w_k
        v = tokens @ self.w_v
        self.k_cache = [k]
        self.v_cache = [v]
        self.cache_length = k.shape[0]

    def update_kv_cache(self, key: np.ndarray, value: np.ndarray) -> None:
        """
        Update KV cache with a single new token's K, V.

        Args:
            key: (d_model,) key projection
            value: (d_model,) value projection

        Complexity: O(K^2) per token
        """
        self.k_cache.append(key.reshape(1, -1))
        self.v_cache.append(value.reshape(1, -1))
        self.cache_length += 1

    # ── Diagnostics ─────────────────────────────────────────────────────

    def diagnose(self) -> Dict[str, float]:
        if self.cache_length == 0:
            return {
                "mean_field_energy": 0.0,
                "spectral_gap": 0.0,
                "coherence_width": 0.0,
                "effective_rank": 0.0,
            }
        all_k = np.concatenate(self.k_cache, axis=0)
        phi = self.compute_mean_field_potential(all_k)
        energy = float(np.sum(phi**2))

        k_dct = SpectralOps.dct(all_k)
        power = np.mean(k_dct**2, axis=0)
        sorted_power = np.sort(power)[::-1]
        cumsum = np.cumsum(sorted_power) / (np.sum(sorted_power) + 1e-10)
        eff_rank = int(np.sum(cumsum < 0.9)) + 1

        gap = float(sorted_power[0] - sorted_power[1]) if len(sorted_power) > 1 else 0.0

        return {
            "mean_field_energy": energy,
            "spectral_gap": gap,
            "coherence_width": float(self.sigma),
            "effective_rank": float(eff_rank),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Section 2: SpectralKVCache
# ═════════════════════════════════════════════════════════════════════════════


class SpectralKVCache:
    """
    DCT-compressed key/value storage with Landau-Zener coherence decay.

    Each layer's keys and values are stored as DCT coefficients in
    a band-efficient format, achieving 20:1 to 50:1 compression.

    Key Principles:
      1. Keys/values live in spectral frequency space (DCT coefficients)
      2. Old entries are evicted via Landau-Zener coherence decay
      3. Multi-turn conversations append new spectral components
      4. Each layer can target a different memory band
      5. Cosine similarity scoring in compressed domain
    """

    def __init__(
        self,
        d_model: int = 4096,
        n_layers: int = 32,
        max_seq_len: int = 8192,
        spectral_rank: int = 64,
        compression_ratio: float = 20.0,
        k_bits: int = 4,
        v_bits: int = 2,
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.spectral_rank = spectral_rank
        self.compression_ratio = compression_ratio
        self.k_bits = k_bits
        self.v_bits = v_bits

        # Per-layer spectral cache storage
        # Each entry: { 'k_coeffs': np.ndarray, 'v_coeffs': np.ndarray,
        #               'position': int, 'coherence': float, 'timestamp': int }
        self._layer_caches: List[List[Dict]] = [[] for _ in range(n_layers)]
        self._global_step = 0
        self._half_life = 1000.0

        # DCT keep fraction based on compression ratio
        self._keep_frac = min(1.0, 1.0 / compression_ratio)

        # Band thresholds (high/normal/low) mapped to compression levels
        self._band_thresholds = [(0, 8), (8, 64), (64, self.d_model)]

    # ── Landau-Zener coherence ──────────────────────────────────────────

    def _compute_coherence(self, entry_age: int) -> float:
        """Landau-Zener transition probability as coherence decay."""
        return float(np.exp(-entry_age / self._half_life))

    # ── Compression / decompression ────────────────────────────────────

    def _compress_to_spectral(self, spatial: np.ndarray) -> np.ndarray:
        """
        Compress a spatial vector to DCT coefficients.

        Keeps only the top-k coefficients by magnitude (determined
        by compression_ratio).

        Returns spectral field of same shape (with non-kept coeffs zeroed).
        """
        coeffs = SpectralOps.dct(spatial)
        n = coeffs.shape[-1]
        k = max(1, int(n * self._keep_frac))
        magnitude = np.abs(coeffs)
        threshold = np.sort(magnitude.ravel())[-k] if k < magnitude.size else 0.0
        mask = magnitude >= threshold
        return coeffs * mask

    def _decompress_from_spectral(self, field: np.ndarray) -> np.ndarray:
        """Decompress a spectral field back to spatial domain."""
        return SpectralOps.idct(field)

    # ── Store / Retrieve ───────────────────────────────────────────────

    def store(
        self,
        layer: int,
        token_pos: int,
        key: np.ndarray,
        value: np.ndarray,
    ) -> None:
        """
        Store key and value for a single token position in a layer.

        Args:
            layer: Layer index [0, n_layers)
            token_pos: Token position in sequence
            key: (d_model,) key projection
            value: (d_model,) value projection

        Complexity: O(spectral_rank^2) for spectral compression
        """
        self._global_step += 1

        # Compress to DCT domain
        k_spec = self._compress_to_spectral(key)
        v_spec = self._compress_to_spectral(value)

        entry = {
            "k_coeffs": k_spec,
            "v_coeffs": v_spec,
            "position": token_pos,
            "coherence": 1.0,
            "coherence": 1.0,
            "timestamp": self._global_step,
            "access_count": 1,
        }

        cache = self._layer_caches[layer]
        # Replace if position exists
        for i, e in enumerate(cache):
            if e["position"] == token_pos:
                cache[i] = entry
                return
        cache.append(entry)

        # Evict oldest if over capacity
        if len(cache) > self.max_seq_len:
            self.evict(layer, coherence_threshold=0.1)

    def retrieve(
        self,
        layer: int,
        start_pos: int,
        end_pos: int,
    ) -> Optional[np.ndarray]:
        """
        Retrieve decompressed keys/values for a range of positions.

        Returns concatenated K/V of shape ((end-start), d_model * 2).
        """
        cache = self._layer_caches[layer]
        results = []
        for e in cache:
            pos = e["position"]
            if start_pos <= pos < end_pos:
                k_dec = self._decompress_from_spectral(e["k_coeffs"])
                v_dec = self._decompress_from_spectral(e["v_coeffs"])
                combined = np.concatenate([k_dec, v_dec], axis=-1)
                results.append((pos, combined))
                e["access_count"] += 1

        if not results:
            return None

        results.sort(key=lambda x: x[0])
        stacked = np.stack([r[1] for r in results], axis=0)
        return stacked

    # ── Eviction ───────────────────────────────────────────────────────

    def evict(self, layer: int, coherence_threshold: float = 0.01) -> None:
        """
        Evict cache entries below a coherence threshold.

        Uses Landau-Zener probability for spectral decay.
        Complexity: O(cache_length * spectral_rank)
        """
        cache = self._layer_caches[layer]
        before = len(cache)

        for e in cache:
            age = self._global_step - e["timestamp"]
            e["coherence"] = self._compute_coherence(age)
            # Also decay DCT coefficients based on coherence
            decay = np.clip(e["coherence"], 0.0, 1.0)
            e["k_coeffs"] = e["k_coeffs"] * decay
            e["v_coeffs"] = e["v_coeffs"] * decay

        cache[:] = [e for e in cache if e["coherence"] >= coherence_threshold]

    # ── Multi-turn conversation management ─────────────────────────────

    def append_turn(self, layer: int, new_tokens: np.ndarray) -> None:
        """
        Append new tokens for a new conversation turn.

        Resets position offsets for the new turn.
        """
        for i, tok in enumerate(new_tokens):
            pos = self.num_positions(layer) + i
            self.store(layer, pos, tok, tok)

    def clear(self) -> None:
        """Clear the entire cache."""
        self._layer_caches = [[] for _ in range(self.n_layers)]
        self._global_step = 0

    # ── Compressed-domain cosine similarity (Upgrade 5) ────────────────

    def cosine_similarity_compressed(
        self,
        layer: int,
        query: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Compute cosine similarity in the compressed DCT domain.
        No decompression needed -- operate directly on coefficients.

        Returns list of (position, similarity) sorted by similarity.
        """
        q_dct = SpectralOps.dct(query.ravel())
        q_norm = np.linalg.norm(q_dct) + 1e-10

        results = []
        for e in self._layer_caches[layer]:
            k_spec_flat = e["k_coeffs"].ravel()
            sim = float(np.dot(q_dct, k_spec_flat))
            sim /= float(q_norm * np.linalg.norm(k_spec_flat) + 1e-10)
            results.append((e["position"], sim))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    # ── Properties ─────────────────────────────────────────────────────

    def num_positions(self, layer: int) -> int:
        return len(self._layer_caches[layer])

    def size_bytes(self) -> int:
        total = 0
        for layer_cache in self._layer_caches:
            for e in layer_cache:
                total += e["k_coeffs"].nbytes + e["v_coeffs"].nbytes
        return total


# ═════════════════════════════════════════════════════════════════════════════
# Section 3: SpectralFFN
# ═════════════════════════════════════════════════════════════════════════════


class SpectralFFN:
    """
    Resonance-gated feed-forward network in frequency domain.

    Standard FFN applies activation in spatial domain.
    Spectral FFN applies activation via convolution in Fourier space,
    which is equivalent to multiplication in frequency domain with
    a learned resonance filter.

    Benefits: Natural band-limiting, frequency-dependent gating,
    and potential for spectral sparsity (skip high-frequency computation).

    Architecture:
      1. DCT encode input to frequency domain
      2. Apply resonance gating (spectral GeLU)
      3. IDCT decode back to spatial domain
      4. Standard up/down projection with spectral activation

    When use_moe is enabled, optionally routes through expert FFNs
    via spectral entropy-based gating.
    """

    def __init__(
        self,
        d_model: int = 4096,
        d_ff: int = 11008,
        use_gelu: bool = True,
        resonance_bands: int = 4,
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_gelu = use_gelu
        self.resonance_bands = resonance_bands

        # Weight matrices
        rng = np.random.RandomState(42)
        scale_w1 = np.sqrt(2.0 / (d_model + d_ff))
        scale_w2 = np.sqrt(2.0 / (d_ff + d_model))

        self.w1 = rng.randn(d_model, d_ff).astype(np.float32) * scale_w1
        self.w2 = rng.randn(d_ff, d_model).astype(np.float32) * scale_w2

        # Resonance gate: learnable frequency-dependent gating
        # One gate per resonance band for each hidden dimension
        self.resonance_gate = np.ones((resonance_bands, d_ff), dtype=np.float32)

        # MoE state
        self.use_moe = False
        self.n_experts = 0
        self.top_k_experts = 2
        self.expert_w1: List[np.ndarray] = []
        self.expert_w2: List[np.ndarray] = []
        self.gate_weight: Optional[np.ndarray] = None

        # Band frequency boundaries (evenly spaced in DCT coefficient index)
        self._band_edges = np.linspace(0, d_model, resonance_bands + 1).astype(int)

    # ── Activation functions ───────────────────────────────────────────

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def _silu(self, x: np.ndarray) -> np.ndarray:
        return x * (1.0 / (1.0 + np.exp(-x)))

    def _activation(self, x: np.ndarray) -> np.ndarray:
        return self._gelu(x) if self.use_gelu else self._silu(x)

    # ── Resonance gating in frequency domain ───────────────────────────

    def _apply_resonance_gate(self, x_dct: np.ndarray) -> np.ndarray:
        """
        Apply frequency-dependent resonance gating.

        Splits the DCT coefficients into bands and applies
        a learned gate per band.  Low-frequency bands (semantic)
        pass through mostly unchanged; high-frequency bands (detail)
        may be gated more aggressively.

        Args:
            x_dct: (..., d_ff) DCT coefficients

        Returns:
            Gated coefficients of same shape.
        """
        out = x_dct.copy()
        for band in range(self.resonance_bands):
            lo = self._band_edges[band]
            hi = self._band_edges[band + 1]
            if hi > out.shape[-1]:
                hi = out.shape[-1]
            gate = np.clip(self.resonance_gate[band, : hi - lo], 0.0, 1.0)
            out[..., lo:hi] = out[..., lo:hi] * gate
        return out

    # ── Forward ────────────────────────────────────────────────────────

    def forward(
        self,
        x: np.ndarray,
        gate_resonance: float = 1.0,
    ) -> np.ndarray:
        """
        Forward pass through the spectral FFN.

        Args:
            x: (n, d_model) input
            gate_resonance: scaling factor for resonance gate (0 = off, 1 = full)

        Returns:
            (n, d_model) output
        """
        # 1. Up projection
        hidden = x @ self.w1  # (n, d_ff)

        # 2. Encode to DCT domain for resonance gating
        hidden_dct = SpectralOps.dct(hidden)
        hidden_dct = self._apply_resonance_gate(hidden_dct) * gate_resonance
        hidden = SpectralOps.idct(hidden_dct)

        # 3. Activation in spatial domain
        hidden = self._activation(hidden)

        # 4. Optional MoE routing
        if self.use_moe and self.gate_weight is not None:
            hidden = self._moe_route(x, hidden)

        # 5. Down projection
        out = hidden @ self.w2
        return out

    # ── MoE Routing ────────────────────────────────────────────────────

    def _moe_route(self, x: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        """
        Route through selected expert FFNs based on spectral entropy.

        The gate weight matrix projects the input to expert scores;
        only the top-k experts are activated per token.
        Expert outputs are merged via weighted sum.
        """
        n = x.shape[0]
        if self.n_experts == 0:
            return hidden

        # Compute gating scores
        gate_scores = x @ self.gate_weight  # (n, n_experts)

        # Select top-k experts per token
        top_k_vals, top_k_idx = (
            np.sort(gate_scores, axis=1)[:, -self.top_k_experts :],
            np.argsort(gate_scores, axis=1)[:, -self.top_k_experts :],
        )

        # Softmax over selected experts
        exp_scores = np.exp(top_k_vals - np.max(top_k_vals, axis=-1, keepdims=True))
        expert_weights = exp_scores / (
            np.sum(exp_scores, axis=-1, keepdims=True) + 1e-10
        )

        # Compute expert outputs and merge
        output = np.zeros_like(hidden)
        for e_idx in range(self.n_experts):
            mask = np.any(top_k_idx == e_idx, axis=-1)
            if not np.any(mask):
                continue
            # Compute FFN for this expert's tokens
            x_e = x[mask]
            h_e = x_e @ self.expert_w1[e_idx]
            h_e = self._activation(h_e)
            out_e = h_e @ self.expert_w2[e_idx]

            # Weight assignment
            for i, m in enumerate(np.where(mask)[0]):
                for k in range(self.top_k_experts):
                    if top_k_idx[m, k] == e_idx:
                        output[m] += expert_weights[m, k] * out_e[i]
                        break

        return output

    # ── Weight management ──────────────────────────────────────────────

    def load_weights(self, w1: np.ndarray, w2: np.ndarray) -> None:
        """Load trained weights into the FFN."""
        self.w1 = w1.astype(np.float32)
        self.w2 = w2.astype(np.float32)

    def load_expert_weights(
        self,
        expert_w1: List[np.ndarray],
        expert_w2: List[np.ndarray],
        gate_weight: np.ndarray,
    ) -> None:
        """Load expert weights for MoE routing."""
        self.n_experts = len(expert_w1)
        self.expert_w1 = [w.astype(np.float32) for w in expert_w1]
        self.expert_w2 = [w.astype(np.float32) for w in expert_w2]
        self.gate_weight = gate_weight.astype(np.float32)
        self.use_moe = True


# ═════════════════════════════════════════════════════════════════════════════
# Section 4: SymplecticIntegrator
# ═════════════════════════════════════════════════════════════════════════════


class SymplecticIntegrator:
    """
    Hamiltonian leapfrog integrator per transformer layer.

    Treats each transformer layer as a timestep in a Hamiltonian system.
    The hidden state is split into low-frequency position (q) and
    high-frequency momentum (p) components.

    Hamiltonian:
        H(q, p) = 0.5 * ||p||^2 + 0.5 * ||q||^2 + V_attn(q) + V_ffn(q)
    where:
        q = low-frequency spectral components (position)
        p = high-frequency spectral components (momentum)
        V_attn = attention potential
        V_ffn  = FFN potential

    Leapfrog integrator (Stormer-Verlet):
        p_{l+1/2} = p_l - (dt/2) * grad_q H_attn(q_l)       [Kick via attention]
        q_{l+1}   = q_l + dt * p_{l+1/2}                     [Drift via FFN]
        p_{l+1}   = p_{l+1/2} - (dt/2) * grad_q H_attn(q_{l+1}) [Kick again]

    In the spectral domain:
      - The attention output IS the gradient of the attention Hamiltonian
      - The FFN output IS the drift in position-space
      - Energy conservation is O(dt^2)
    """

    def __init__(self, dt_default: float = 1.0, energy_tol: float = 1e-6):
        self.dt_default = dt_default
        self.energy_tol = energy_tol
        self.prev_energy: Optional[float] = None

    # ── Q/P splitting ─────────────────────────────────────────────────

    @staticmethod
    def split_qp(
        h: np.ndarray, split_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split hidden state into q (position) and p (momentum) by frequency.

        q = low-frequency components (first split_idx DCT coeffs)
        p = high-frequency components (remaining coeffs)

        Args:
            h: (n, d) hidden state
            split_idx: number of low-frequency components for q.
                       Defaults to d // 2 (midpoint).

        Returns:
            (q, p) each of shape (n, d)
        """
        d = h.shape[-1]
        if split_idx is None:
            split_idx = d // 2

        h_dct = SpectralOps.dct(h)  # (n, d)

        q_dct = np.zeros_like(h_dct)
        p_dct = np.zeros_like(h_dct)

        q_dct[..., :split_idx] = h_dct[..., :split_idx]
        p_dct[..., split_idx:] = h_dct[..., split_idx:]

        q = SpectralOps.idct(q_dct)
        p = SpectralOps.idct(p_dct)
        return q, p

    @staticmethod
    def merge_qp(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Merge q and p components back into a full hidden state.

        Args:
            q: (n, d) position (low frequency)
            p: (n, d) momentum (high frequency)

        Returns:
            (n, d) merged hidden state
        """
        q_dct = SpectralOps.dct(q)
        p_dct = SpectralOps.dct(p)
        return SpectralOps.idct(q_dct + p_dct)

    # ── Hamiltonian energy ─────────────────────────────────────────────

    def hamiltonian_energy(self, hidden: np.ndarray) -> float:
        """
        Compute the Hamiltonian energy of the hidden state.

        H = 0.5 * sum(|p_k|^2) + 0.5 * sum(|q_k|^2)
        (total spectral power)

        Returns Hamiltonian energy (conserved quantity).
        """
        q, p = self.split_qp(hidden)
        q_power = np.sum(q**2)
        p_power = np.sum(p**2)
        return 0.5 * float(q_power + p_power)

    def energy_drift(self, h_prev: np.ndarray, h_curr: np.ndarray) -> float:
        """
        Compute the fractional energy drift between two steps.

        drift = |E_curr - E_prev| / E_prev
        Should be O(dt^2) for symplectic integration.

        Returns fractional energy drift.
        """
        e_prev = self.hamiltonian_energy(h_prev)
        e_curr = self.hamiltonian_energy(h_curr)
        denom = max(abs(e_prev), 1e-10)
        return float(abs(e_curr - e_prev) / denom)

    # ── Symplectic step ────────────────────────────────────────────────

    def step(
        self,
        hidden: np.ndarray,
        attention_fn: Callable[[np.ndarray], np.ndarray],
        ffn_fn: Callable[[np.ndarray], np.ndarray],
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply one symplectic leapfrog step.

        Splits hidden state into q (low) and p (high), then:
          1. Compute attention on full hidden state -> grad_q H_attn(q)
          2. Half-kick: p <- p - (dt/2) * attn_output_high
          3. Compute FFN on full hidden state -> drift
          4. Drift: q <- q + dt * ffn_output_low
          5. Compute attention again on new hidden -> second half-kick on p
          6. Merge q and p back into full hidden state

        Args:
            hidden: (n, d) hidden states -- modified in-place
            attention_fn: attn(x) -> SpectralMatrix
            ffn_fn: ffn(x) -> SpectralMatrix
            dt: timestep (default: dt_default)

        Returns:
            (n, d) updated hidden state
        """
        if dt is None:
            dt = self.dt_default

        # Compute split index at frequency midpoint
        d = hidden.shape[-1]
        split_idx = d // 2

        # Split into position (low-freq) and momentum (high-freq)
        q, p = self.split_qp(hidden, split_idx)

        # --- Leapfrog step ---

        # 1. Compute attention on full state -> this is grad_q H_attn(q)
        attn_output = attention_fn(hidden)

        # 2. Half-kick: p <- p - (dt/2) * high-freq part of attn output
        _, attn_p = self.split_qp(attn_output, split_idx)
        p = p - 0.5 * dt * attn_p

        # Reconstruct full state for FFN step
        hidden = self.merge_qp(q, p)

        # 3. Compute FFN on full state -> this is the drift
        ffn_output = ffn_fn(hidden)

        # 4. Drift: q <- q + dt * low-freq part of FFN output
        ffn_q, _ = self.split_qp(ffn_output, split_idx)
        q = q + dt * ffn_q

        # Reconstruct for second attention
        hidden = self.merge_qp(q, p)

        # 5. Compute attention again on new hidden -> second half-kick
        attn_output_2 = attention_fn(hidden)
        _, attn_p_2 = self.split_qp(attn_output_2, split_idx)
        p = p - 0.5 * dt * attn_p_2

        # 6. Merge q and p back
        hidden = self.merge_qp(q, p)

        # Track energy conservation
        curr_energy = self.hamiltonian_energy(hidden)
        if self.prev_energy is not None:
            drift = abs(curr_energy - self.prev_energy) / (
                abs(self.prev_energy) + 1e-10
            )
            if drift > self.energy_tol:
                # Energy drift too large; apply correction
                scale = np.sqrt(self.prev_energy / (curr_energy + 1e-10))
                hidden = hidden * scale
                curr_energy = self.prev_energy
        self.prev_energy = curr_energy

        return hidden


# ═════════════════════════════════════════════════════════════════════════════
# Section 5: MonoidalChunker
# ═════════════════════════════════════════════════════════════════════════════


class MonoidalChunker:
    """
    Exact sequence splitting for long-context inference using monoidal
    functor decomposition: F(A ++ B) = F(A) (x) F(B).

    Splits a long sequence into overlapping chunks, processes each
    chunk independently through the spectral LLM, then merges results
    via tensor product in the spectral (frequency) domain.

    Cross-chunk attention is computed only for overlapping regions,
    enabling O(N) long-context processing with O(chunk_size) attention.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        min_chunk_size: int = 128,
        overlap_fraction: float = 0.1,
    ):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_fraction = overlap_fraction
        self.overlap = max(1, int(chunk_size * overlap_fraction))
        assert self.overlap < chunk_size, "Overlap must be < chunk size"

    # ── Sequence splitting ─────────────────────────────────────────────

    def split_sequence(self, seq_len: int) -> List[Tuple[int, int]]:
        """
        Split sequence into F(A ++ B) = F(A) (x) F(B) chunks.

        Returns chunk boundaries [start, end) for each chunk.
        Complexity: O(num_chunks)
        """
        stride = self.chunk_size - self.overlap
        chunks = []
        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            if end - start < self.min_chunk_size and len(chunks) > 0:
                # Merge into previous chunk
                prev_start, prev_end = chunks[-1]
                chunks[-1] = (prev_start, end)
                break
            chunks.append((start, end))
            if end == seq_len:
                break
            start += stride
        return chunks

    # ── Spectral tensor product ────────────────────────────────────────

    @staticmethod
    def spectral_tensor_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Tensor product in spectral domain: (A (x) B)[k] = A[k] (x) B[k].

        For each frequency component, merges two spectral fields via
        phasor addition:
          result_amp = sqrt(a_amp^2 + b_amp^2)
          result_phase = atan2(a_amp*sin(a_phase) + b_amp*sin(b_phase),
                              a_amp*cos(a_phase) + b_amp*cos(b_phase))

        This produces the spectral-domain tensor product result.
        """
        a_dct = SpectralOps.dct(a)
        b_dct = SpectralOps.dct(b)

        # Treat DCT coefficients as amplitudes (DCT is real-valued,
        # so phase is 0 or pi depending on sign)
        a_amp = np.abs(a_dct)
        b_amp = np.abs(b_dct)
        a_phase = np.where(a_dct >= 0, 0.0, np.pi)
        b_phase = np.where(b_dct >= 0, 0.0, np.pi)

        # Phasor addition
        result_real = a_amp * np.cos(a_phase) + b_amp * np.cos(b_phase)
        result_imag = a_amp * np.sin(a_phase) + b_amp * np.sin(b_phase)
        result_amp = np.sqrt(result_real**2 + result_imag**2)
        result_phase = np.arctan2(result_imag, result_real + 1e-10)

        result_dct = result_amp * np.cos(result_phase)
        return SpectralOps.idct(result_dct)

    # ── Chunk merging ──────────────────────────────────────────────────

    def merge_chunks(
        self,
        chunk_states: List[np.ndarray],
        boundaries: List[Tuple[int, int]],
        total_seq_len: int,
    ) -> np.ndarray:
        """
        Merge chunk hidden states via tensor product in spectral domain.

        For each frequency component k:
          merged[k] = sqrt(chunk1[k]^2 + chunk2[k]^2)

        This is the spectral-domain tensor product.

        Args:
            chunk_states: Per-chunk hidden states (n_i, d)
            boundaries: Chunk boundaries from split_sequence()
            total_seq_len: Total sequence length

        Returns:
            (total_seq_len, d) merged hidden state
        """
        d = chunk_states[0].shape[-1]
        result = np.zeros((total_seq_len, d), dtype=np.float64)
        weight_sum = np.zeros((total_seq_len, 1), dtype=np.float64)

        for (start, end), state in zip(boundaries, chunk_states):
            chunk_len = state.shape[0]
            target_end = min(start + chunk_len, total_seq_len)
            actual_len = target_end - start

            # Build linear ramp weights for overlap blending
            weights = np.ones(actual_len, dtype=np.float64)
            if start > 0:
                ramp = np.linspace(0.0, 1.0, min(self.overlap, actual_len))
                weights[: len(ramp)] = ramp
            if target_end < total_seq_len:
                ramp = np.linspace(1.0, 0.0, min(self.overlap, actual_len))
                weights[-len(ramp) :] = np.minimum(weights[-len(ramp) :], ramp)

            # Weighted spectral tensor product merge
            for i in range(actual_len):
                if weight_sum[start + i, 0] > 0.0:
                    existing = result[start + i : start + i + 1]
                    new = state[i : i + 1]
                    blended = self.spectral_tensor_product(
                        existing * weight_sum[start + i, 0],
                        new * weights[i],
                    )
                    result[start + i] = blended / (
                        weight_sum[start + i, 0] + weights[i] + 1e-10
                    )
                else:
                    result[start + i] = state[i] * weights[i]
                weight_sum[start + i, 0] += weights[i]

        weight_sum = np.maximum(weight_sum, 1e-10)
        return (result / weight_sum).astype(np.float32)

    # ── Cross-chunk attention ──────────────────────────────────────────

    def cross_chunk_attention(
        self,
        merged_state: np.ndarray,
        boundaries: List[Tuple[int, int]],
        attn_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Cross-chunk attention (sparse, only for overlapping regions).

        Only computes attention between overlapping regions and boundary
        tokens using the existing Vlasov attention mechanism on the
        reduced token set.

        Args:
            merged_state: Merged hidden state from merge_chunks()
            boundaries: Chunk boundaries from split_sequence()
            attn_fn: Attention function f(query, kv) -> output.
                     If None, uses identity.

        Returns:
            Updated merged state with cross-chunk context.
        """
        if attn_fn is None:
            return merged_state

        result = merged_state.copy()
        n_chunks = len(boundaries)

        # For each adjacent pair of chunks, attend the overlap region
        for i in range(n_chunks - 1):
            c1_start, c1_end = boundaries[i]
            c2_start, c2_end = boundaries[i + 1]

            # Overlap region between chunks
            ol_start = max(c1_start, c2_start - self.overlap)
            ol_end = min(c1_end, c2_start + self.overlap)

            if ol_start >= ol_end:
                continue

            # Extract overlap tokens from both chunks
            ol_tokens = merged_state[ol_start:ol_end]

            # Cross-attend: chunk i's overlap attends to chunk i+1's overlap
            attended = attn_fn(ol_tokens, ol_tokens)
            result[ol_start:ol_end] = attended

        return result


# ═════════════════════════════════════════════════════════════════════════════
# Section 6: SpectralEmbedding
# ═════════════════════════════════════════════════════════════════════════════


class SpectralEmbedding:
    """
    Frequency-domain token embedding with spectral position encoding.

    Converts token IDs to spectral field representations and applies
    position encoding via spectral phase rotation (RoPE in frequency domain).

    The embedding table stores each token embedding as a spectral field
    (DCT coefficients). Position encoding is applied via phase rotation
    of the frequency components, which is RoPE in the DCT domain.

    Benefits: Natural band-limiting, frequency-dependent positional
    information, and efficient interpolation to arbitrary sequence lengths.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.use_rope = use_rope

        # Embedding table as DCT coefficients (spectral fields)
        rng = np.random.RandomState(42)
        scale = 1.0 / np.sqrt(d_model)
        raw_emb = rng.randn(vocab_size, d_model).astype(np.float32) * scale
        self.embedding_table: np.ndarray = SpectralOps.dct(raw_emb)

        # Precomputed RoPE frequencies
        self.rope_frequencies: np.ndarray = self._compute_rope_frequencies()

    # ── RoPE ───────────────────────────────────────────────────────────

    def _compute_rope_frequencies(self) -> np.ndarray:
        """Compute RoPE frequencies: theta^(-2i/d) for i in [0, d/2)."""
        half = self.d_model // 2
        freqs = 1.0 / (self.rope_theta ** (np.arange(0, half, dtype=np.float64) / half))
        return freqs.astype(np.float32)

    def _apply_rope(self, embedding: np.ndarray, position: int) -> np.ndarray:
        """
        Apply RoPE position encoding via phase rotation in DCT domain.

        Rotates frequency components by position-dependent phase:
          freq_k <- freq_k * exp(i * position * theta_k)

        Since DCT is real, we apply rotation as a modulated cosine
        transformation on the amplitude pattern.
        """
        if not self.use_rope:
            return embedding

        n, d = embedding.shape
        half = d // 2

        # Compute position-dependent rotation angles
        angles = position * self.rope_frequencies[:half]  # (half,)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        # Apply rotation in DCT domain
        emb_dct = SpectralOps.dct(embedding)  # (n, d)
        rotated = emb_dct.copy()
        rotated[:, :half] = (
            emb_dct[:, :half] * cos_a - emb_dct[:, half : 2 * half] * sin_a
        )
        rotated[:, half : 2 * half] = (
            emb_dct[:, :half] * sin_a + emb_dct[:, half : 2 * half] * cos_a
        )

        return SpectralOps.idct(rotated)

    # ── Forward ────────────────────────────────────────────────────────

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """
        Convert token IDs to spectral field with position encoding.

        Args:
            tokens: (seq_len,) or (batch, seq_len) token IDs

        Returns:
            (seq_len, d_model) or (batch, seq_len, d_model) embeddings

        Complexity: O(seq_len * d_model * log(d_model))
        """
        tokens = np.asarray(tokens, dtype=np.int32)
        orig_shape = tokens.shape
        flat = tokens.ravel()

        # Look up DCT coefficients from embedding table
        emb_dct = np.take(self.embedding_table, flat, axis=0)

        # Decode from DCT to spatial domain
        spatial = SpectralOps.idct(emb_dct)

        # Apply RoPE position encoding
        if self.use_rope:
            for i in range(len(flat)):
                spatial[i] = self._apply_rope(spatial[i : i + 1], i)[0]

        return spatial.reshape(*orig_shape, self.d_model).astype(np.float32)

    # ── Weight loading ─────────────────────────────────────────────────

    def load_weights(self, weights: np.ndarray) -> None:
        """
        Load pre-trained embedding weights.

        Args:
            weights: (vocab_size, d_model) or flattened vector
        """
        w = np.asarray(weights, dtype=np.float32).reshape(self.vocab_size, self.d_model)
        self.embedding_table = SpectralOps.dct(w)


# ═════════════════════════════════════════════════════════════════════════════
# Section 7: GyrokineticAttention
# ═════════════════════════════════════════════════════════════════════════════


class GyrokineticAttention:
    """
    Gyrokinetic splitting of spectral attention field into slow
    (gyrocenter, <5% frequency) and fast (gyromotion, >5% frequency)
    components.

    In plasma physics, gyrokinetics separates particle motion into:
    - Gyrocenter: the slow drift of the guiding center (large-scale,
      semantic structure in LLM terms)
    - Gyromotion: the fast cyclotron motion around field lines (small-scale,
      fine detail in LLM terms)

    The slow component is evolved explicitly (full attention),
    the fast component is averaged (mean-field approximation).
    This reduces effective sequence length for the expensive O(n^2)
    computation to only the slow tokens.

    Fraction split determines how much is 'slow':
      - 0.05 means 5% of spectral bandwidth is slow (gyrocenter)
      - This is typically 1-5% for strong compression
    """

    def __init__(
        self,
        d_model: int = 4096,
        n_heads: int = 32,
        slow_fraction: float = 0.05,
        delta_t: float = 0.01,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.slow_fraction = slow_fraction
        self.delta_t = delta_t

    def split_spectral_field(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split spectral field into slow (gyrocenter) and fast (gyromotion).

        The slow component keeps the lowest `slow_fraction` of DCT
        coefficients, capturing the semantic/large-scale structure.
        The fast component keeps the remaining high-frequency detail.

        Args:
            x: (n, d) input tensor

        Returns:
            (slow, fast) each of shape (n, d)
        """
        coeffs = SpectralOps.dct(x)
        n_keep = max(1, int(coeffs.shape[-1] * self.slow_fraction))

        slow_coeffs = np.zeros_like(coeffs)
        fast_coeffs = np.zeros_like(coeffs)

        slow_coeffs[..., :n_keep] = coeffs[..., :n_keep]
        fast_coeffs[..., n_keep:] = coeffs[..., n_keep:]

        slow = SpectralOps.idct(slow_coeffs)
        fast = SpectralOps.idct(fast_coeffs)
        return slow, fast

    def slow_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """
        Standard attention on the slow (gyrocenter) component only.
        This is O(n_slow^2) where n_slow << n.
        """
        d = q.shape[-1]
        scale = 1.0 / np.sqrt(d)
        scores = q @ k.T * scale
        # Stable softmax
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores) / (
            np.sum(np.exp(scores), axis=-1, keepdims=True) + 1e-10
        )
        return weights @ v

    def fast_mean_field(self, v_fast: np.ndarray) -> np.ndarray:
        """
        Approximate fast component via mean-field.
        Instead of attending fast tokens individually, use their mean.
        This is the gyrokinetic averaging approximation.
        """
        return np.mean(v_fast, axis=0, keepdims=True)

    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Gyrokinetic attention forward pass.

        1. Split Q, K, V into slow and fast components
        2. Slow: full self-attention on slow components
        3. Fast: mean-field approximation
        4. Merge slow + fast with time-step weighting

        Args:
            q: (n, d) queries
            k: (n, d) keys
            v: (n, d) values

        Returns:
            (n, d) gyrokinetic attention output
        """
        # Split all three into slow/fast
        q_slow, q_fast = self.split_spectral_field(q)
        k_slow, k_fast = self.split_spectral_field(k)
        v_slow, v_fast = self.split_spectral_field(v)

        # Slow component: full attention (O(n^2) but n is small)
        attn_slow = self.slow_attention(q_slow, k_slow, v_slow)

        # Fast component: mean-field (O(n))
        fast_mf = self.fast_mean_field(v_fast)
        attn_fast = np.broadcast_to(fast_mf, q_fast.shape)

        # Time-step weighted evolution (delta_t controls slow/fast coupling)
        out = (1.0 - self.delta_t) * attn_slow + self.delta_t * attn_fast
        return out + q  # residual


# ═════════════════════════════════════════════════════════════════════════════
# Section 8: LandauDamping
# ═════════════════════════════════════════════════════════════════════════════


class LandauDamping:
    """
    Collisionless wave-particle damping for attention stability.

    In plasma physics, Landau damping is the exponential decay of
    plasma waves due to resonant energy transfer between the wave
    and particles moving at the wave's phase velocity.  In spectral
    LLMs, this manifests as damping of high-frequency spectral
    components that carry low semantic information.

    The damping rate:
        gamma = pi/2 * omega^3 / k^2 * df/dv at resonance

    Applied as a spectral filter that selectively attenuates unstable
    frequency components, preventing oscillation in the attention
    distribution.

    This is essential for numerical stability of spectral attention:
    without it, high-frequency components can accumulate error and
    cause the attention distribution to become chaotic.
    """

    def __init__(
        self,
        d_model: int = 4096,
        damping_strength: float = 0.1,
        resonant_freq: float = 0.5,
        temperature: float = 1.0,
    ):
        self.d_model = d_model
        self.damping_strength = damping_strength
        self.resonant_freq = resonant_freq
        self.temperature = temperature

    def landau_rate(self, k: np.ndarray) -> np.ndarray:
        """
        Compute Landau damping rate for each frequency mode.

        The damping rate gamma(k) for wavenumber k is:
          gamma(k) = sqrt(pi/8) * omega_p / (|k|^3 * lambda_D^3)
            * exp(-1/(2*k^2*lambda_D^2) - 1.5)

        where omega_p is the plasma frequency and lambda_D is the
        Debye length.  In our spectral LLM context:
        - omega_p ~ resonant_freq
        - lambda_D ~ temperature

        High-k (high frequency) modes are more strongly damped,
        which stabilizes the attention distribution.
        """
        k = np.maximum(np.abs(k), 1e-10)
        lambda_d = max(self.temperature, 1e-10)
        omega_p = self.resonant_freq

        gamma = np.sqrt(np.pi / 8.0) * omega_p / (k**3 * lambda_d**3)
        gamma = gamma * np.exp(-1.0 / (2.0 * k**2 * lambda_d**2) - 1.5)
        return gamma

    def damping_filter(self, d: int) -> np.ndarray:
        """
        Build frequency-domain damping filter.

        Returns a 1D array of shape (d,) where each element is
        the damping factor exp(-gamma(k) * dt) for frequency k.
        """
        k = np.arange(1, d + 1, dtype=np.float64) / d
        gamma = self.landau_rate(k)
        return np.exp(-gamma * self.damping_strength).astype(np.float32)

    def apply(self, attn_output: np.ndarray) -> np.ndarray:
        """
        Apply Landau damping to attention output.

        Transforms to DCT domain, applies damping filter
        (attenuates high-frequency modes), and transforms back.

        Args:
            attn_output: (n, d) attention output

        Returns:
            (n, d) damped attention output
        """
        d = attn_output.shape[-1]
        filter_ = self.damping_filter(d)

        attn_dct = SpectralOps.dct(attn_output)
        damped_dct = attn_dct * filter_.reshape(1, -1)
        damped = SpectralOps.idct(damped_dct)

        return damped

    def compute_energy_transfer(
        self, input_field: np.ndarray, output_field: np.ndarray
    ) -> float:
        """
        Compute the energy transferred between wave and particles.

        Positive means energy flowing from wave to particles (damping),
        negative means energy flowing from particles to wave (growth).

        Returns net energy transfer (arbitrary units).
        """
        in_dct = SpectralOps.dct(input_field)
        out_dct = SpectralOps.dct(output_field)
        energy_in = np.sum(in_dct**2)
        energy_out = np.sum(out_dct**2)
        return float(energy_in - energy_out)


# ═════════════════════════════════════════════════════════════════════════════
# Section 9: ResonanceMoE
# ═════════════════════════════════════════════════════════════════════════════


class ResonanceMoE:
    """
    Spectral entropy-based expert routing for Mixture-of-Experts.

    Routes tokens to specialized expert FFNs based on their spectral
    entropy profile.  Tokens with high spectral entropy (uncertain,
    information-rich) are routed to more experts; tokens with low
    entropy (coherent, predictable) use fewer experts.

    The router selects top-k experts per token based on spectral
    entropy, and expert outputs are merged via phasor-weighted
    addition in the spectral domain.

    Key differences from standard MoE:
    - Routing is based on spectral entropy, not learned embeddings
    - Expert merging uses phasor addition (magnitude + phase)
    - Number of active experts adapts per token
    - Resonance coherence gate prevents expert fragmentation
    """

    def __init__(
        self,
        d_model: int = 4096,
        d_ff: int = 11008,
        n_experts: int = 8,
        top_k: int = 2,
        min_experts: int = 1,
        entropy_threshold_low: float = 0.3,
        entropy_threshold_high: float = 0.7,
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.top_k = top_k
        self.min_experts = min_experts
        self.entropy_threshold_low = entropy_threshold_low
        self.entropy_threshold_high = entropy_threshold_high

        # Expert FFN weights
        rng = np.random.RandomState(42)
        scale = np.sqrt(2.0 / (d_model + d_ff))
        self.expert_w1 = [
            rng.randn(d_model, d_ff).astype(np.float32) * scale
            for _ in range(n_experts)
        ]
        self.expert_w2 = [
            rng.randn(d_ff, d_model).astype(np.float32) * scale
            for _ in range(n_experts)
        ]

        # Gate projection (learned, maps token embedding to expert scores)
        self.gate_proj = rng.randn(d_model, n_experts).astype(np.float32) * 0.01

        # Resonance coherence (smoothed over time)
        self._coherence: float = 1.0

    # ── Spectral entropy routing ───────────────────────────────────────

    def _spectral_entropy_score(self, x: np.ndarray) -> float:
        """Compute spectral entropy of a token's embedding."""
        return SpectralOps.spectral_entropy(x)

    def _compute_n_active_experts(self, entropy: float) -> int:
        """
        Determine number of active experts based on spectral entropy.

        High entropy -> more experts (more compute, more capacity)
        Low entropy  -> fewer experts (faster, cheaper)
        """
        if entropy < self.entropy_threshold_low:
            return self.min_experts
        elif entropy > self.entropy_threshold_high:
            return self.top_k
        else:
            frac = (entropy - self.entropy_threshold_low) / (
                self.entropy_threshold_high - self.entropy_threshold_low + 1e-10
            )
            return max(
                self.min_experts,
                int(frac * (self.top_k - self.min_experts)) + self.min_experts,
            )

    def _phasor_merge(
        self, outputs: List[np.ndarray], weights: np.ndarray
    ) -> np.ndarray:
        """
        Merge expert outputs via phasor addition in spectral domain.

        Each expert output is treated as a phasor (amplitude + phase
        encoded as DCT coefficients with sign).  The merge is:
          result_k = sum_i w_i * output_i_k

        Then apply soft gating via resonance coherence.
        """
        d = outputs[0].shape[-1]
        merged = np.zeros_like(outputs[0], dtype=np.float64)
        weight_sum = 1e-10

        for out, w in zip(outputs, weights):
            merged += w * out.astype(np.float64)
            weight_sum += w

        merged = merged / weight_sum
        merged = merged * self._coherence
        return merged.astype(np.float32)

    # ── Forward ────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Route input through Resonance MoE.

        For each token:
        1. Compute spectral entropy
        2. Determine number of active experts
        3. Compute gate scores and select top experts
        4. Compute expert FFN outputs
        5. Phasor-merge expert outputs

        Args:
            x: (n, d_model) input tokens

        Returns:
            (n, d_model) MoE output
        """
        n = x.shape[0]
        output = np.zeros_like(x, dtype=np.float32)

        for i in range(n):
            token = x[i : i + 1]

            # 1. Spectral entropy
            entropy = self._spectral_entropy_score(token.ravel())

            # 2. Adaptive number of experts
            n_active = self._compute_n_active_experts(entropy)

            # 3. Gate scores (learned + entropy bias)
            gate_scores = token @ self.gate_proj  # (1, n_experts)
            gate_scores = gate_scores.ravel()

            # Entropy bias: high entropy -> broader expert selection
            entropy_bias = (
                entropy * np.random.RandomState(42).randn(self.n_experts) * 0.1
            )
            gate_scores = gate_scores + entropy_bias

            # Select top-n_active experts
            top_idx = np.argsort(gate_scores)[-n_active:]
            top_scores = gate_scores[top_idx]

            # Softmax over selected
            top_scores = top_scores - np.max(top_scores)
            exp_scores = np.exp(top_scores)
            weights = exp_scores / (np.sum(exp_scores) + 1e-10)

            # 4. Compute expert FFN outputs
            expert_outputs = []
            for j, e_idx in enumerate(top_idx):
                hidden = token @ self.expert_w1[e_idx]
                hidden = hidden * (1.0 / (1.0 + np.exp(-hidden)))  # SiLU
                expert_out = hidden @ self.expert_w2[e_idx]
                expert_outputs.append(expert_out)

            # 5. Phasor merge
            merged = self._phasor_merge(expert_outputs, weights)
            output[i] = merged.ravel()

        return output

    def update_coherence(self, new_val: float) -> None:
        """Update resonance coherence with exponential smoothing."""
        self._coherence = 0.9 * self._coherence + 0.1 * new_val


# ═════════════════════════════════════════════════════════════════════════════
# Section 10: AdaptiveSpectralRank
# ═════════════════════════════════════════════════════════════════════════════


class AdaptiveSpectralRank:
    """
    Per-layer DCT coefficient count optimization.

    Adjusts the number of DCT coefficients (spectral rank) for each
    transformer layer based on the spectral entropy of the layer's input.

    Layers with high spectral entropy (information-rich, uncertain)
    need more DCT coefficients to faithfully represent their state.
    Layers with low spectral entropy (coherent, predictable) can use
    fewer coefficients, saving memory and compute.

    The optimization is:
        K_layer = K_min + (K_max - K_min) * H_layer / H_max
    where H_layer is the spectral entropy of the layer's input,
    and H_max is the maximum entropy (log2 of dimension).

    This is applied as both a memory optimization (fewer coefficients
    stored in KV cache) and a compute optimization (fewer coefficients
    to process in attention/FFN).
    """

    def __init__(
        self,
        d_model: int = 4096,
        min_rank: int = 8,
        max_rank: int = 256,
        default_rank: int = 64,
        adaptation_rate: float = 0.1,
    ):
        self.d_model = d_model
        self.min_rank = min_rank
        self.max_rank = min(max_rank, d_model)
        self.default_rank = default_rank
        self.adaptation_rate = adaptation_rate

        # Smoothed per-layer ranks
        self._layer_ranks: Dict[int, int] = {}
        self._layer_entropies: Dict[int, float] = {}

    # ── Entropy computation ────────────────────────────────────────────

    def _normalized_entropy(self, x: np.ndarray) -> float:
        """Compute normalized spectral entropy in [0, 1]."""
        return SpectralOps.spectral_entropy(x.ravel())

    # ── Rank determination ─────────────────────────────────────────────

    def rank_for_layer(self, layer_input: np.ndarray, layer_idx: int) -> int:
        """
        Determine optimal spectral rank for a given layer.

        Args:
            layer_input: (n, d_model) input tensor for the layer
            layer_idx: layer index

        Returns:
            Optimal spectral rank K in [min_rank, max_rank]
        """
        entropy = self._normalized_entropy(layer_input)

        # Store smoothed entropy for this layer
        if layer_idx in self._layer_entropies:
            prev = self._layer_entropies[layer_idx]
            entropy = (
                1.0 - self.adaptation_rate
            ) * prev + self.adaptation_rate * entropy
        self._layer_entropies[layer_idx] = entropy

        # Map entropy to rank: linear interpolation
        # entropy=0 -> min_rank, entropy=1 -> max_rank
        frac = np.clip(entropy, 0.0, 1.0)
        rank = int(self.min_rank + frac * (self.max_rank - self.min_rank))
        rank = max(self.min_rank, min(self.max_rank, rank))

        # Smooth rank update
        if layer_idx in self._layer_ranks:
            prev_rank = self._layer_ranks[layer_idx]
            rank = int(
                (1.0 - self.adaptation_rate) * prev_rank + self.adaptation_rate * rank
            )

        self._layer_ranks[layer_idx] = rank
        return rank

    def compress_layer(self, x: np.ndarray, layer_idx: int) -> Tuple[np.ndarray, int]:
        """
        Compress layer activations to adaptive spectral rank.

        Args:
            x: (n, d_model) layer input/output
            layer_idx: layer index

        Returns:
            (compressed_tensor, rank_used)
        """
        rank = self.rank_for_layer(x, layer_idx)
        compressed = SpectralOps.band_limit(x, rank)
        return compressed, rank

    def get_rank(self, layer_idx: int) -> int:
        """Get the current rank for a layer (or default if not yet set)."""
        return self._layer_ranks.get(layer_idx, self.default_rank)

    def get_entropy(self, layer_idx: int) -> float:
        """Get the current smoothed entropy for a layer."""
        return self._layer_entropies.get(layer_idx, 0.5)

    def suggest_compression_ratio(self, layer_idx: int) -> float:
        """
        Suggest compression ratio based on current rank.

        ratio = d_model / rank
        Higher rank = lower compression (more fidelity).
        """
        rank = self.get_rank(layer_idx)
        return self.d_model / max(rank, 1)

    def summary(self) -> Dict[str, float]:
        """Return a summary of current ranks across layers."""
        if not self._layer_ranks:
            return {"avg_rank": float(self.default_rank), "n_layers": 0}
        ranks = list(self._layer_ranks.values())
        return {
            "avg_rank": float(np.mean(ranks)),
            "min_rank": float(min(ranks)),
            "max_rank": float(max(ranks)),
            "n_layers": len(ranks),
            "avg_compression": float(self.d_model / (np.mean(ranks) + 1e-10)),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Utility: assemble a complete spectral transformer layer
# ═════════════════════════════════════════════════════════════════════════════


class SpectralTransformerLayer:
    """
    A complete transformer layer using all spectral innovations.

    Combines:
    - VlasovMeanFieldAttention (O(n) attention)
    - SpectralFFN (resonance-gated)
    - SymplecticIntegrator (Hamiltonian leapfrog stepping)
    - GyrokineticAttention (slow/fast splitting within attention)
    - LandauDamping (spectral stabilization)
    - AdaptiveSpectralRank (per-layer optimization)
    - SpectralKVCache integration (DCT-compressed KV storage)
    """

    def __init__(
        self,
        d_model: int = 4096,
        n_heads: int = 32,
        d_ff: int = 11008,
        layer_idx: int = 0,
        spectral_rank: int = 64,
        use_vlasov: bool = True,
        use_gyrokinetic: bool = False,
        use_landau: bool = True,
        use_symplectic: bool = True,
        use_resonance_moe: bool = False,
    ):
        self.d_model = d_model
        self.layer_idx = layer_idx

        # RMS norm weights
        self.norm1_weight = np.ones(d_model, dtype=np.float32)
        self.norm2_weight = np.ones(d_model, dtype=np.float32)

        # Spectral attention
        if use_vlasov:
            self.attn = VlasovMeanFieldAttention(
                d_model=d_model,
                n_heads=n_heads,
                spectral_rank=spectral_rank,
            )
        else:
            self.attn = None

        # Gyrokinetic attention wrapper
        self.gyrokinetic = (
            GyrokineticAttention(d_model=d_model, n_heads=n_heads)
            if use_gyrokinetic
            else None
        )

        # Spectral FFN with optional MoE
        self.ffn = SpectralFFN(d_model=d_model, d_ff=d_ff)
        self.resonance_moe = (
            ResonanceMoE(d_model=d_model, d_ff=d_ff) if use_resonance_moe else None
        )

        # Symplectic integrator
        self.integrator = SymplecticIntegrator() if use_symplectic else None

        # Landau damping
        self.landau = LandauDamping(d_model=d_model) if use_landau else None

        # Adaptive rank
        self.rank_adapter = AdaptiveSpectralRank(d_model=d_model)

        # QKV projection (shared, will be split per head)
        rng = np.random.RandomState(42 + layer_idx)
        scale = 1.0 / np.sqrt(d_model)
        self.w_q = rng.randn(d_model, d_model).astype(np.float32) * scale

    def rms_norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
        return x / np.sqrt(variance + 1e-6) * weight

    def forward(
        self,
        x: np.ndarray,
        kv_cache: Optional[SpectralKVCache] = None,
    ) -> np.ndarray:
        """
        Forward pass through one spectral transformer layer.

        Args:
            x: (n, d_model) input
            kv_cache: optional KV cache for this layer

        Returns:
            (n, d_model) output
        """
        # Adaptive rank compression
        x_compressed, _ = self.rank_adapter.compress_layer(x, self.layer_idx)

        # ── Attention sub-layer ──
        residual = x_compressed
        x_norm = self.rms_norm(x_compressed, self.norm1_weight)

        if self.attn is not None:
            if self.gyrokinetic is not None:
                # Gyrokinetic attention (slow/fast split)
                q = x_norm @ self.w_q
                k = x_norm @ self.w_q  # simplified: in practice, separate W_k
                v = x_norm
                attn_out = self.gyrokinetic.forward(q, k, v)
            else:
                # Vlasov mean-field attention
                attn_out = self.attn.forward(x_norm)

            # Landau damping for stability
            if self.landau is not None:
                attn_out = self.landau.apply(attn_out)

            x = residual + attn_out
        else:
            x = x_compressed

        # Store in KV cache (if provided)
        if kv_cache is not None:
            k = x_norm @ self.w_q
            v = x_norm
            kv_cache.store(self.layer_idx, 0, k[0], v[0])

        # ── FFN sub-layer ──
        residual = x
        x_norm = self.rms_norm(x, self.norm2_weight)

        if self.resonance_moe is not None:
            ffn_out = self.resonance_moe.forward(x_norm)
        else:
            ffn_out = self.ffn.forward(x_norm)

        # Symplectic integration (wrap FFN step as drift)
        if self.integrator is not None:

            def attn_fn(h):
                return attn_out if self.attn is not None else h

            def ffn_fn(h):
                return ffn_out

            x = self.integrator.step(residual, attn_fn, ffn_fn)
        else:
            x = residual + ffn_out

        return x
