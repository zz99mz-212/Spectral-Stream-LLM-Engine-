# Dynamic Tuning for SpectralStream Compression Methods

**Research Document** — June 2026

> **Problem:** Every compression method in SpectralStream currently uses fixed hard-coded parameters (block size, bit width, rank, threshold, etc). Users must manually select a method and accept whatever ratio/error it produces. We need each method to be **dynamically tunable**: given a target compression ratio (or target error), the method auto-tunes its internal parameters to meet the target with minimal loss.

---

## Table of Contents

1. [Mathematical Foundation: Ratio & Error as Functions of Parameters](#1-mathematical-foundation-ratio--error-as-functions-of-parameters)
2. [Decomposition Methods (SVD, TT, CP, Tucker)](#2-decomposition-methods-svd-tt-cp-tucker)
3. [Spectral Methods (DCT, Wavelet, FFT, Hadamard)](#3-spectral-methods-dct-wavelet-fft-hadamard)
4. [Quantization Methods (INT8, INT4, NF4, etc.)](#4-quantization-methods-int8-int4-nf4-etc)
5. [Structural Methods (Toeplitz, BlockDiagonal, Sparsity)](#5-structural-methods-toeplitz-blockdiagonal-sparsity)
6. [Physics/Quantum Methods](#6-physicsquantum-methods)
7. [Entropy / Lossless Methods](#7-entropy--lossless-methods)
8. [Unified Tuning API](#8-unified-tuning-api)
9. [Error-Ratio Pareto Frontier](#9-error-ratio-pareto-frontier)
10. [Cross-Method Optimization](#10-cross-method-optimization)
11. [Implementation Plan](#11-implementation-plan)

---

## 1. Mathematical Foundation: Ratio & Error as Functions of Parameters

Every compression method is a function `C(tensor, θ) → (compressed_bytes, metadata)` governed by a parameter vector `θ`. Two derived functions define tunability:

| Function | Purpose | Form |
|---|---|---|
| `ratio(θ, tensor)` | What compression ratio does θ achieve? | `original_bytes / len(compressed_bytes)` |
| `error(θ, tensor)` | What reconstruction error does θ produce? | `‖tensor - decompress(C(tensor,θ))‖ / ‖tensor‖` |

**The tuning problem:** Given target ratio `R_target` (or target error `E_target`), find `θ` such that `ratio(θ) ≥ R_target` and `error(θ)` is minimized.

**The key insight:** `ratio(θ)` is a **deterministic closed-form function** for most methods. `error(θ)` is a **stochastic function** that depends on the tensor's statistics but follows predictable trends (exponential decay for spectral/decomposition, power law for quantization).

### 1.1 The Tuning Meta-Algorithm

```
For each method:
  1. Compute compression size as function of parameter: size = f_size(param)
  2. Compute ratio as original_size / f_size(param)
  3. Invert: param = f_size^{-1}(original_size / R_target)
  4. Compute error model: error_hat = f_error(param, tensor_stats)
  5. If error_hat > E_target, refine or cascade
```

### 1.2 One-Shot Ratio Prediction

Most methods allow ratio prediction **without compressing** by computing the output byte size analytically:

```python
def predict_ratio(tensor, param) -> float:
    """Predict compression ratio for given parameter without compressing."""
    compressed_size = compute_compressed_size(tensor.shape, tensor.nbytes, param)
    return tensor.nbytes / compressed_size
```

---

## 2. Decomposition Methods (SVD, TT, CP, Tucker)

### 2.1 SVD Truncated

**Current:** Fixed `rank=k`. Ratio is fixed per-rank regardless of tensor content.

**Tuning approach:**

- **Ratio function:** For an `(m, n)` matrix with truncated rank `k`:

  ```
  U: (m, k) × 4 bytes  → 4mk
  S: (k,) × 4 bytes     → 4k
  V: (k, n) × 4 bytes   → 4nk
  Total: 4k(m + n + 1)
  Ratio = (4mn) / (4k(m + n + 1)) = mn / (k(m + n + 1))
  ```

- **Inverse for target ratio R_target:**

  ```
  k = floor(mn / (R_target * (m + n + 1)))
  ```

- **Alternative: energy threshold.** Keep top-k singular values until cumulative energy ≥ threshold τ:

  ```
  S = svd(tensor)
  cumulative = cumsum(S²) / sum(S²)
  k = argmin(cumulative < τ) + 1
  ratio = mn / (k(m + n + 1))        # can be derived from τ
  ```

- **Error model:** For singular values σ₁ ≥ σ₂ ≥ ... ≥ σ_r:

  ```python
  error(k) = sqrt(sum(σ[k:]²) / sum(σ²))   # exact for Frobenius norm
  ```

  This is **analytically exact** (not a heuristic) — the truncation error is the square root of the sum of squares of discarded singular values divided by total.

- **Binary search for k given target ratio:** Since k is integer, use binary search on [1, min(m, n)].

- **Decay rate model:** For most weight matrices, singular values decay roughly exponentially: `σᵢ ≈ exp(-αi)`. The decay rate `α` can be estimated from the first few singular values and used to predict error without full SVD:

  ```python
  def estimate_decay_rate(tensor):
      # Use randomized SVD on first ~20 singular values
      U, S, Vt = randomized_svd(tensor, n_components=20)
      log_S = np.log(S)
      alpha = -np.polyfit(np.arange(len(S)), log_S, 1)[0]
      return max(alpha, 1e-6)
  ```

### 2.2 Tensor Train (TT)

**Current:** Fixed `rank=r`. Ratio depends on (d, m, n, r) for 2D matrices reshaped to d-dimensional tensors.

**Tuning approach:**

- **TT decomposition** reshapes a matrix into a d-dimensional tensor `n₁ × n₂ × ... × n_d` with TT-ranks `r₀, r₁, ..., r_d` (r₀ = r_d = 1).

- **Ratio function** (equal ranks r):

  ```
  Total elements = sum(r_{i-1} * n_i * r_i) for i=1..d
  For uniform rank r: total ≈ r² * sum(n_i)
  Ratio = prod(n_i) * 4 / (r² * sum(n_i) * 4) = prod(n_i) / (r² * sum(n_i))
  ```

- **Inverse for target ratio:**

  ```
  r = floor(sqrt(prod(n_i) / (R_target * sum(n_i))))
  ```

  Clamp to [1, min(n_i)].

- **Error model:** TT truncation error after ALS sweep or TT-SVD is proportional to the sum of discarded singular values across all TT cores. A practical estimate:

  ```python
  def predict_tt_error(tensor, rank):
      # Quick rank-1 approximation error as proxy
      U, S, Vt = randomized_svd(tensor.reshape(tensor.shape[0], -1), n_components=rank+5)
      return sqrt(sum(S[rank:]²) / sum(S²))
  ```

- **Adaptive rank search:** Sweep from r=1 upward, computing ratio each step, stop when ratio ≥ R_target.

### 2.3 CP Decomposition

**Current:** Fixed `rank=r`. 

**Tuning approach:**

- **Ratio function:** For an N-way tensor of shape `(n₁, ..., n_N)` with rank R:

  ```
  Total elements = R * sum(n_i)  (one factor vector per mode per component)
  Ratio = prod(n_i) / (R * sum(n_i))
  ```

- **Inverse:**

  ```
  R = floor(prod(n_i) / (R_target * sum(n_i)))
  ```

- **Error model:** CP error is not analytically tractable without ALS. Use a **sampling-based proxy**: compute CP-ALS for 3-5 candidate ranks, fit a power-law error model `error(R) ≈ a * R^{-b}`.

### 2.4 Tucker Decomposition

**Current:** Fixed core rank `(r₁, ..., r_N)`.

**Tuning approach:**

- **Ratio function:** Core of size `prod(r_i)` plus N factor matrices of size `n_i × r_i`:

  ```
  Total = prod(r_i) + sum(n_i * r_i)
  Ratio = prod(n_i) / (prod(r_i) + sum(n_i * r_i))
  ```

- **Tuning via multilinear SVD (HOSVD):** Compute truncated HOSVD with energy threshold τ on each mode. The resulting core rank per mode is `r_i = argmin(cumsum(S_i²) / sum(S_i²) < τ) + 1`.

- **Error approximation:** Tucker error is bounded by sum of errors from each mode's truncation:

  ```python
  error(τ) ≈ sqrt(sum_i(1 - cumulative_energy_i))
  ```

### 2.5 Unified Tuning Interface for Decomposition

```python
class DecompositionTuner:
    """Tuning engine for all decomposition-based methods."""

    @staticmethod
    def rank_for_ratio(method: str, shape: tuple, target_ratio: float) -> int:
        """Compute rank needed to achieve target ratio (no tensor needed)."""
        ...

    @staticmethod
    def predict_ratio(method: str, shape: tuple, rank: int) -> float:
        """Predict ratio from rank (deterministic)."""
        ...

    @staticmethod
    def predict_error(tensor: np.ndarray, method: str, rank: int) -> float:
        """Predict error from rank and tensor (SVD-based estimate)."""
        ...
```

---

## 3. Spectral Methods (DCT, Wavelet, FFT, Hadamard)

### 3.1 DCT Block

**Current:** Fixed block size and keep ratio.

**Tuning approach:**

- **Tunable parameters:**
  - `block_size`: size of DCT blocks (power of 2)
  - `keep_ratio`: fraction of coefficients to keep per block (0 to 1)
  - `threshold`: keep coefficient if |value| > threshold

- **Ratio function:**

  ```
  Coeffs per block = block_size
  Kept per block = block_size * keep_ratio (or count(|coeff| > threshold))
  Overhead = 1 bit per coefficient (mask) + scaler per block
  
  If using threshold: total_kept = sum(count(|coeff_i| > threshold)) across blocks
  ratio = total_elements * 4 / (total_kept * 4 + overhead)
  ```

- **Energy compaction approach (preferred):**

  For each DCT block, sort coefficients by magnitude. Keep coefficients until cumulative energy reaches threshold τ. This is optimal for energy compaction.

  ```python
  def tune_dct(tensor, target_ratio, block_size=64):
      """Find threshold τ to hit target ratio."""
      flat = tensor.ravel()
      padded_len = -(-len(flat) // block_size) * block_size  # ceil division
      padded = np.zeros(padded_len)
      padded[:len(flat)] = flat
      blocks = padded.reshape(-1, block_size)
      dct_blocks = dct(blocks, norm='ortho')
      
      # Sort all coefficients by magnitude
      all_coeffs = np.sort(np.abs(dct_blocks).ravel())[::-1]
      
      # Binary search on number of coefficients to keep
      total = len(all_coeffs)
      lo, hi = 1, total
      while lo < hi:
          mid = (lo + hi) // 2
          # Compute ratio: kept coefficients * 4 bytes + mask overhead
          compressed_size = mid * 4 + total // 8
          ratio = tensor.nbytes / compressed_size
          if ratio >= target_ratio:
              hi = mid
          else:
              lo = mid + 1
      
      threshold = all_coeffs[lo - 1]
      return threshold, lo  # threshold and count of kept coefficients
  ```

- **Error prediction:**

  ```python
  def predict_dct_error(coeffs, threshold):
      """Error = ‖discarded_coeffs‖ / ‖all_coeffs‖."""
      mask = np.abs(coeffs) < threshold
      discarded_energy = np.sum(coeffs[mask]**2)
      total_energy = np.sum(coeffs**2)
      return np.sqrt(discarded_energy / total_energy)
  ```

  This is **analytically exact** in the DCT domain (Parseval's theorem ensures energy is preserved between spatial and DCT domains for orthonormal DCT).

### 3.2 Wavelet

**Current:** Fixed level and keep ratio.

**Tuning approach:**

- **Tunable parameters:**
  - `level`: number of decomposition levels (1 to log2(min_dims))
  - `keep_ratio`: fraction of wavelet coefficients to keep
  - `threshold`: hard/soft threshold on detail coefficients

- **Ratio function:** Similar to DCT — coefficients are stored sparsely after thresholding. Ratio is determined by the number of non-zero coefficients after quantization + overhead for the wavelet tree structure.

- **Error prediction:** Energy of thresholded coefficients is deterministic:

  ```python
  def predict_wavelet_error(coeffs, threshold):
      kept = coeffs[np.abs(coeffs) >= threshold]
      error = sqrt((sum(coeffs**2) - sum(kept**2)) / sum(coeffs**2))
      return error
  ```

- **Tuning:** Multi-level wavelet produces a hierarchy. Higher levels (coarse) are kept at higher precision; lower levels (detail) are thresholded more aggressively. The tuning parameter is the per-level threshold.

### 3.3 FFT / Fourier

**Current:** Fixed keep ratio in frequency domain.

**Tuning approach:**

- Same as DCT but in complex frequency domain.
- **Key difference:** FFT produces complex coefficients. Error is in `L2` norm (Parseval holds).
- **Tuning parameter:** Frequency threshold — keep coefficients with magnitude above threshold.

### 3.4 Hadamard / FWHT

**Current:** Block size + bit width (INT8/INT4).

**Tuning approach:**

- Hadamard transform is energy-preserving (orthogonal). Same energy-compaction logic as DCT.
- **Unique property:** Hadamard is fast (O(n log n) with no multiplies). This makes it feasible to do a **full sweep** of all possible thresholds in `O(n log n + n log n) = O(n log n)` time by computing the transform once and scanning thresholds.

### 3.5 Unified Spectral Tuning

```python
class SpectralTuner:
    """Tuning engine for all spectral/transform methods."""

    @staticmethod
    def transform(tensor: np.ndarray, method: str) -> np.ndarray:
        """Apply spectral transform and return coefficients."""
        ...

    @staticmethod
    def threshold_for_ratio(coeffs: np.ndarray, target_ratio: float) -> float:
        """Find threshold value to achieve target ratio via binary search."""
        sorted_magnitudes = np.sort(np.abs(coeffs).ravel())[::-1]
        # Binary search on count
        ...

    @staticmethod
    def predict_error(coeffs: np.ndarray, threshold: float) -> float:
        """Predict relative error from thresholding (Parseval)."""
        kept = coeffs[np.abs(coeffs) >= threshold]
        return sqrt((sum(np.abs(coeffs)**2) - sum(np.abs(kept)**2))
                     / sum(np.abs(coeffs)**2))

    @staticmethod
    def tune(tensor: np.ndarray, method: str, target_ratio: float, max_error: float = None):
        """Auto-tune spectral method for target ratio or error."""
        coeffs = transform(tensor, method)
        threshold = threshold_for_ratio(coeffs, target_ratio)
        actual_error = predict_error(coeffs, threshold)
        if max_error and actual_error > max_error:
            # Fall back to lower ratio (more coefficients)
            ...
        return dict(method=method, threshold=threshold, 
                   kept_fraction=..., error=actual_error)
```

---

## 4. Quantization Methods (INT8, INT4, NF4, etc.)

### 4.1 Block Quantization (INT8, INT4)

**Current:** Fixed `block_size` and `n_bits`.

**Tuning approach:**

- **Tunable parameters:** `block_size` (power of 2, 16-512), `n_bits` (1-16)

- **Ratio function:**

  ```
  n_blocks = ceil(n_elements / block_size)
  scales_bytes = n_blocks * 4          # float32 scales
  quantized_bytes = n_elements * n_bits / 8  # packed bits
  overhead = scales_bytes + header
  Ratio = (n_elements * 4) / (quantized_bytes + overhead + header)
  ```

  For small block_size, the overhead from scales becomes significant. The optimal block size for a given ratio can be derived analytically.

- **Inverse for target ratio:**

  ```
  n_bits_given_block_size(R, block_size) = (32 / R - 4 / block_size) * 8
  ```

  Or more precisely:

  ```python
  def bits_for_ratio(n_elements, target_ratio, block_size):
      """Compute bit width needed for target ratio with given block size."""
      n_blocks = (n_elements + block_size - 1) // block_size
      scales_bytes = n_blocks * 4
      header = 8  # struct header
      
      # Solve: ratio = n_elements*4 / (n_elements*bits/8 + scales_bytes + header)
      target_bytes = n_elements * 4 / target_ratio
      bits = (target_bytes - scales_bytes - header) * 8 / n_elements
      return max(1, min(int(bits), 16))
  ```

- **Error model:**

  For uniform quantization of a zero-mean signal with variance σ² in block quantization:

  ```python
  def predict_quant_error(tensor, n_bits, block_size=128):
      """Predict relative error for uniform quantization."""
      flat = tensor.ravel()
      n = len(flat)
      n_blocks = (n + block_size - 1) // block_size
      padded = np.zeros(n_blocks * block_size)
      padded[:n] = flat
      blocks = padded.reshape(-1, block_size)
      
      # Error per block: quantization noise = Δ²/12 where Δ = 2*max|value|/(2^n_bits - 1)
      block_max = np.max(np.abs(blocks), axis=1)
      delta = 2 * block_max / (2**n_bits - 1)
      per_element_noise = delta**2 / 12  # per element, per block
      
      total_noise = np.sum(per_element_noise * block_size) / n_blocks
      signal_energy = np.mean(flat**2)
      return np.sqrt(total_noise / signal_energy)
  ```

  This is the **optimal mean-squared error for uniform quantization** under the high-rate approximation. For weight tensors with roughly uniform distribution per block, it's accurate within 10-20%.

- **Adaptive bits per block:**
  
  Instead of uniform bits across all blocks, allocate bits based on block variance:

  ```python
  def allocate_bits(tensor, target_ratio, block_size=64):
      """Allocate heterogeneous bit widths per block to hit target ratio."""
      flat = tensor.ravel()
      blocks = flat.reshape(-1, block_size)
      block_variance = np.var(blocks, axis=1)
      
      # High-variance blocks get more bits, low-variance get fewer
      # Solve the optimization: minimize total error subject to total size ≤ target
      """
      This is a water-filling problem:
      minimize sum(sigma_i² * 2^{-2b_i})
      subject to sum(b_i) = total_bits
      
      Solution: b_i = b_avg + 0.5 * log2(sigma_i² / geometric_mean(sigma²))
      Clamp to [1, 8]
      """
      log_var = np.log2(block_variance + 1e-30)
      b_avg = total_bits / n_blocks
      b_i = b_avg + 0.5 * (log_var - np.mean(log_var))
      return np.clip(np.round(b_i), 1, 8).astype(int)
  ```

### 4.2 NF4 (Normal Float 4)

**Current:** Fixed 4-bit.

**Tuning approach:**

- NF4 is a non-uniform 4-bit format (16 possible values derived from normal distribution quantiles).
- **No direct bit-width tuning** (it's always 4-bit), but can tune block size.
- **Error model:** NF4 has approximately 1.5-2x better quantization error than uniform INT4 for normally distributed weights. Can be modeled as:

  ```python
  def predict_nf4_error(tensor, block_size=64):
      """NF4 error is roughly 0.6x uniform INT4 error for Gaussian weights."""
      int4_error = predict_quant_error(tensor, 4, block_size)
      gaussianity = measure_gaussianity(tensor.ravel())
      return int4_error * (0.6 + 0.4 * (1 - gaussianity))
  ```

### 4.3 Binary / Ternary Quantization

**Current:** Fixed 1-bit (binary) or ~1.6-bit (ternary).

**Tuning approach:**

- **Ratio is fixed** for a given format, but we can tune by mixing binary/ternary/fp16 per block:

  ```python
  def tune_binary_mixed(tensor, target_ratio):
      """Mix binary and higher-precision blocks to hit target."""
      blocks = tensor.reshape(-1, block_size)
      block_norms = np.linalg.norm(blocks, axis=1)
      
      # Sort blocks by norm; low-norm blocks get binary, high-norm get int4/int8
      # Binary ratio: ~32x, INT4 ratio: ~8x
      # Solve for fraction_high to hit overall target
      ...
  ```

### 4.4 Mixed Precision

**Current:** Fixed per-layer bit allocation.

**Tuning approach:**

- **Global optimization:** Given a total model size budget `B`, allocate bits per layer to minimize total error:

  ```python
  def optimal_bit_allocation(layers: List[LayerProfile], budget_bytes: int):
      """Water-filling across layers: more bits to high-variance/sensitive layers."""
      sensitivities = [layer.sensitivity for layer in layers]
      variances = [layer.variance for layer in layers]
      
      # Score = sensitivity * variance
      scores = np.array(sensitivities) * np.array(variances)
      
      # Allocate bits proportionally to sqrt(score)
      bits = budget_bytes / sum(np.sqrt(scores)) * np.sqrt(scores)
      return bits
  ```

### 4.5 K-Means / Codebook Quantization

**Current:** Fixed number of centroids.

**Tuning approach:**

- **Ratio function for k centroids with code_size entries:**

  ```
  codebook = k * code_size * 4 bytes
  indices = n_elements * log2(k) / 8 bytes
  Ratio = (n_elements * 4) / (k * code_size * 4 + n_elements * log2(k) / 8)
  ```

- **Inverse:** Binary search on k to hit ratio (k = 1..256 typically).

- **Error model:** Distortion decreases as k increases. A practical approach: run k-means for a range of k values, measure distortion, fit `error(k) ≈ a * k^{-2/d}` where d is the code vector dimension.

---

## 5. Structural Methods (Toeplitz, BlockDiagonal, Sparsity)

### 5.1 Block-Sparse

**Current:** Fixed block_size and density.

**Tuning approach:**

- **Tunable parameters:** `block_size` (4-64), `density` (0.0-1.0)

- **Ratio function:**

  ```
  n_blocks = n_elements / block_size
  kept_blocks = n_blocks * density
  kept_elements = kept_blocks * block_size
  mask_bytes = n_blocks / 8  # bit mask
  data_bytes = kept_elements * 4
  Ratio = (n_elements * 4) / (data_bytes + mask_bytes + header)
  ```

- **Inverse:**

  ```
  density = (n_elements * 4 / R_target - mask_bytes - header) / (kept_blocks * block_size * 4)
  ```

- **Error model:** Block sparsity error depends on the norm of discarded blocks. Can be predicted from the block norm distribution:

  ```python
  def predict_sparsity_error(tensor, density, block_size=16):
      blocks = tensor.reshape(-1, block_size)
      norms = np.linalg.norm(blocks, axis=1)
      sorted_norms = np.sort(norms)[::-1]
      n_keep = int(len(norms) * density)
      kept_energy = sum(sorted_norms[:n_keep]**2)
      total_energy = sum(norms**2)
      return sqrt(1 - kept_energy / total_energy)
  ```

### 5.2 Block Diagonal

**Current:** Fixed block_size.

**Tuning approach:**

- **Tunable parameter:** `block_size` — larger = lower ratio but potentially lower error.
- **Ratio function for d×d matrix with b×b blocks:**

  ```
  All blocks stored as dense float32: n_blocks = ceil(d/b)² on diagonal
  Each block: b² × 4 bytes
  Ratio = d² / (ceil(d/b)² * b²) = d² / (ceil(d/b)² * b²)
  
  For d divisible by b: Ratio = (d/b)² / (d/b)² = 1.0 (no compression!)
  Wait — block diagonal is ONLY the diagonal blocks:
  Non-zero = (d/b) * b² = d*b
  Ratio = d² / (d*b) = d/b
  ```

- **Inverse:** `block_size = d / R_target`

- **Error model:** The off-diagonal energy is discarded. Can be estimated by sampling a few off-diagonal blocks.

### 5.3 N:M Sparsity (2:4, 4:8)

**Current:** Fixed N:M pattern.

**Tuning approach:**

- **Tunable parameter:** N in N:M (e.g., 1:4, 2:4, 3:4).

- **Ratio is fixed** by the sparsity pattern — N:M always has `N/M * 2` bytes per element (storing indices + values). But N directly controls quality:

  ```
  Ratio = M/N * storage_efficiency_factor
  Error ≈ error(N, M) — depends on which elements are kept
  ```

- **Tuning:** Use adaptive N per group based on the group's norm distribution. Groups with flat distributions get smaller N (more sparsity); groups with peaked distributions get larger N.

### 5.4 Toeplitz / Circulant / Structured Matrices

**Current:** Full matrix replaced by Toeplitz with first row/column.

**Tuning approach:**

- **Parameters:** Structure type (Toeplitz, Hankel, Circulant, etc.) + band width.
- **Ratio is deterministic** from structure (e.g., Toeplitz ≈ 2n-1 parameters vs n² for dense).
- **Error prediction:** Measure how close the tensor is to the assumed structure.

  ```python
  def toeplitz_approximation_error(tensor):
      """Error of best Toeplitz approximation."""
      n = min(tensor.shape)
      # Compute best Toeplitz approximation
      first_row = tensor[0, :]
      first_col = tensor[:, 0]
      toeplitz = np.fromfunction(lambda i, j: first_col[i-j], tensor.shape, dtype=int)
      ...
  ```

---

## 6. Physics/Quantum Methods

### 6.1 Vlasov Distribution / Mean Field

**Current:** Fixed `grid_size` or `n_particles`.

**Tuning approach:**

- **Tunable parameter:** `grid_size` (8-1024) for histogram, `n_particles` for mean-field.

- **Ratio function:**

  ```
  For Vlasov: grid_size bins → grid_size centers (float32) + sparse transition matrix
  Kept elements ≈ grid_size² * density_of_kept_cells
  Ratio = tensor.nbytes / (grid_size*4 + kept*2 + kept*2)  # centers + indices + values
  ```

- **Error model:** As grid_size increases, the histogram better approximates the distribution → lower error. Empirically, `error ≈ a * grid_size^{-b}`.

- **Tuning algorithm:**

  ```python
  def tune_vlasov(tensor, target_ratio):
      n_elements = tensor.nbytes // 4
      # Binary search on grid_size
      lo, hi = 8, 1024
      while lo < hi:
          mid = (lo + hi) // 2
          # Quick compress with grid_size=mid to measure ratio
          data, meta = VlasovDistribution().compress(tensor, grid_size=mid)
          ratio = tensor.nbytes / len(data)
          if ratio >= target_ratio:
              hi = mid
          else:
              lo = mid + 1
      return {"method": "vlasov", "grid_size": lo}
  ```

### 6.2 Quantum-Inspired Methods

**Tunable parameters:**

- **Number of qubits** (2-16): More qubits = higher fidelity, lower ratio.
- **Circuit depth** (1-100): Deeper = more expressive, better reconstruction.
- **Bond dimension** for tensor network methods (4-128).

**Tuning approach:**

- All quantum-inspired methods are **parameterized by rank/bond dimension**. Same mathematical framework as tensor decompositions.
- **Ratio function:** Exponential in qubits for amplitude encoding (`2^n` amplitudes stored as quantum circuit parameters). More practically: matrix product operator with bond dimension D stores `O(n * D²)` parameters.

### 6.3 Plasma Physics (MHD, Gyrokinetic, etc.)

**Tunable parameters:**

- **Mode truncation threshold:** Keep top-k Fourier modes.
- **Grid resolution:** Coarse vs fine grid.
- **Spectral truncation:** Truncate spectral representation at frequency f_max.

**Tuning approach:** Same as spectral methods — energy-based truncation with ratio computed analytically.

---

## 7. Entropy / Lossless Methods

### 7.1 rANS / Huffman / Arithmetic Coding

**Current:** Entropy encoding is applied after lossy compression. Not directly tunable for ratio.

**Tuning approach:**

- rANS is **near-optimal** for the given symbol distribution. The compressed size is close to `n_elements * H(distribution)` where H is entropy in bits.
- **Ratio prediction:**

  ```python
  def predict_entropy_ratio(tensor, quantize_bits=8):
      """Predict best-case entropy ratio."""
      flat = tensor.ravel()
      # Quantize to n_bits for entropy estimation
      q = np.round(flat * (2**(quantize_bits-1))).astype(np.int32)
      unique, counts = np.unique(q, return_counts=True)
      probs = counts / len(q)
      entropy = -np.sum(probs * np.log2(probs))
      compressed_bits = len(q) * entropy
      return tensor.nbytes * 8 / compressed_bits
  ```

- **Practical tuning:** Entropy methods are applied downstream of lossy compression. The tunable parameter is the **precision** (number of symbols) fed into the encoder.

---

## 8. Unified Tuning API

### 8.1 Base Class Design

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TuningResult:
    """Result of tuning a compression method to a target."""
    method_name: str
    parameters: Dict[str, any]       # Optimal parameters found
    predicted_ratio: float            # Expected compression ratio
    predicted_error: float            # Expected relative error
    actual_ratio: float               # Actual ratio after compression
    actual_error: float               # Actual error after compression
    confidence: float                 # Confidence in prediction (0-1)
    computation_time: float           # Time spent tuning + compressing


class TunedCompressionMethod(ABC):
    """A compression method that can be dynamically tuned.
    
    Subclasses implement:
    - _compress_impl(tensor, params) -> (bytes, dict)
    - _decompress_impl(data, metadata) -> np.ndarray
    - _predict_size(shape, nbytes, params) -> int
    - _predict_error(tensor, params) -> float
    - _parameter_sweep() -> List[Dict[str, any]]
    """
    
    name: str = "base_method"
    category: str = "generic"
    
    @abstractmethod
    def _compress_impl(self, tensor: np.ndarray, **params) -> Tuple[bytes, dict]:
        """Perform actual compression with given parameters."""
        ...
    
    @abstractmethod
    def _decompress_impl(self, data: bytes, metadata: dict) -> np.ndarray:
        """Decompress data."""
        ...
    
    def compress(self, tensor: np.ndarray, 
                 target_ratio: Optional[float] = None,
                 target_error: Optional[float] = None,
                 max_error: Optional[float] = None,
                 tune: bool = True) -> Tuple[bytes, dict]:
        """Auto-tune to meet targets, then compress.
        
        If tune=True, search for parameters that satisfy constraints.
        If target_ratio given, find params to meet ratio; minimize error.
        If target_error given, find params to meet error; maximize ratio.
        If both given, find params that satisfy both.
        """
        if tune and (target_ratio is not None or target_error is not None):
            params = self.tune(tensor, target_ratio, target_error, max_error)
        else:
            params = self._default_params()
        
        data, meta = self._compress_impl(tensor, **params)
        return data, {**meta, **params}
    
    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        return self._decompress_impl(data, metadata)
    
    def predict_outcome(self, tensor: np.ndarray, 
                        target_ratio: Optional[float] = None,
                        target_error: Optional[float] = None) -> TuningResult:
        """Predict ratio and error for a target WITHOUT compressing.
        
        This is the key method for fast exploration of the ratio/error space.
        Implementation must NOT actually compress — just compute what WOULD
        happen, using analytical formulas or very fast approximations.
        """
        if target_ratio is not None:
            params = self._find_params_for_ratio(tensor, target_ratio)
        elif target_error is not None:
            params = self._find_params_for_error(tensor, target_error)
        else:
            params = self._default_params()
        
        ratio = self._predict_size(tensor.shape, tensor.nbytes, params)
        ratio = tensor.nbytes / ratio if isinstance(ratio, int) else ratio
        error = self._predict_error(tensor, params)
        
        return TuningResult(
            method_name=self.name,
            parameters=params,
            predicted_ratio=ratio,
            predicted_error=error,
            actual_ratio=0.0, actual_error=0.0,
            confidence=self._estimate_confidence(tensor, params),
            computation_time=0.0,
        )
    
    def get_parameter_sweep(self, tensor: np.ndarray) -> List[TuningResult]:
        """Return (ratio, error) for range of parameter values.
        
        Returns a curve of achievable (ratio, error) pairs.
        Used to construct the Pareto frontier.
        """
        results = []
        for params in self._parameter_sweep():
            ratio = tensor.nbytes / self._predict_size(tensor.shape, tensor.nbytes, params)
            error = self._predict_error(tensor, params)
            results.append(TuningResult(
                method_name=self.name, parameters=params,
                predicted_ratio=ratio, predicted_error=error,
                actual_ratio=0.0, actual_error=0.0,
                confidence=self._estimate_confidence(tensor, params),
                computation_time=0.0,
            ))
        return results
    
    @abstractmethod
    def _default_params(self) -> Dict[str, any]:
        """Return default parameter set."""
        ...
    
    @abstractmethod
    def _find_params_for_ratio(self, tensor: np.ndarray, target_ratio: float) -> Dict[str, any]:
        """Inverse: find parameters that achieve target ratio."""
        ...
    
    @abstractmethod
    def _find_params_for_error(self, tensor: np.ndarray, target_error: float) -> Dict[str, any]:
        """Inverse: find parameters that achieve target error."""
        ...
    
    @abstractmethod
    def _predict_size(self, shape: tuple, nbytes: int, params: dict) -> int:
        """Predict compressed size in bytes for given parameters (no compression needed)."""
        ...
    
    @abstractmethod
    def _predict_error(self, tensor: np.ndarray, params: dict) -> float:
        """Predict reconstruction error for given parameters (no decompression needed)."""
        ...
    
    @abstractmethod
    def _parameter_sweep(self) -> List[Dict[str, any]]:
        """Generate all discrete parameter combinations to try."""
        ...
    
    def _estimate_confidence(self, tensor: np.ndarray, params: dict) -> float:
        """Estimate how reliable the prediction is (0-1).
        
        Influenced by:
        - How close the tensor distribution matches the method's assumptions
        - How well the error model fits empirical data for this tensor type
        """
        return 0.9  # Default: high confidence
```

### 8.2 Concrete Example: TunedBlockINT8

```python
class TunedBlockINT8(TunedCompressionMethod):
    """BlockINT8 with dynamic block_size tuning."""
    
    name = "block_int8"
    category = "quantization"
    
    def _compress_impl(self, tensor: np.ndarray, 
                       block_size: int = 128) -> Tuple[bytes, dict]:
        return _BlockINT8().compress(tensor, block_size=block_size)
    
    def _decompress_impl(self, data: bytes, metadata: dict) -> np.ndarray:
        return _BlockINT8().decompress(data, metadata)
    
    def _predict_size(self, shape: tuple, nbytes: int, params: dict) -> int:
        n = nbytes // 4  # elements (assuming float32)
        block_size = params.get('block_size', 128)
        n_blocks = (n + block_size - 1) // block_size
        header = 8
        scales = n_blocks * 4
        quantized = n_blocks * block_size  # int8 = 1 byte per element
        return header + scales + quantized
    
    def _predict_error(self, tensor: np.ndarray, params: dict) -> float:
        block_size = params.get('block_size', 128)
        return predict_quant_error(tensor, 8, block_size)
    
    def _find_params_for_ratio(self, tensor: np.ndarray, target_ratio: float) -> dict:
        n = tensor.nbytes // 4
        # Block size primarily affects overhead ratio
        # For large tensors, block size has minimal impact on ratio
        # Default block_size = 128 is fine for most cases
        block_size = 128
        return {"block_size": block_size}
    
    def _find_params_for_error(self, tensor: np.ndarray, target_error: float) -> dict:
        # INT8 gives fixed precision; can't tune below ~0.5% relative error typically
        # Block size only marginally affects error (larger blocks = slightly more error)
        return {"block_size": 64}  # Smaller blocks for lower error
    
    def _parameter_sweep(self) -> List[Dict[str, any]]:
        return [{"block_size": bs} for bs in [16, 32, 64, 128, 256, 512]]
    
    def _default_params(self) -> Dict[str, any]:
        return {"block_size": 128}
```

### 8.3 Concrete Example: TunedSVD

```python
class TunedSVDTruncated(TunedCompressionMethod):
    """SVD with tunable rank."""
    
    name = "svd_truncated"
    category = "decomposition"
    
    def _compress_impl(self, tensor: np.ndarray, rank: int = 16) -> Tuple[bytes, dict]:
        m, n = tensor.shape
        U, S, Vt = np.linalg.svd(tensor.astype(np.float32), full_matrices=False)
        return _serialize(U[:, :rank]) + _serialize(S[:rank]) + _serialize(Vt[:rank, :]), {
            'shape': tensor.shape, 'rank': rank,
        }
    
    def _decompress_impl(self, data: bytes, metadata: dict) -> np.ndarray:
        shape = metadata['shape']
        rank = metadata['rank']
        U = _deserialize(data[:shape[0]*rank*4]).reshape(shape[0], rank)
        S = _deserialize(data[shape[0]*rank*4:(shape[0]+rank)*rank*4])
        Vt = _deserialize(data[(shape[0]+rank)*rank*4:]).reshape(rank, shape[1])
        return (U * S).dot(Vt).reshape(shape)
    
    def _predict_size(self, shape: tuple, nbytes: int, params: dict) -> int:
        m, n = shape
        rank = params.get('rank', 16)
        return 4 * (m * rank + rank + rank * n)  # U + S + Vt
    
    def _predict_error(self, tensor: np.ndarray, params: dict) -> float:
        rank = params.get('rank', 16)
        # Use SVD to compute exact error (fast for first k singular values)
        m, n = tensor.shape
        k = min(rank + 5, min(m, n))
        _, S, _ = np.linalg.svd(tensor.astype(np.float64), full_matrices=False)
        S = S[:k]
        # Pad with zeros if rank > k
        if rank <= k:
            discarded = S[rank:] if rank < len(S) else []
        else:
            discarded = []
        discarded_energy = sum(d**2 for d in discarded)
        total_energy = sum(S**2) + max(0, sum(np.linalg.svd(
            tensor.astype(np.float64), full_matrices=False)[1][k:]**2))
        return np.sqrt(discarded_energy / total_energy) if total_energy > 0 else 0.0
    
    def _find_params_for_ratio(self, tensor: np.ndarray, target_ratio: float) -> dict:
        m, n = tensor.shape
        rank = int(m * n / (target_ratio * (m + n + 1)))
        rank = max(1, min(rank, min(m, n)))
        return {"rank": rank}
    
    def _find_params_for_error(self, tensor: np.ndarray, target_error: float) -> dict:
        m, n = tensor.shape
        _, S, _ = np.linalg.svd(tensor.astype(np.float64), full_matrices=False)
        total_energy = sum(S**2)
        cumulative = np.cumsum(S**2) / total_energy
        # Error = sqrt(1 - cumulative), find rank where error <= target_error
        target_cumulative = 1 - target_error**2
        rank = int(np.searchsorted(cumulative, target_cumulative) + 1)
        rank = min(rank, min(m, n))
        return {"rank": rank}
    
    def _parameter_sweep(self) -> List[Dict[str, any]]:
        return [{"rank": k} for k in [1, 2, 4, 8, 16, 32, 64, 128]]
    
    def _default_params(self) -> Dict[str, any]:
        return {"rank": 32}
```

### 8.4 Rapid Outcome Prediction (Fast Path)

The `predict_outcome` API is designed for **rapid exploration** without compression. The engine can call `predict_outcome` on dozens of methods in microseconds, before selecting the best one for actual compression:

```python
# Fast cost-benefit analysis before any compression
candidates = []
for method_name in available_methods:
    method: TunedCompressionMethod = registry[method_name]
    result = method.predict_outcome(tensor, target_ratio=50.0)
    candidates.append(result)

# Sort by predicted error
candidates.sort(key=lambda r: r.predicted_error)
best_method = candidates[0]
# Now actually compress
data, meta = registry[best_method.method_name].compress(tensor, target_ratio=50.0)
```

---

## 9. Error-Ratio Pareto Frontier

### 9.1 Definition

For each compression method, the **Pareto frontier** is the set of achievable (ratio, error) pairs where no other parameter setting gives both higher ratio AND lower error. Points on the frontier are **Pareto-optimal**.

### 9.2 Computing the Frontier

```python
def compute_pareto_frontier(tensor: np.ndarray, 
                           method: TunedCompressionMethod) -> List[TuningResult]:
    """Compute the Pareto-optimal curve for a single method."""
    sweep = method.get_parameter_sweep(tensor)
    
    # Sort by ratio (ascending)
    sweep.sort(key=lambda r: r.predicted_ratio)
    
    # Pareto filter: keep only points where error decreases as ratio decreases
    frontier = []
    min_error = float('inf')
    for point in sweep:
        if point.predicted_error < min_error:
            frontier.append(point)
            min_error = point.predicted_error
    
    return frontier  # Now monotonic: higher ratio = higher error
```

### 9.3 Cross-Method Pareto Frontier

```python
def compute_cross_method_frontier(tensor: np.ndarray,
                                 methods: List[TunedCompressionMethod]
                                 ) -> Dict[str, List[TuningResult]]:
    """Compute Pareto frontiers for multiple methods and find the envelope."""
    all_points = []
    for method in methods:
        frontier = compute_pareto_frontier(tensor, method)
        for point in frontier:
            all_points.append((point, method.name))
    
    # Sort by ratio
    all_points.sort(key=lambda x: x[0].predicted_ratio)
    
    # Global Pareto envelope
    global_frontier = {}
    min_error = float('inf')
    for point, method_name in all_points:
        if point.predicted_error < min_error:
            global_frontier[point.predicted_ratio] = (method_name, point)
            min_error = point.predicted_error
    
    return global_frontier
```

### 9.4 Interpolation

For ratio values between discrete parameter settings, interpolate:

```python
def interpolate_error(frontier: List[TuningResult], target_ratio: float) -> float:
    """Estimate error for a ratio between discrete points on the frontier."""
    if target_ratio <= frontier[0].predicted_ratio:
        return frontier[0].predicted_error
    if target_ratio >= frontier[-1].predicted_ratio:
        return frontier[-1].predicted_error
    
    for i in range(len(frontier) - 1):
        r1, e1 = frontier[i].predicted_ratio, frontier[i].predicted_error
        r2, e2 = frontier[i+1].predicted_ratio, frontier[i+1].predicted_error
        if r1 <= target_ratio <= r2:
            # Log-linear interpolation (error ~ exponential in ratio)
            log_e1, log_e2 = np.log(e1 + 1e-30), np.log(e2 + 1e-30)
            t = (target_ratio - r1) / (r2 - r1)
            return np.exp(log_e1 + t * (log_e2 - log_e1))
    
    return frontier[-1].predicted_error
```

### 9.5 Pareto Observations

| Method Category | Typical Ratio Range | Typical Error Range | Frontier Shape |
|---|---|---|---|
| BlockINT8 | 3-4x | 0.1-2% | Flat (error insensitive to block size) |
| BlockINT4 | 6-9x | 0.5-5% | Concave up |
| HadamardINT8 | 3-5x | 0.1-3% | Similar to BlockINT8 |
| SVD | 2-100x+ | 0.01-50% | Concave down (steep at low ratio) |
| DCT | 2-50x | 0.1-20% | Concave down |
| TT | 3-50x | 1-30% | Similar to SVD |
| Binary | 30-33x | 5-20% | Near-vertical (fixed ratio) |
| BlockSparse | 2-20x | 1-30% | Linear |
| Vlasov | 5-50x | 5-30% | Power law |

---

## 10. Cross-Method Optimization

### 10.1 Method Selection for Target Ratio

The system selects the optimal method by:

1. **Computing Pareto frontiers** for all methods (via `predict_outcome`)
2. **Finding which method gives lowest error** at the target ratio
3. **Verifying** with actual compression

```python
def select_method_for_target(tensor: np.ndarray,
                            target_ratio: float,
                            methods: Dict[str, TunedCompressionMethod],
                            max_error: Optional[float] = None) -> str:
    """Select the best method for a target ratio.
    
    Returns the method name that minimizes error at target_ratio.
    """
    best_method = None
    best_error = float('inf')
    
    for name, method in methods.items():
        # Try the exact ratio first
        result = method.predict_outcome(tensor, target_ratio=target_ratio)
        
        if result.predicted_error < best_error:
            best_error = result.predicted_error
            best_method = name
            
            # Fast exit if we're already within max_error
            if max_error and result.predicted_error <= max_error:
                return name
    
    return best_method
```

### 10.2 Cascading Methods for Extreme Ratios

For ratios exceeding any single method's capability, cascade methods:

```python
def cascade_compress(tensor: np.ndarray,
                    target_ratio: float,
                    max_error: Optional[float] = None) -> TuningResult:
    """Cascade multiple methods to achieve extreme compression ratios.
    
    Strategy: 
    - First stage: decomposition (SVD, TT) for 2-10x
    - Second stage: quantization (INT8, INT4) for additional 3-8x
    - Third stage: entropy encoding (rANS) for 1.5-3x
    - Total: 10-240x+
    """
    # Select cascade stages based on target ratio
    if target_ratio <= 10:
        # Single stage: best single method
        return single_stage(tensor, target_ratio, max_error)
    elif target_ratio <= 50:
        # Two stages: decomposition + quantization
        # ratio = ratio_1 * ratio_2
        ratios = _split_ratio(target_ratio, 2)
        stage1 = cascade_compress(tensor, ratios[0], max_error)
        residual = tensor - stage1.reconstructed  # compress residual
        stage2 = cascade_compress(residual, ratios[1], max_error)
        return merge_stages([stage1, stage2])
    else:
        # Three stages: decomposition + quantization + entropy
        ratios = _split_ratio(target_ratio, 3)
        ...
```

### 10.3 Cascade Design Principles

1. **SVD first** (captures global low-rank structure)
2. **Then DCT/wavelet** (captures local frequency structure in residual)
3. **Then quantization** (eliminates imperceptible noise)
4. **Then entropy coding** (lossless, no additional error)

Each stage operates on the **residual** of the previous stage. The total error is:

```
total_error ≈ sqrt(error_1² + error_2² + ... + error_n²)
```

### 10.4 Error Budget Allocation Across Cascades

```python
def allocate_ratio_across_stages(n_stages: int, target_ratio: float) -> List[float]:
    """Split target ratio across cascade stages.
    
    Strategy: earlier stages get more aggressive ratios since
    they capture more energy.
    """
    if n_stages == 2:
        r1 = sqrt(target_ratio) * 1.2  # First stage slightly more aggressive
        r2 = target_ratio / r1
        return [r1, r2]
    elif n_stages == 3:
        r1 = target_ratio ** (1/3) * 1.5
        r2 = target_ratio ** (1/3)
        r3 = target_ratio / (r1 * r2)
        return [r1, r2, r3]
    ...
```

---

## 11. Implementation Plan

### Phase 1: `predict_outcome()` for Each Method ***(2 weeks)***

**Goal:** Add `predict_outcome()` to every method — predict ratio and error without compressing.

**Tasks:**

1. **Core quant methods** (block_int8, block_int4, hadamard_int8/4):
   - Implement `_predict_size()` using analytical formulas
   - Implement `_predict_error()` using quantization noise model
   - File: `spectralstream/compression/engine/_methods.py`

2. **Spectral methods** (DCT, FFT, Wavelet, FWHT):
   - Implement `_predict_size()` from coefficient threshold
   - Implement `_predict_error()` via Parseval energy ratio
   - File: `spectralstream/compression/methods/spectral/`

3. **Decomposition methods** (SVD, TT, CP, Tucker):
   - Implement `_predict_size()` from rank
   - Implement `_predict_error()` from singular values / energy
   - File: `spectralstream/compression/methods/decomposition/`

4. **Structural methods** (sparsity, block-diagonal, etc.):
   - Implement `_predict_size()` from density/block_size
   - Implement `_predict_error()` from discarded block norms
   - File: `spectralstream/compression/methods/structural/`

5. **Physics methods** (Vlasov, quantum, plasma):
   - Implement `_predict_size()` from grid_size/n_particles
   - Implement `_predict_error()` from histogram approximation quality
   - File: `spectralstream/compression/methods/physics/`

6. **Modify registry** to support `predict_outcome()`:
   - Update method metadata to include tunable parameters
   - Add parameter range/sweep info to registration
   - File: `spectralstream/compression/registry/`

7. **Tests:**
   - For each method: verify `predict_outcome()` matches actual compression within tolerance
   - Test edge cases: empty tensors, rank-1 tensors, scalar tensors
   - File: `tests/test_tuning.py`

**Validation criteria:**
- `|predicted_ratio - actual_ratio| / actual_ratio < 0.05` (5% tolerance)
- `|predicted_error - actual_error| / actual_error < 0.20` (20% tolerance)

### Phase 2: `compress_with_target()` for Each Method ***(1 week)***

**Goal:** Add `compress(tensor, target_ratio, max_error)` to each method.

**Tasks:**

1. **Add parameter search logic:**
   - For methods with continuous parameters (threshold, rank): use `_find_params_for_ratio()` with binary search
   - For methods with discrete parameters (block_size, n_bits): use `_find_params_for_ratio()` with nearest-neighbor
   
2. **Implement fallback:**
   - If target ratio is impossible (above theoretical max), return max achievable ratio with error estimate
   - If error too high, notify user and suggest cascade

3. **Modify `_orchestrator.py`:**
   - Add `target_ratio` field to `CompressionConfig`
   - Modify `compress_tensor_with_validation()` to use `compress(tensor, target_ratio=...)`
   - Preserve existing behavior as default (backward compat)

### Phase 3: Tuning Engine ***(2 weeks)***

**Goal:** Build the `TuningEngine` class that orchestrates cross-method optimization.

**Tasks:**

1. **Create `spectralstream/compression/engine/tuning_engine.py`:**

```python
class TuningEngine:
    """Orchestrates dynamic tuning across all compression methods.
    
    Given a tensor and target_ratio:
    1. Predicts outcome for every available method (fast path)
    2. Finds Pareto-optimal methods for target ratio
    3. Selects best method and tunes its parameters
    4. Optionally designs a cascade for extreme ratios
    """
    
    def __init__(self, registry: Dict[str, TunedCompressionMethod]):
        self.methods = registry
    
    def find_best_method(self, tensor: np.ndarray, target_ratio: float,
                         max_error: Optional[float] = None) -> TuningResult:
        """Select and tune the best method for target ratio."""
        ...
    
    def get_pareto_frontier(self, tensor: np.ndarray) -> Dict[str, List[TuningResult]]:
        """Return Pareto frontier for every method."""
        ...
    
    def design_cascade(self, tensor: np.ndarray, target_ratio: float,
                       max_error: Optional[float] = None) -> List[str]:
        """Design optimal cascade of methods for extreme ratio."""
        ...
```

2. **Integrate with `CompressionIntelligenceEngine`:**
   - `TuningEngine` becomes the method selector
   - Replace `method_candidates` with `TuningEngine.find_best_method()`
   - Store Pareto frontier per tensor type for future lookups

### Phase 4: CLI Integration ***(1 week)***

**Goal:** Expose tuning functionality in the CLI.

**Tasks:**

1. **New CLI commands:**
   ```
   $ python -m spectralstream.compression.cli tune tensor.npy --ratio 50 --max-error 0.01
   $ python -m spectralstream.compression.cli predict tensor.npy --method svd --rank 16
   $ python -m spectralstream.compression.cli frontier tensor.npy --output frontier.json
   $ python -m spectralstream.compression.cli cascade tensor.npy --ratio 500
   ```

2. **Modified `compress` command:**
   ```
   # These become equivalent:
   $ python -m spectralstream.compression.cli compress model.safetensors --ratio 50
   $ python -m spectralstream.compression.cli compress model.safetensors --ratio 50 --max-error 0.001
   ```

3. **Output enhancements:**
   - Show method selection rationale
   - Show predicted vs actual ratio/error
   - Show Pareto frontier if `--verbose`
   - Export frontier as JSON for analysis

### Phase 5: Learning & Adaptation ***(2 weeks)***

**Goal:** The system learns from experience which parameters work best for which tensor types.

**Tasks:**

1. **Outcome database:**
   ```python
   # ~/.spectralstream/tuning_cache.json
   {
     "tensor_type:attention_q_proj": {
       "method:svd_truncated": {
         "rank:16": {
           "avg_ratio": 12.3, "avg_error": 0.015,
           "n_samples": 47
         },
         ...
       }
     }
   }
   ```

2. **Cold start fallback:** Analytical predictions used initially.
   **Warm start:** Cached empirical results override analytical predictions after 5+ samples.

3. **Confidence calibration:**
   - Track prediction error over time
   - Adjust `_estimate_confidence()` based on historical accuracy
   - Flag methods/tensors where predictions are unreliable

4. **Active learning:**
   - Occasionally try alternative methods even when not selected (epsilon-greedy)
   - Compare actual vs predicted outcomes to improve error models
   - Update decay rate estimates per tensor type

### Phase 6: Polish & Productionize ***(1 week)***

**Tasks:**

1. **Benchmarking framework:**
   - Benchmark every method's tuning accuracy
   - Measure average prediction error across model families (Gemma, Llama, Mistral)
   - Report worst-case errors

2. **Edge cases:**
   - Very small tensors (< 64 elements)
   - Non-2D tensors (1D biases, 3D+ embeddings)
   - Integer dtypes (int32, uint8)
   - Tensors with NaN/inf values

3. **Documentation:**
   - Update DYNAMIC_TUNING_RESEARCH.md with experimental results
   - Add flowcharts showing tuning decision tree
   - Document Pareto frontier for each method family

---

## Appendix A: Summary of Tuning Formulas

| Method | Parameter | Ratio Function | Error Estimate | Inverse |
|---|---|---|---|---|
| SVD | rank `k` | `mn / (k(m+n+1))` | `√(∑σᵢ>ₖ² / ∑σ²)` | `k = mn / (R(m+n+1))` |
| TT | rank `r` | `∏nᵢ / (r²∑nᵢ)` | SVD proxy | `r = √(∏nᵢ / (R∑nᵢ))` |
| CP | rank `R` | `∏nᵢ / (R∑nᵢ)` | Power law `aR⁻ᵇ` | `R = ∏nᵢ / (R_target∑nᵢ)` |
| Tucker | rank `rᵢ` | `∏nᵢ / (∏rᵢ + ∑nᵢrᵢ)` | Mode-wise truncated SVD | Binary search |
| DCT | threshold `τ` | `N·4 / (k·4 + overhead)` | `√(E_discard / E_total)` | Sort coeffs, find τ |
| Wavelet | threshold `τ` | Same as DCT | Same as DCT | Same as DCT |
| FFT | threshold `τ` | Same as DCT | Same as DCT | Same as DCT |
| BlockINT8 | block_size `b` | `4N / (N + Nᵦ·4 + 8)` | `Δ²/12` per block | Fixed by N |
| BlockINT4 | block_size `b` | `4N / (N/2 + Nᵦ·4 + 8)` | `Δ²/12` per block | Fixed by N |
| NF4 | block_size `b` | `4N / (N/2 + Nᵦ·2 + 8)` | 0.6× INT4 error | Fixed by N |
| Binary | — | ~32× (fixed) | `σ²` | Fixed |
| BlockSparse | density `d` | `4n / (4·n·d + n/8b + header)` | Block norm proxy | `d = f(R)` |
| BlockDiagonal | block_size `b` | `d/b` (square matrix) | Off-diagonal energy | `b = d/R` |
| Vlasov | grid_size `g` | Grid-dependent | Histogram binning | Binary search on g |

## Appendix B: Key Implementation Risks

1. **SVD error prediction requires O(k) work** for k singular values. For large matrices, use randomized SVD with k=20 to estimate decay rate, then analytically extrapolate.

2. **TT/CP error models are approximate.** The SVD proxy error is a lower bound. Actual error depends on tensor structure. Start with conservative error estimates (add 20% margin).

3. **Entropy methods (rANS, Huffman) are lossless** when applied directly. Their tunable aspect is the precision of the input symbols. Apply after lossy compression only.

4. **Overhead dominates for small tensors.** Methods with high overhead (block quantization with small block sizes, sparse storage) may have unpredictable ratio for tensors < 128 elements. Fall through to passthrough.

5. **Predicted vs actual error gap.** The analytical error models assume the decompressed tensor exactly matches the predicted error bound. In practice, quantization rounding, FFT rounding, and other implementation details add 5-15% extra error. The `_estimate_confidence()` mechanism should account for this.

## Appendix C: Quick Reference — Method Tuning Type

| Type | Primary Tunable | Tuning Method | Ratio Prediction Accuracy | Error Prediction Accuracy |
|---|---|---|---|---|
| Decomposition | rank | Binary or closed-form | Exact | High (SVD: exact, others: moderate) |
| Spectral | threshold | Binary search | Exact | High (Parseval) |
| Quantization | block_size / n_bits | Closed-form | Exact (except overhead) | Moderate (high-rate approx) |
| Structural | density/block_size | Closed-form | Exact | Moderate |
| Physics | grid_size/n_particles | Binary search | Moderate | Low (empirical) |
| Entropy | precision | N/A (lossy first) | Moderate (entropy bound) | N/A (lossless) |

---

*This research document defines the architecture for dynamically tunable compression in SpectralStream. Implementation follows the 6-phase plan detailed above.*
