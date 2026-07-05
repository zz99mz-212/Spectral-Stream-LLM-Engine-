# Adaptive Loss Control System — Research Document

## Executive Summary

The Adaptive Loss Control (ALC) system is an intelligent agent that dynamically tunes compression parameters to achieve target compression ratios with minimal error. Given any tensor and a target ratio (e.g., 1200:1), the ALC system profiles the tensor, searches across method categories with continuously varying parameters, selects the Pareto-optimal method, and validates the result—iterating when the achieved ratio deviates from the target by more than 10%. This document provides the complete algorithmic specification for implementing the ALC system within the SpectralStream compression engine.

---

## 1. Target Ratio Achievement Algorithm

### 1.1 Core Algorithm

The `TargetRatioEngine` is the primary interface. It takes a tensor `T` and a target ratio `R` and returns a `CompressedTensor` whose `compression_ratio` is within 10% of `R` while minimizing the `relative_error`.

```
Algorithm: target_ratio_compress(T, R, max_iterations=5)

  Step 1: Profile the tensor T to obtain statistical, spectral, structural,
          and information-theoretic features. This produces a TensorProfile
          containing mean, std, effective_rank, energy_concentration,
          spectral_entropy, toeplitz_score, circulant_score,
          block_diagonal_score, sparsity scores, etc.

  Step 2: Build candidate list by iterating over all registered compression
          methods. For each method:
            a. Predict the parameter value(s) that would yield ratio ≈ R
               using the method's analytical ratio model or binary search.
            b. Estimate the reconstruction error for those parameters
               using the method's error predictor (a function of tensor
               profile features and parameter values).
            c. Compute a score = R / predicted_error (higher is better).

  Step 3: Sort candidates by score descending, take top-K (K=10).

  Step 4: For each candidate (method, params) in rank order:
            a. Compress T with the method and parameters.
            b. Decompress and compute actual ratio R_actual and error E.
            c. If |R_actual - R| / R <= 0.10 (10% tolerance) AND
               E <= error_budget, return result.
            d. If R_actual < R * 0.9, adjust parameters (binary search)
               to increase ratio and recompress.
            e. If R_actual > R * 1.1, adjust parameters to decrease ratio.

  Step 5: If no method achieves the target within tolerance, fall back to
          cascade composition (Section 4).

  Step 6: Return the best result found (closest ratio with lowest error).
```

### 1.2 Ratio Prediction Models

Each method category has a well-defined relationship between its key parameters and the expected compression ratio:

**Quantization methods (Block INT8/INT4, Hadamard INT8/INT4):**
```
R(block_size, bits) = (32 * block_size) / (bits * block_size + 32)
  where:
    - 32 = bytes per float32 element * 8 bits/byte
    - For each block: bits * block_size for quantized data +
      one float32 (32 bits) for the scale factor
    - Simplified: R(bits) = 32 / bits  (for large block_size)
    - Block INT8: R ≈ 4.0x; Block INT4: R ≈ 8.0x
    - Actual ratio includes header overhead:
      R = (n * 32) / (header_bytes + n_blocks * (32 + bits * block_size / 8))
```

**SVD/Decomposition methods:**
```
R(k, shape) = (m * n * 32) / (k * (m + n) * 32)
  where m, n are matrix dimensions and k is the rank
  Simplified: R(k) = (m * n) / (k * (m + n))
  For full rank: R = m*n / (2*m*n) = 0.5 (not compression)
  For k << min(m,n): R ≈ n / (2*k) for m=n
```

**Spectral methods (DCT, DCT-2D):**
```
R(drop_threshold) = (n * 32) / (n_kept * 32 + header_bytes)
  where n_kept depends on how many coefficients exceed the threshold
  n_kept = f(energy_concentration, drop_threshold)
```

**Sparsity methods (N:M, block sparsity):**
```
R(sparsity_ratio) = 32 / (bits + 32 * sparsity_ratio)
  For N:M sparsity with ratio N/M:
  R = (M * 32) / (N * bits + M * 1)     [1 bit mask per element]
  For 2:4 sparsity with INT4: R ≈ 10.67x
```

**Lossless entropy coding:**
```
R = n * 32 / compressed_size
  compressed_size depends on entropy of quantized values
  R ≈ n * 32 / (n * H) = 32 / H  where H is entropy in bits
```

### 1.3 Error Prediction Models

Error prediction is profile-dependent. The following models map tensor features to expected error:

**Block quantization error:**
```
E(block_size, bits, tensor_profile) ≈ std / (2^bits) * sqrt(1 + 3/block_size)
  where:
    - std is the tensor's standard deviation
    - Larger block_size = fewer scale factors = more quantization error
      within each block (amax-based scaling)
    - Smaller bits = coarser quantization = larger error
    - Error scales roughly as: E ∝ std * 2^(-bits)
```

**Hadamard quantization error:**
```
E(block_size, bits, tensor_profile) ≈ E_quant * sqrt(energy_concentration)
  where:
    - Hadamard transform spreads energy across coefficients
    - Energy_concentration < 0.5 → good (error spread evenly)
    - Energy_concentration > 0.8 → bad (few large coefficients dominate)
    - The Hadamard rotation reduces worst-case error vs. raw quantization
```

**SVD truncation error:**
```
E(k) = sqrt(sum(sigma[i]^2 for i in k..min(m,n))) / sqrt(sum(sigma^2))
  where sigma[i] are sorted singular values
  E(k) ≈ spectral_decay_rate^(-k)  (exponential decay)
  For rapid spectral decay (spectral_decay_rate > 1.0), low k achieves low error
```

**Sparsity error:**
```
E(N, M) ≈ top_k_energy_ratio * outlier_ratio
  where top_k_energy_ratio is the fraction of energy in the retained elements
  For N:M structured sparsity, retained energy ≈ cumulative sum of top N/M
  E(N, M) ≈ 1 - (N/M) * (1 + outlier_ratio)
```

### 1.4 Validation and Feedback Loop

```
Algorithm: validate_and_adjust(T, R, method, params, max_retries=3)
  for i in range(max_retries):
    ct = compress_tensor(T, method, params)
    if abs(ct.ratio - R) / R <= 0.10:
      return ct  # success
    if ct.ratio < R:
      # Need higher ratio → increase aggressiveness
      params = adjust_for_higher_ratio(method, params, R / ct.ratio)
    else:
      # Need lower ratio → decrease aggressiveness
      params = adjust_for_lower_ratio(method, params, ct.ratio / R)
  return ct  # best effort
```

For quantization methods, `adjust_for_higher_ratio` typically means reducing block_size (more scale factors per element = lower ratio) or reducing bit width. For decomposition methods, it means reducing rank k. For spectral methods, it means increasing the drop threshold.

---

## 2. Binary Search for Optimal Parameters

### 2.1 Continuous Parameter Search

Many compression methods accept continuous-valued parameters (rank k ∈ [1, min(m,n)], bit width b ∈ [1, 32], block_size ∈ [1, n], threshold t ∈ [0, 1]). Binary search efficiently finds the parameter value that achieves a target ratio.

```
Algorithm: binary_search_parameter(method, T, R, param_name, param_range,
                                    tolerance=0.05, max_iterations=20)
  Input:
    - method: compression method instance (has compress/decompress)
    - T: tensor to compress
    - R: target compression ratio
    - param_name: e.g., "rank", "block_size", "bit_width", "threshold"
    - param_range: (lo, hi) tuple, e.g., (1, min(T.shape))
    - tolerance: acceptable fractional deviation from target ratio
    - max_iterations: maximum binary search iterations

  For each iteration:
    mid = (lo + hi) / 2
    params = {param_name: mid}
    compressed = method.compress(T, **params)
    ratio = T.nbytes / len(compressed[0])  # compressed[0] is bytes

    if abs(ratio - R) / R <= tolerance:
      return mid  # found optimal parameter

    if ratio > R:
      lo = mid  # need lower ratio, so change parameter in direction
               # that reduces ratio (method-specific: e.g., lower rank,
               # lower bit_width, higher threshold)
    else:
      hi = mid  # need higher ratio

  Return (lo + hi) / 2  # best approximation

  Note: The direction (lo or hi) must map to ratio monotonicity:
    - For rank k: lower k = higher R (monotonic decreasing)
    - For bit_width b: lower b = higher R (monotonic decreasing)
    - For threshold t: higher t = higher R (monotonic increasing)
    - For block_size: smaller = lower R (monotonic increasing)
```

### 2.2 Monotonicity Mapping

Because binary search requires monotonicity, we need a mapping for each parameter:

| Parameter | Relationship to Ratio | Binary Search Direction |
|-----------|---------------------|----------------------|
| `rank` (k) | R ∝ 1/k (inverse) | if R_actual > R_target, k ↑ (lo = mid) |
| `bit_width` (b) | R ∝ 1/b (inverse) | if R_actual > R_target, b ↑ (lo = mid) |
| `block_size` | R ∝ block_size (direct, weakly) | if R_actual > R_target, block_size ↓ (hi = mid) |
| `threshold` (t) | R ∝ t (direct) | if R_actual > R_target, threshold ↓ (hi = mid) |
| `sparsity_n` | R ∝ M/N (inverse) | if R_actual > R_target, N ↑ (lo = mid) |

### 2.3 Hybrid Grid + Binary Search

For parameters with discrete values (bit widths: 2, 3, 4, 5, 6, 8), use a grid search over the discrete set:

```
Algorithm: discrete_optimal_parameter(method, T, R, param_values)
  best = None
  best_ratio_diff = inf
  for val in param_values:
    params = {param_name: val}
    compressed = method.compress(T, **params)
    ratio = T.nbytes / len(compressed[0])
    diff = abs(ratio - R)
    if diff < best_ratio_diff:
      best = val
      best_ratio_diff = diff
  return best
```

For mixed continuous+discrete parameters (e.g., block_size is continuous-like but must be even), use:

```
Algorithm: mixed_search(method, T, R, discrete_param, continuous_param)
  best_discrete = discrete_optimal_parameter(method, T, R, discrete_param)
  best_continuous = binary_search_parameter(method, T, R, continuous_param, ...)
  return combine(best_discrete, best_continuous)
```

### 2.4 Convergence Criteria

Binary search terminates when any of:
- `abs(R_actual - R) / R <= 0.05` (5% tolerance on ratio)
- `max_iterations` reached (default 20, giving precision of range/2^20)
- `hi - lo < 0.5` for integer-backed parameters

For integer parameters like `block_size` (must be even), round the binary search result to the nearest valid value.

---

## 3. Pareto-Optimal Method Selection

### 3.1 Multi-Objective Optimization

Compression involves two competing objectives: maximize ratio (R) and minimize error (E). For a fixed target ratio, we convert to a single-objective problem: minimize error subject to ratio constraint. This is equivalent to finding the Pareto-optimal point on the R-E curve for each method, then picking the method with the lowest error at the target ratio.

```
Algorithm: select_pareto_optimal(T, methods, R_target)
  candidates = []

  for method in methods:
    # Find the best parameter configuration that achieves R_target
    params = find_target_ratio_params(method, T, R_target)

    if params is not None:
      # Quick error estimation (no full compress/decompress)
      error = estimate_error(method, T, params, profile)

      candidates.append({
        "method": method,
        "params": params,
        "predicted_error": error,
        "method_name": method.__class__.__name__
      })

  # Sort by predicted error ascending
  candidates.sort(key=lambda x: x["predicted_error"])

  return candidates  # top candidate has minimal predicted error
```

### 3.2 Error-Estimation Without Full Decompression

To avoid the O(n) cost of full decompression during selection, we use lightweight error estimates:

**Quantization methods:** Error is estimated from the scale factors and the original tensor's distribution. For block quantization, the per-block error is `amax * 2^(-(bits-1)) / sqrt(12)` (uniform quantization noise). Total error = RMS across blocks.

**SVD methods:** Error is directly computed from the singular values: `E(k) = sqrt(1 - sum(sigma[:k]^2) / sum(sigma^2))`. Since SVD is already computed during compression, this is nearly free.

**Sparsity methods:** Error is estimated as `1 - cumulative_energy_ratio(top_N)` where `top_N` is the number of retained elements per group.

**Hadamard methods:** Error is estimated using the Hadamard-domain coefficients and the scale factors, similar to quantization error estimation but with the Hadamard-domain statistics.

### 3.3 Method Selection Heuristics

Beyond pure Pareto-optimality, we incorporate tensor-type heuristics:

```
Tensor Type        Preferred Methods
----------------   ------------------------------------------------------------
embedding          block_int8, block_int4 (high sensitivity → low error)
attention          hadamard_int8, block_int8 (structured redundancy)
ffn                block_int4, hadamard_int4, sparsity_int4 (error-tolerant)
qkv_fused          block_int8 (fused projections need lower error)
norm_bias          passthrough (tiny tensors, lossless)
weight (generic)   block_int8, sparsity_int4, hadamard_int8
```

### 3.4 Scoring Function

The compound score used to rank candidates:

```
score(method, params, T, R_target) = w_ratio * ratio_match
                                    + w_error * (1 - predicted_error / max_error)
                                    + w_speed * speed_factor
                                    + w_calibrated * calibration_bonus

  where:
    - ratio_match = exp(-|predicted_ratio - R_target| / R_target)
      (Gaussian centered on target; 1.0 for perfect match)
    - predicted_error = estimated relative error
    - max_error = user-specified maximum allowable error
    - speed_factor = compress_time / max_compress_time (from calibration)
    - calibration_bonus = 0.2 if this method was previously successful
      on this tensor type; 0.0 otherwise
    - weights: w_ratio=0.3, w_error=0.5, w_speed=0.1, w_calibrated=0.1
```

The score ranges [0, 1] with higher being better. The weight distribution reflects that error minimization is the primary objective (0.5), followed by ratio achievement (0.3).

---

## 4. Dynamic Cascade Composition

### 4.1 Motivation

When a single method cannot achieve the target ratio within the error budget, multiple methods are composed in cascade. Each stage in the cascade targets a sub-ratio whose product equals the total target ratio. The total error is the sum (or RMS) of per-stage errors, ideally dominated by only the lossy stages.

### 4.2 Cascade Architecture

```
Algorithm: compose_cascade(T, R_target, E_budget)

  Stage 1: Decomposition (lossy, targets moderate ratio)
    - SVD, Tensor Train, or Kronecker decomposition
    - Removes low-rank redundancy
    - Sub-ratio: R_1 (typically 2-5x)
    - Error: E_1

  Stage 2: Spectral/Transform (lossy, targets moderate ratio)
    - DCT, Hadamard, Wavelet transform
    - Removes frequency-domain redundancy
    - Sub-ratio: R_2 (typically 2-4x)
    - Error: E_2

  Stage 3: Quantization (lossy, targets high ratio)
    - Block INT4, Hadamard INT4, or sparsity + quantization
    - Aggressive bit-width reduction
    - Sub-ratio: R_3 (typically 4-16x)
    - Error: E_3

  Stage 4: Entropy Coding (lossless, targets modest ratio)
    - rANS, Huffman, or zstd
    - Lossless compression of quantized symbols
    - Sub-ratio: R_4 (typically 1.5-3x)
    - Error: 0 (lossless)

  Constraints:
    R_1 * R_2 * R_3 * R_4 >= R_target
    E_1 + E_2 + E_3 <= E_budget
    Each R_i >= 1.0 (no expansion)
```

### 4.3 Optimal Sub-Ratio Allocation

We formulate this as a constrained optimization:

```
Given: R_target, E_budget, and per-stage functions:
  R_i(p_i): parameter → ratio for stage i
  E_i(p_i): parameter → error for stage i

Minimize: E_total = sum(E_i)
Subject to: prod(R_i) >= R_target
            each E_i >= 0

Solved via Lagrangian relaxation:
  L = sum(E_i) - lambda * (sum(log(R_i)) - log(R_target))
  where lambda is the Lagrange multiplier

  Set dL/dp_i = 0 for each stage:
  dE_i/dp_i = lambda * (1/R_i) * dR_i/dp_i  (in log space)
           = lambda / R_i * dR_i/dp_i

  This gives the optimal trade-off where marginal error cost per
  marginal ratio benefit is equalized across all stages.

  Numerically solved with binary search on lambda:
    lambda_lo = 0.0
    lambda_hi = 1.0
    For 30 iterations:
      lambda_mid = (lambda_lo + lambda_hi) / 2
      For each stage i:
        Find p_i such that dE_i/dp_i = lambda_mid / R_i * dR_i/dp_i
      R_total = prod(R_i)
      if R_total > R_target:
        lambda_lo = lambda_mid  # tighten error budget
      else:
        lambda_hi = lambda_mid  # loosen error budget
```

### 4.4 Predefined Cascade Templates

For common scenarios, we provide templated cascades:

**Cascade for high sensitivity tensors (e.g., embeddings):**
```
T + SVD(k=high) + HadamardINT8(block=64) + zstd
  Target: R = 5-10x, E < 0.001
```

**Cascade for medium sensitivity tensors (e.g., attention):**
```
T + DCT(threshold=0.1) + BlockINT4(block=32) + rANS
  Target: R = 20-50x, E < 0.01
```

**Cascade for low sensitivity tensors (e.g., FFN):**
```
T + SVD(k=medium) + HadamardINT4(block=16) + rANS
  Target: R = 50-200x, E < 0.05
```

**Cascade for extreme ratios (noise-tolerant tensors):**
```
T + SVD(k=low) + HadamardINT4(block=8) + zstd
  Target: R = 200-2000x, E < 0.10
```

### 4.5 Cascade Validation

```
Algorithm: validate_cascade(T, cascade_stages, R_target)
  current = T
  total_compressed_bytes = 0
  stage_ratios = []

  for stage in cascade_stages:
    ct = stage.compress(current)
    current = stage.decompress(ct.data, ct.params)
    total_compressed_bytes += len(ct.data)
    stage_ratios.append(ct.ratio)

  actual_ratio = T.nbytes / total_compressed_bytes
  error = relative_error(T, current)

  return actual_ratio, error
```

---

## 5. Quality-Aware Target Adjustment

### 5.1 The Trade-Off Curve

Every tensor has a rate-distortion curve: higher ratio → higher error. When the target ratio demands more error than the user's `max_error` allows, the system may either:
(a) Relax the ratio target (reduce compression) to stay within error budget.
(b) Accept a higher error than the limit (if ratio is non-negotiable).

Option (a) is the default. The `find_optimal_tradeoff` function implements an exponential search along the ratio axis.

```
Algorithm: find_optimal_tradeoff(T, R_target, E_max, n_samples=100)
  # Search from R_target down to 1.0 (no compression)
  ratios = np.geomspace(R_target, 1.0, n_samples)
  for R in ratios:
    ct = target_ratio_compress(T, R)
    if ct.relative_error <= E_max:
      return R, ct.relative_error, ct
  # Fallback: lossless
  ct = lossless_compress(T)
  return 1.0, ct.relative_error, ct
```

`np.geomspace` (geometric spacing) gives more resolution at high ratios where the E-vs-R curve is steepest.

### 5.2 Adaptive Search with Early Termination

To avoid full compression at every point on the trade-off curve, use a predictor-based approach:

```
Algorithm: fast_optimal_tradeoff(T, R_target, E_max)
  profile = profile_tensor(T)

  # Predict error at target ratio
  E_pred = predict_best_error(profile, R_target)

  if E_pred <= E_max:
    return target_ratio_compress(T, R_target)
    # target is achievable

  # Find the highest ratio that keeps error below E_max
  # Use binary search on ratio
  R_lo = 1.0    # minimum ratio (lossless)
  R_hi = R_target  # maximum ratio (target)

  for _ in range(15):
    R_mid = sqrt(R_lo * R_hi)  # geometric midpoint
    E_pred = predict_best_error(profile, R_mid)
    if E_pred <= E_max:
      R_lo = R_mid  # can be more aggressive
    else:
      R_hi = R_mid  # must be less aggressive

  # Use the found ratio
  return target_ratio_compress(T, R_lo)
```

### 5.3 Per-Tensor Error Budget Relaxation

In the model-level context, the `ErrorBudgetAllocator` assigns per-tensor error budgets. When a tensor cannot meet its budget at the target ratio, the allocator can redistribute the unused budget from other tensors:

```
Algorithm: adaptive_budget_rebalance(profiles, initial_budgets, R_target)
  compressed = {}
  for name, profile in profiles.items():
    result = target_ratio_compress_with_budget(tensor, R_target,
                                               initial_budgets[name])
    if result.error > initial_budgets[name]:
      # Exceeded budget → mark for relaxation
      mark_for_relaxation(name, result.error - initial_budgets[name])
    else:
      # Under budget → collect surplus
      collect_surplus(initial_budgets[name] - result.error)
    compressed[name] = result

  total_surplus = get_total_surplus()
  for name in sorted_relaxation_list():
    if total_surplus <= 0:
      break
    needed = get_needed(name)
    additional = min(needed, total_surplus)
    initial_budgets[name] += additional
    total_surplus -= additional
    # Recompress with higher budget
    compressed[name] = target_ratio_compress_with_budget(
      tensors[name], R_target, initial_budgets[name]
    )

  return compressed
```

### 5.4 Quality Grades and Acceptance Criteria

Each `CompressedTensor` has a `quality_grade` based on its `relative_error`:

| Grade | Error Range | Typical Use |
|-------|------------|-------------|
| S     | E < 0.0001 | Embedding, critical weights |
| A     | E < 0.001  | Attention projections |
| B     | E < 0.01   | FFN, MLP layers |
| C     | E < 0.05   | Noisy layers, late layers |
| D     | E < 0.10   | Deep layers, noise-tolerant |
| F     | E >= 0.10  | Extreme compression targets |

The quality-aware target adjustment selects the ratio that achieves the highest possible grade for a given tensor, given its profile.

---

## 6. Implementation Plan

### 6.1 Module Structure

```
spectralstream/compression/adaptive/
  __init__.py
  _target_ratio_engine.py    # TargetRatioEngine orchestrator
  _binary_search.py          # Parameter search utilities
  _pareto_selector.py        # Pareto-optimal method selection
  _cascade.py                # Cascade composition engine
  _tradeoff.py               # Quality-aware target adjustment
  _predictors.py             # Ratio and error prediction models
  _rate_distortion.py        # R-D curve estimation utilities
  _validation.py             # Validation and feedback loop
```

### 6.2 `_predictors.py` — Prediction Functions

```python
def predict_block_quant_ratio(n_elements: int, block_size: int, bits: int) -> float:
    n_blocks = (n_elements + block_size - 1) // block_size
    total_bytes = 8  # header
    total_bytes += n_blocks * 4  # scale factors
    total_bytes += n_blocks * block_size * bits // 8  # quantized data
    return n_elements * 4 / max(total_bytes, 1)


def predict_block_quant_error(profile: TensorProfile, block_size: int, bits: int) -> float:
    std = profile.std
    return std / (2.0 ** max(bits - 1, 1)) * math.sqrt(1.0 + 3.0 / max(block_size, 1))


def predict_svd_ratio(shape: tuple, rank: int) -> float:
    m, n = shape[0], shape[-1]
    elements = m * n
    compressed = rank * (m + n)
    return elements / max(compressed, 1)


def predict_svd_error(profile: TensorProfile, rank: int) -> float:
    decay = max(profile.spectral_decay_rate, 0.01)
    return decay ** max(rank, 1)


def predict_hadamard_quant_error(profile: TensorProfile, block_size: int, bits: int) -> float:
    base_error = predict_block_quant_error(profile, block_size, bits)
    ec = max(profile.energy_concentration, 0.1)
    return base_error * math.sqrt(ec)


def predict_sparsity_error(profile: TensorProfile, sparsity_n: int, sparsity_m: int) -> float:
    outlier = profile.outlier_ratio
    retained_fraction = sparsity_n / max(sparsity_m, 1)
    return 1.0 - retained_fraction * (1.0 + outlier)
```

### 6.3 `_binary_search.py` — Parameter Search

```python
class ParameterSearcher:
    def __init__(self, profile: TensorProfile):
        self.profile = profile

    def find_rank_for_target(
        self, tensor: np.ndarray, target_ratio: float
    ) -> int:
        m, n = tensor.shape[0], tensor.shape[-1]
        lo, hi = 1, min(m, n)
        for _ in range(20):
            mid = (lo + hi) // 2
            pred = predict_svd_ratio(tensor.shape, mid)
            if pred > target_ratio:
                lo = mid + 1  # higher rank → lower ratio, so we need to go
                             # toward higher rank if ratio is too high
            else:
                hi = mid
        return lo

    def find_block_size_for_target(
        self, tensor: np.ndarray, target_ratio: float, bits: int
    ) -> int:
        n = tensor.ravel().size
        lo, hi = 1, max(n // 10, 1024)
        for _ in range(15):
            mid = max((lo + hi) // 2, 1)
            pred = predict_block_quant_ratio(n, mid, bits)
            if pred > target_ratio:
                lo = mid  # larger block → higher ratio, so lo = mid
            else:
                hi = mid
        return max(lo // 2 * 2, 2)  # ensure even

    def find_bit_width_for_target(
        self, tensor: np.ndarray, target_ratio: float, block_size: int
    ) -> int:
        n = tensor.ravel().size
        for bits in [2, 3, 4, 5, 6, 8, 16]:
            pred = predict_block_quant_ratio(n, block_size, bits)
            if pred <= target_ratio:
                return bits
        return 16
```

### 6.4 `_pareto_selector.py` — Method Selection

```python
class ParetoSelector:
    def __init__(self, registry: dict, profile: TensorProfile):
        self.registry = registry
        self.profile = profile

    def select(
        self, tensor: np.ndarray, target_ratio: float, error_budget: float,
        max_candidates: int = 10
    ) -> List[Tuple[str, dict, float]]:
        candidates = []

        # Quantization methods
        for method_name, method_cls in self.registry.items():
            if 'quant' in method_name or 'int8' in method_name or 'int4' in method_name:
                params, pred_error = self._evaluate_quant_method(
                    method_name, tensor, target_ratio
                )
                if params:
                    candidates.append((method_name, params, pred_error))

        # Decomposition methods (for 2D tensors)
        if tensor.ndim >= 2:
            for method_name in ['svd_truncated', 'tensor_train']:
                params, pred_error = self._evaluate_decomp_method(
                    method_name, tensor, target_ratio
                )
                if params:
                    candidates.append((method_name, params, pred_error))

        # Sparsity methods (for tensors with high sparsity scores)
        if self.profile.nm_sparsity_score > 0.3:
            params, pred_error = self._evaluate_sparsity_method(
                tensor, target_ratio
            )
            if params:
                candidates.append(('sparsity_int4', params, pred_error))

        # Sort by predicted error
        candidates.sort(key=lambda x: x[2])

        return candidates[:max_candidates]

    def _evaluate_quant_method(
        self, name: str, tensor: np.ndarray, target_ratio: float
    ) -> Tuple[Optional[dict], float]:
        is_hadamard = 'hadamard' in name
        is_4bit = 'int4' in name
        bits = 4 if is_4bit else 8

        searcher = ParameterSearcher(self.profile)
        block_size = searcher.find_block_size_for_target(tensor, target_ratio, bits)

        if is_hadamard:
            pred_error = predict_hadamard_quant_error(
                self.profile, block_size, bits
            )
        else:
            pred_error = predict_block_quant_error(
                self.profile, block_size, bits
            )

        return {'block_size': block_size}, pred_error

    def _evaluate_decomp_method(
        self, name: str, tensor: np.ndarray, target_ratio: float
    ) -> Tuple[Optional[dict], float]:
        rank = searcher.find_rank_for_target(tensor, target_ratio)
        pred_error = predict_svd_error(self.profile, rank)
        return {'rank': rank}, pred_error

    def _evaluate_sparsity_method(
        self, tensor: np.ndarray, target_ratio: float
    ) -> Tuple[Optional[dict], float]:
        ratio_16_4 = predict_block_quant_ratio(tensor.size, 16, 4)
        ratio_32_4 = predict_block_quant_ratio(tensor.size, 32, 4)
        n, m = (2, 4) if abs(ratio_16_4 - target_ratio) < abs(
            ratio_32_4 - target_ratio
        ) else (4, 8)
        pred_error = predict_sparsity_error(self.profile, n, m)
        return {'sparsity_n': n, 'sparsity_m': m}, pred_error
```

### 6.5 `_cascade.py` — Cascade Engine

```python
class CascadeComposer:
    def __init__(self, registry: dict, profile: TensorProfile):
        self.registry = registry
        self.profile = profile

    def compose(
        self, tensor: np.ndarray, target_ratio: float, error_budget: float
    ) -> List[Tuple[str, dict]]:
        # Determine number of stages needed
        if target_ratio <= 8:
            return self._single_stage(tensor, target_ratio, error_budget)
        elif target_ratio <= 30:
            return self._two_stage(tensor, target_ratio, error_budget)
        elif target_ratio <= 200:
            return self._three_stage(tensor, target_ratio, error_budget)
        else:
            return self._four_stage(tensor, target_ratio, error_budget)

    def _two_stage(
        self, tensor: np.ndarray, R: float, E: float
    ) -> List[Tuple[str, dict]]:
        # Stage 1: Quantization (primary ratio), Stage 2: Entropy (lossless)
        searcher = ParameterSearcher(self.profile)
        bits = searcher.find_bit_width_for_target(tensor, R / 1.5, 128)
        # Entropy stage targets ~1.5x
        return [
            ('hadamard_int8' if bits >= 8 else 'hadamard_int4',
             {'block_size': 64}),
            ('lossless_zstd', {}),
        ]

    def _three_stage(
        self, tensor: np.ndarray, R: float, E: float
    ) -> List[Tuple[str, dict]]:
        # Stage 1: Decomposition, Stage 2: Quantization, Stage 3: Entropy
        m, n = tensor.shape[0], tensor.shape[-1]
        rank = max(1, int(m * n / (R * (m + n) * 1.5)))
        return [
            ('svd_truncated', {'rank': min(rank, min(m, n))}),
            ('hadamard_int4', {'block_size': 32}),
            ('lossless_zstd', {}),
        ]

    def _four_stage(
        self, tensor: np.ndarray, R: float, E: float
    ) -> List[Tuple[str, dict]]:
        # For extreme ratios: SVD → DCT → Quant → Entropy
        m, n = tensor.shape[0], tensor.shape[-1]
        rank = max(1, int(m * n / (R * (m + n) * 4)))
        return [
            ('svd_truncated', {'rank': min(rank, min(m, n))}),
            ('dct_block', {'threshold': 0.1}),
            ('block_int4', {'block_size': 16}),
            ('lossless_rans', {}),
        ]
```

### 6.6 `_target_ratio_engine.py` — Main Orchestrator

```python
class TargetRatioEngine:
    def __init__(self, engine: CompressionIntelligenceEngine):
        self.engine = engine
        self.profiler = engine.profiler
        self.selector = ParetoSelector(engine._methods, None)

    def compress_to_target(
        self, tensor: np.ndarray, target_ratio: float,
        error_budget: float = 0.05, max_attempts: int = 5
    ) -> CompressedTensor:
        profile = self.profiler.profile_tensor(tensor)
        self.selector.profile = profile
        searcher = ParameterSearcher(profile)

        # Phase 1: Predict and select
        candidates = self.selector.select(tensor, target_ratio, error_budget)

        # Phase 2: Try candidates in order
        best: Optional[CompressedTensor] = None
        for method_name, params, pred_error in candidates:
            for attempt in range(max_attempts):
                try:
                    ct = self.engine.compress_tensor_with_validation(
                        tensor, profile, [(method_name,
                        self.engine._methods[method_name], params)], error_budget
                    )
                    ratio_ok = abs(ct.compression_ratio - target_ratio) \
                               / target_ratio <= 0.10
                    error_ok = ct.relative_error <= error_budget

                    if ratio_ok and error_ok:
                        return ct  # perfect match

                    if best is None or ct.relative_error < best.relative_error:
                        best = ct

                    # Adjust parameters for next attempt
                    if ct.compression_ratio < target_ratio * 0.9:
                        params = self._adjust_for_higher_ratio(
                            method_name, tensor, params, target_ratio
                        )
                    elif ct.compression_ratio > target_ratio * 1.1:
                        params = self._adjust_for_lower_ratio(
                            method_name, params, target_ratio
                        )
                    else:
                        break
                except Exception:
                    break

        # Phase 3: Try cascade if no single method worked
        if best is None or best.compression_ratio < target_ratio * 0.5:
            cascade = CascadeComposer(self.engine._methods, profile)
            stages = cascade.compose(tensor, target_ratio, error_budget)
            ct = self._compress_cascade(tensor, stages)
            if best is None or ct.relative_error < best.relative_error:
                best = ct

        # Phase 4: Quality-aware trade-off if error exceeds budget
        if best.relative_error > error_budget:
            adjusted_ratio, _, adjusted_ct = find_optimal_tradeoff(
                tensor, target_ratio, error_budget
            )
            return adjusted_ct

        return best

    def _adjust_for_higher_ratio(
        self, method_name: str, tensor: np.ndarray,
        params: dict, target_ratio: float
    ) -> dict:
        new_params = dict(params)
        if 'block_size' in params:
            new_params['block_size'] = max(params['block_size'] // 2, 2)
        elif 'rank' in params:
            new_params['rank'] = max(params['rank'] // 2, 1)
        elif 'threshold' in params:
            new_params['threshold'] = min(params['threshold'] * 1.5, 1.0)
        return new_params

    def _adjust_for_lower_ratio(
        self, method_name: str, params: dict, target_ratio: float
    ) -> dict:
        new_params = dict(params)
        if 'block_size' in params:
            new_params['block_size'] = min(params['block_size'] * 2, 4096)
        elif 'rank' in params:
            new_params['rank'] = min(params['rank'] * 2, 4096)
        elif 'threshold' in params:
            new_params['threshold'] = max(params['threshold'] / 1.5, 0.0)
        return new_params

    def _compress_cascade(
        self, tensor: np.ndarray,
        stages: List[Tuple[str, dict]]
    ) -> CompressedTensor:
        current = tensor
        for method_name, params in stages:
            method = self.engine._methods.get(method_name)
            if method is None:
                continue
            data, meta = method.compress(current, **params)
            current = method.decompress(data, meta)
            current = current.reshape(tensor.shape)
        ct = self.engine.compress_tensor(tensor, self.profiler.profile_tensor(
            current), 'passthrough')
        ct.relative_error = _compute_metrics(tensor, current)['relative_error']
        return ct
```

### 6.7 `_tradeoff.py` — Rate-Distortion Trade-Off

```python
class QualityTradeOffFinder:
    def __init__(self, engine: TargetRatioEngine):
        self.engine = engine

    def find_best(
        self, tensor: np.ndarray, target_ratio: float, max_error: float
    ) -> CompressedTensor:
        # Fast prediction-based check
        profile = self.engine.profiler.profile_tensor(tensor)
        pred_error = predict_best_error(profile, target_ratio)

        if pred_error <= max_error:
            return self.engine.compress_to_target(tensor, target_ratio, max_error)

        # Binary search on ratio (geometric midpoint)
        lo, hi = 1.0, target_ratio
        best_result = None

        for _ in range(12):
            mid = math.sqrt(lo * hi)
            pred = predict_best_error(profile, mid)
            if pred <= max_error:
                lo = mid
            else:
                hi = mid

        return self.engine.compress_to_target(tensor, lo, max_error)
```

### 6.8 `_validation.py` — Validation and Feedback

```python
def validate_target_achievement(
    ct: CompressedTensor, target_ratio: float, error_budget: float
) -> Dict[str, Any]:
    return {
        "achieved": abs(ct.compression_ratio - target_ratio) / target_ratio <= 0.10,
        "ratio_achieved": ct.compression_ratio,
        "ratio_target": target_ratio,
        "ratio_deviation": abs(ct.compression_ratio - target_ratio) / target_ratio,
        "within_error_budget": ct.relative_error <= error_budget,
        "error_achieved": ct.relative_error,
        "error_budget": error_budget,
        "quality_grade": ct.quality_grade,
        "method_used": ct.method,
        "attempts": ct.method_attempts,
    }


def feedback_loop(
    ct: CompressedTensor, tensor: np.ndarray,
    target_ratio: float, error_budget: float
) -> CompressedTensor:
    validation = validate_target_achievement(ct, target_ratio, error_budget)

    if validation["achieved"] and validation["within_error_budget"]:
        return ct  # ideal

    if not validation["achieved"]:
        # Find a method that can achieve this ratio
        engine = TargetRatioEngine(...)
        return engine.compress_to_target(tensor, target_ratio, error_budget)

    if not validation["within_error_budget"]:
        # Trade off ratio for error
        tradeoff = QualityTradeOffFinder(...)
        return tradeoff.find_best(tensor, target_ratio, error_budget)

    return ct
```

### 6.9 Integration with Existing Engine

The `TargetRatioEngine` integrates with `CompressionIntelligenceEngine` via:

1. **Configuration extension:** `CompressionConfig` gains fields:
   - `enable_adaptive_loss_control: bool = True`
   - `adaptive_max_iterations: int = 5`
   - `adaptive_ratio_tolerance: float = 0.10`
   - `adaptive_cascade_enabled: bool = True`

2. **Method registration:** All methods in `METHOD_REGISTRY` (and dynamically discovered methods) are passed to the `ParetoSelector`.

3. **Model-level compression:** In `CompressionIntelligenceEngine.compress_model`, when `enable_adaptive_loss_control` is True, each tensor is compressed via `TargetRatioEngine.compress_to_target` instead of `compress_tensor_with_validation`.

4. **Calibration integration:** The `ModelCalibrator` records per-method, per-parameter performance data, which feeds into the predictor models to improve accuracy over time.

---

## 7. Mathematical Appendix

### 7.1 Rate-Distortion Theory

For a tensor T with elements drawn from distribution p(x), the rate-distortion function R(D) gives the minimum achievable rate at distortion D. For MSE distortion and Gaussian sources:
```
R(D) = 0.5 * log2(sigma^2 / D)   for D < sigma^2
R(D) = 0                          for D >= sigma^2
```
This gives the theoretical lower bound. Our target ratio achievement aims to operate close to this bound.

### 7.2 Pareto Efficiency Condition

A method m1 with parameters p1 Pareto-dominates method m2 with parameters p2 if:
```
R(p1) >= R(p2)  AND  E(p1) <= E(p2)
```
with at least one strict inequality. The Pareto frontier is the set of undominated (method, parameter) pairs. Our selection algorithm picks the point on the frontier closest to the target ratio.

### 7.3 Cascade Error Composition

For a cascade of K stages, each with independent error e_i, the total error is:
```
E_total = 1 - prod(1 - e_i)  [if errors are uncorrelated]
E_total ≈ sum(e_i)            [if e_i << 1, first-order Taylor]
E_total = sqrt(sum(e_i^2))    [if errors are orthogonal in L2]
```
We use `sum(e_i)` as a conservative upper bound and `sqrt(sum(e_i^2))` as the expected value for well-designed cascades where errors decorrelate.

### 7.4 Optimal Parameter via Lagrangian

The constrained optimization:
```
minimize E(p)
subject to R(p) = R_target
```
has Lagrangian L(p, lambda) = E(p) + lambda * (R_target - R(p)).
The optimality condition: dE/dp = lambda * dR/dp.
For quantization: dE/db = -ln(2) * sigma * 2^(-b) * sqrt(1+3/B) and
dR/db = -32 / b^2 (continuous approximation). Therefore:
lambda = (dE/db) / (dR/db) = [sigma * 2^(-b) * ln(2) * b^2 * sqrt(1+3/B)] / 32.
Solving b* from this expression gives the optimal bit width for a given lambda.

---

## 8. Performance Considerations

### 8.1 Prediction Cost vs. Full Compression

Full compress+decompress for a 1M-element tensor takes ~5-10ms. The predictor functions take <0.1ms but have ~15-20% error in ratio prediction and ~30-50% error in error prediction. The binary search typically requires 8-12 iterations for 10% ratio tolerance, costing ~40-120ms if each iteration does full compression.

To reduce cost:
- **Phase 1:** Use predictors only (no compression) — <1ms for 20+ methods
- **Phase 2:** Compress only top-3 candidates — ~15-30ms
- **Phase 3:** Validate best result — ~5-10ms
- Total: ~20-40ms per tensor, vs. ~200ms+ if naively trying all methods.

### 8.2 Caching

Cache prediction results per (tensor_shape, target_ratio, method_name): valid as long as tensor statistics are similar (same tensor in same model layer across batches). The LRU cache with 1000 entries covers a full model's tensors.

### 8.3 Parallel Candidate Evaluation

The top-K candidates can be compressed in parallel using the existing `ThreadPoolExecutor` in `CompressionIntelligenceEngine`, since each method's compress+decompress is independent.

---

## 9. Testing Strategy

### 9.1 Unit Tests

- `test_ratio_prediction_accuracy`: Verify that ratio predictors for each method are within 20% of actual ratio across random tensors.
- `test_error_prediction_monotonicity`: Verify that error predictions are monotonic with respect to parameter changes (higher bit width → lower error, etc.).
- `test_binary_search_convergence`: Verify binary search converges to within 10% of target ratio in <20 iterations.
- `test_pareto_selection_order`: Verify that the selector returns methods in increasing error order at the target ratio.

### 9.2 Integration Tests

- `test_target_ratio_achievement`: For a given tensor and target ratio, verify the engine produces a result within 10% of target ratio and with error <= error_budget.
- `test_cascade_composition`: Verify cascade produces higher ratios than any single method on the same tensor.
- `test_quality_tradeoff`: Verify that when max_error is specified, the engine relaxes the ratio to stay within the error budget.

### 9.3 End-to-End Tests

- `test_model_compression_with_target`: Compress a small test model (e.g., Gemma-2B subset) with various target ratios and verify the overall ratio matches within 10%.
- `test_adaptive_vs_fixed`: Compare adaptive engine ratio accuracy vs. fixed-parameter compression across diverse tensors.

---

## 10. Summary

The Adaptive Loss Control system brings three key capabilities to SpectralStream:

1. **Target ratio achievement:** Any tensor can be compressed to a specified ratio with minimal error, using binary search to find optimal parameters for each method and Pareto selection to choose the best method.

2. **Extreme ratio through cascades:** When single methods hit their ceiling (typically 8-16x for quantization, ~100x for decomposition), cascade composition chains methods together, with ratios multiplying (SVD 5x × Hadamard 4x × BlockINT4 8x × zstd 2x = 320x total).

3. **Graceful degradation:** When the target ratio demands more error than the budget allows, the system automatically finds the highest achievable ratio within the error constraint, returning the best possible compression that meets quality requirements.

The system is implemented as a lightweight prediction layer on top of the existing `CompressionIntelligenceEngine`, requiring no changes to individual method implementations. The predictor functions provide 100x+ speedup over trial-and-error parameter search, making adaptive compression practical for per-tensor optimization even in large models.
