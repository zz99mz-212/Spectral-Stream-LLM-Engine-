# Acceptable Loss Levels for LLM Weight Compression

## 1. Research Background: Error vs. Perplexity

### 1.1 Key Findings from Literature

| Paper | Weight Error | Perplexity Impact | Method | Ratio Achieved |
|-------|-------------|-------------------|--------|----------------|
| **GPTQ** (Frantar et al., 2023) | ~1% relative weight RMSE | ~+0.1 PPL (WikiText-2) | Optimal Brain Quantization, groupwise 3-4 bit | ~8-10x |
| **AWQ** (Lin et al., 2024) | ~1-2% per-group error | ~+0.1-0.3 PPL | Activation-aware scaling, 4-bit | ~8x |
| **GGUF Q4_K_M** | ~0.5% weight error | ~+0.05 PPL (llama.cpp reports) | Importance-aware 4-bit with k-quants | ~3.9x |
| **QuIP#** (Tseng et al., 2024) | ~0.3% error (incoherent) | ~+0.02 PPL | Incoherence processing + E8P lattice | ~8x |
| **AQLM** (Egiazarian et al., 2024) | ~2-3% codebook error | ~+0.3-0.8 PPL | Additive quantization, 2-bit | ~16x |
| **SqueezeLLM** (Kim et al., 2024) | ~0.8% error (sensitive) | ~+0.1 PPL | Non-uniform + sparse, 3-bit | ~10x |
| **SpQR** (Dettmers et al., 2023) | ~0.5% error | ~+0.03 PPL | Sparse-Quantized, 3-4 bit | ~10x |

### 1.2 Empirical Error→PPL Mapping

From cross-paper analysis of LLaMA-family models (7B-70B):

```
Error ≤ 0.01%  → PPL Δ ≈ +0.001  (effectively lossless)
Error ≤ 0.05%  → PPL Δ ≈ +0.005  (imperceptible)
Error ≤ 0.1%   → PPL Δ ≈ +0.01   (imperceptible in practice)
Error ≤ 0.5%   → PPL Δ ≈ +0.03   (detectable only in calibration set)
Error ≤ 1%     → PPL Δ ≈ +0.05-0.1 (minor degradation on hard tasks)
Error ≤ 2%     → PPL Δ ≈ +0.2-0.5 (noticeable degradation)
Error ≤ 5%     → PPL Δ ≈ +1-3     (significant degradation)
Error > 5%     → PPL Δ > +3       (likely model collapse on some tasks)
```

**Key insight:** Weight RMSE and perplexity follow a roughly quadratic relationship for small errors:
```
ΔPPL ≈ c · (weight_error)²
```
where c ≈ 0.1-0.5 for most layers, meaning halving the weight error reduces PPL impact by 4×.

## 2. Per-Layer Sensitivity Analysis

### 2.1 Sensitivity Ranking (Most to Least)

| Layer Type | Sensitivity | Why | Acceptable Error Target |
|------------|-------------|-----|------------------------|
| **Embedding** | Very High (1.0) | 1:1 error propagation to every token's input representation. Errors compound across layers. | < 0.05% |
| **LM Head / Output** | Very High (1.0) | Directly determines token probabilities. Small errors shift the logit distribution, affecting sampling quality. | < 0.05% |
| **Attention Q projection** | High (0.9) | Query distribution drives attention patterns. Errors in Q change which tokens are attended to. | < 0.1% |
| **Attention O projection** | High (0.9) | Output of attention mechanism; errors propagate directly to residual stream. | < 0.1% |
| **Attention V projection** | Medium-High (0.8) | Value content; moderate sensitivity as attention weights filter values anyway. | < 0.2% |
| **Attention K projection** | Medium (0.75) | Key vectors; attention is somewhat robust to key errors due to softmax normalization. | < 0.5% |
| **FFN Down projection** | Medium-High (0.85) | Output of FFN; directly added to residual stream. Sensitive but FFN has redundancy. | < 0.2% |
| **FFN Gate projection** | Medium (0.7) | Gating mechanism; sigmoid/gelu nonlinearity provides some error tolerance. | < 0.5% |
| **FFN Up projection** | Medium (0.65) | Intermediate expansion; high redundancy in wide FFN layers. | < 0.5% |
| **QKV fused** | Medium (0.8) | Composite; treat as average of Q/K/V sensitivities. | < 0.2% |
| **Layer norms (RMS/LN)** | Low (0.3) | Scale+shift parameters; very robust. Can often be passed through losslessly or compressed aggressively. | < 5% or passthrough |
| **Bias terms** | Low (0.3) | Single scalars; negligible size. Always lossless. | < 5% or passthrough |

### 2.2 Sensitivity→Error Budget Mapping

```
sensitivity  → max_acceptable_error
1.0          → 0.0005  (0.05%)
0.9-0.99     → 0.001   (0.1%)
0.8-0.89     → 0.002   (0.2%)
0.7-0.79     → 0.005   (0.5%)
0.6-0.69     → 0.005   (0.5%)
0.5-0.59     → 0.01    (1.0%)
0.4-0.49     → 0.01    (1.0%)
0.3-0.39     → 0.05    (5.0%)
< 0.3        → 0.10    (10.0%)
```

### 2.3 Tensor-Type Error Budgets (Final Table)

```python
ACCEPTABLE_ERRORS = {
    "embedding":     0.0005,  # 0.05% — extremely sensitive
    "output":        0.0005,  # 0.05% — LM head
    "attention_q":   0.001,   # 0.1%
    "attention_o":   0.001,   # 0.1%
    "attention_v":   0.002,   # 0.2%
    "attention_k":   0.005,   # 0.5%
    "qkv_fused":     0.002,   # 0.2%
    "ffn_down":      0.002,   # 0.2%
    "ffn_gate":      0.005,   # 0.5%
    "ffn_up":        0.005,   # 0.5%
    "norm":          0.05,    # 5.0% — very robust
    "weight":        0.005,   # 0.5% — generic fallback
}
```

## 3. Quality Tiers

### 3.1 Tier Definitions

| Tier | Name | Max Error | PPL Impact | Use Case |
|------|------|-----------|------------|----------|
| **S** | Lossless | < 0.01% | ΔPPL < +0.001 | Critical: embeddings, LM head, benchmark evaluation |
| **A** | High Fidelity | < 0.1% | ΔPPL < +0.01 | Most layers, production-grade compression |
| **B** | Good | < 1% | ΔPPL < +0.1 | FFN layers, low-sensitivity weights |
| **C** | Acceptable | < 5% | ΔPPL < +1 | Norms, biases, experimental compression |
| **D** | Degraded | > 5% | ΔPPL > +1 | Extreme ratios, ablation studies only |

### 3.2 Per-Tensor-Type Tier Recommendations

| Tensor Type | Min Acceptable Tier | Recommended Tier | Max Useful Tier |
|-------------|--------------------|-----------------|-----------------|
| embedding | A | S | S |
| output/lm_head | A | S | S |
| attention_q | A | A | S |
| attention_o | A | A | S |
| attention_v | B | A | S |
| attention_k | B | B | A |
| ffn_down | B | A | S |
| ffn_gate | C | B | A |
| ffn_up | C | B | A |
| norm | D | C | B |
| bias | D | D | C |

## 4. Compressibility vs. Sensitivity

### 4.1 The Fundamental Trade-off

```
High Compressibility ∩ Low Sensitivity  →  Aggressive compression (FFN gate/up)
Low Compressibility ∩ High Sensitivity   →  Conservative compression (embeddings)
```

Tensors that are both high-sensitivity and high-compressibility (attention O projection with low effective rank) are the ideal candidates for Tier 1 methods (decomposition).

### 4.2 SVD Decomposition Error Bounds

For SVD with rank `k` truncation on a weight matrix `W ∈ R^{m×n}`:

```
relative_error = sqrt( Σ_{i=k+1}^{r} σ_i² / Σ_{i=1}^{r} σ_i² )
```

| Spectral Decay Rate | Rank Kept | Typical Error | Tier |
|---------------------|-----------|---------------|------|
| Fast (α > 2.0) | 1% of min(m,n) | < 0.05% | S |
| Fast (α > 1.5) | 5% of min(m,n) | < 0.1% | A |
| Moderate (α > 0.8) | 10% of min(m,n) | < 0.5% | B |
| Slow (α > 0.3) | 25% of min(m,n) | < 1% | B |
| Very Slow (α < 0.3) | 50% of min(m,n) | < 5% | D |

### 4.3 Quantization Error Bounds

For block quantization with block size `B` and bit width `b`:

```
relative_error ≈ σ_avg · 2^{-(b-1)} · sqrt(1/12 + 3/(12·B))
```

| Bits | Block Size | Typical Error (σ=0.1) | Tier |
|------|------------|----------------------|------|
| 8 | 128 | ~0.03% | S |
| 8 | 32 | ~0.02% | S |
| 6 | 128 | ~0.12% | A |
| 4 | 128 | ~0.5% | B |
| 4 | 32 | ~0.35% | A |
| 3 | 128 | ~1.0% | C |
| 2 | 128 | ~2.0% | C |

## 5. The Auto-Discovery Philosophy

### 5.1 Principle

**Do not hardcode targets. Let the engine discover them.**

The engine should:

1. **Start from maximum compression** for each tensor
2. **Measure the error** of each compression attempt
3. **Compare against per-tensor-type acceptable error** (defined above)
4. **Dial back compression** until error is within bounds
5. **Record the discovered optimal** ratio for future reference

### 5.2 Algorithm

```python
def auto_discover_optimal(tensor, tensor_type):
    """
    Find the maximum compression ratio while keeping error
    below the acceptable threshold for this tensor type.
    
    No hardcoded ratio target — the tensor itself determines
    how much compression it can tolerate.
    """
    acceptable_error = ACCEPTABLE_ERRORS.get(tensor_type, 0.005)
    
    # Try Tier 1 methods first (decomposition, spectral)
    for method in TIER1_METHODS:
        params = method.most_aggressive_params()
        result = method.compress_with_validation(tensor, **params)
        
        if result.error <= acceptable_error:
            return result  # max compression within budget
        
        # Error too high — binary search for optimal params
        optimal = method.binary_search_params(
            tensor, target_error=acceptable_error
        )
        if optimal and optimal.error <= acceptable_error:
            return optimal
    
    # Fall through to Tier 2, 3, 4, 5
    # ...
    
    # Last resort: quantization
    return quantize_to_budget(tensor, acceptable_error)
```

### 5.3 Binary Search for Optimal Parameters

Given a method with parameter `p` (rank, block_size, threshold, etc.) and the constraint `error ≤ acceptable_error`:

```
1. Start at most aggressive p_max (minimum rank, smallest block, highest threshold)
2. If error > acceptable_error, reduce aggression
3. Binary search between p_max and p_min (conservative)
4. Return the most aggressive p that satisfies error ≤ acceptable_error
```

This finds the Pareto-optimal point on the rate-distortion curve for each tensor individually.

## 6. Summary

| Metric | Value | Meaning |
|--------|-------|---------|
| Lossless bound | < 0.01% relative error | No measurable quality impact |
| High-fidelity bound | < 0.1% relative error | ~+0.01 PPL, imperceptible |
| Acceptable bound | < 1% relative error | ~+0.1 PPL, minor degradation |
| Degraded bound | < 5% relative error | ~+1 PPL, noticeable |
| Embedding/LM Head | 0.05% max error | Conservative required |
| Attention Q/O | 0.1% max error | Moderate compression |
| Attention K/V | 0.2-0.5% max error | More aggressive OK |
| FFN Gate/Up | 0.5% max error | Most aggressive safe targets |
| Norm/Bias | 5%+ or passthrough | Nearly irrelevant |

The fundamental insight: **per-tensor-type acceptable error budgets replace hardcoded ratio targets**, enabling true auto-discovery of optimal compression for each tensor independently.
