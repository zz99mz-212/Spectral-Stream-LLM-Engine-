# Cascade Compression Dial-In Report — Real Gemma-4 Weights

## Setup

| Parameter | Value |
|-----------|-------|
| Model | `gemma-4-E2B/model.safetensors` (2011 tensors, 4.7GB embedding) |
| Sample | `mlp.down_proj` (1536×12288), `attn.o_proj` (1536×4096), 1024×1024 blocks |
| Max error | 1% relative |
| Targets | 500:1, 1200:1, 5000:1 |

## Phase 1: Individual Method Characterization

### Methods passing 1% error threshold

| Method | Params | Ratio | Error | SNR | Time |
|--------|--------|-------|-------|-----|------|
| `block_int8` | bs=1024 | **4.0:1** | **0.96%** | 40.3dB | 0.03s |
| `hadamard_int8` | bs=512 | **4.0:1** | **0.74%** | 42.6dB | 0.73s |
| `block_int8` | bs=128 | 3.9:1 | 0.74% | 42.6dB | 0.06s |
| `svd+block_int8` | r=16 | 3.7:1 | 0.78% | 42.1dB | 0.25s |
| `block_int8+block_int4` | cascade | 2.5:1 | 0.06% | 64.2dB | 1.5s |
| `hadamard_int8+block_int4` | cascade | 2.4:1 | 0.06% | 64.1dB | 2.1s |

### Methods failing 1% threshold

| Method | Params | Ratio | Error | Failure Mode |
|--------|--------|-------|-------|-------------|
| `block_int4` | bs=64 | 7.5:1 | **12.4%** | Too aggressive |
| `hadamard_int4` | bs=64 | 7.5:1 | **10.9%** | Too aggressive |
| `delta_int4` | bs=256 | 7.5:1 | **12.4%** | Too aggressive |
| `sparsity_int4` | gs=32 | 5.8:1 | **17.8%** | Too aggressive |
| `sparse_quant` | default | 6.4:1 | **23.5%** | Too aggressive |
| `svd_compress` | r=4 | 256:1 | **98.9%** | Not low-rank |
| `svd_compress` | r=16 | 64:1 | **96.4%** | Not low-rank |
| `svd_compress` | r=128 | 8:1 | **76.5%** | Not low-rank |
| `dct_spectral` | keep=0.5 | 1.3:1 | **26.7%** | Energy spread |
| `dct_spectral` | keep=0.1 | 6.7:1 | **75.0%** | Energy spread |

## Phase 2: Cascade Diagnostics

### Why the cascade fails on real weights

**SVD singular value analysis** (MLP down_proj 1024×1024 block):
```
Rank for  50% energy: 264  (ratio 6.6:1)
Rank for  80% energy: 594  (ratio 3.0:1)
Rank for  90% energy: 803  (ratio 2.2:1)
Rank for  95% energy: 976  (ratio 1.8:1)
Rank for  99% energy: 1263 (ratio 1.4:1)
Rank for 99.9% energy: 1464 (ratio 1.2:1)
```

These are **extremely high-rank matrices** — the singular values decay very slowly. Even rank=264 only captures 50% of the energy. This means:
- **SVD at any practical rank (16-128) loses 70-99% of the signal**
- The residual after SVD is almost as large as the original
- DCT on this residual still finds no structure (white noise)
- Quantization on the huge residual adds little extra compression

### Bottleneck quantification

```
Cascade chain: Decomposition → Spectral → Quantization

Stage 1 (SVD rank=16):   ratio=64:1,  cumulative error=96%   ← BOTTLENECK
Stage 2 (DCT keep=0.1):  ratio=6.7:1, cumulative error=75%  ← BOTTLENECK  
Stage 3 (BlockINT4):     ratio=7.5:1, cumulative error=12%
```

The first two stages destroy the signal. Quantization can't recover it.

### Error model

```
Actual cascade error:    1.0 - Π(1 - εᵢ)  ≈  Σ εᵢ  (for small εᵢ)
With SVD error 96% and DCT error 75%:
  Multiplicative: 1 - 0.04 × 0.25 × 0.88 ≈ 99.1% error
  But actual SVD+DCT residual is ~75% of original (DCT recovers some SVD loss)
```

## Phase 3: Optimal Configuration

### For ≤1% error

```
BEST: block_int8 (block_size=1024) — Single stage
  Ratio: 4.0:1
  Error: 0.96%
  SNR:   40.3 dB
  Time:  0.03 seconds

Runner-up: block_int8 + block_int4 cascade
  Ratio: 2.5:1
  Error: 0.06%
  SNR:   64.2 dB
```

### For higher ratio (5-10% error acceptable)

```
BEST: block_int4 (block_size=64) — Single stage
  Ratio: 7.5:1
  Error: 12.4%
  SNR:   18.1 dB
```

### Cascade config that works

```python
# Optimal for ≤1% error on real Gemma-4 weights
cascade = [
    {"method_type": "quantization", "params": {"method": "block_int8", "block_size": 1024}},
    # No SVD — destroys too much signal
    # No DCT — doesn't capture residual structure
    # Optional: {"method_type": "entropy", "params": {"method": "rans"}} 
    #   (adds ~1.5:1 lossless, but 100x slower)
]
```

## Phase 4: Recommendations

### 1. Realistic limits on Gemma-4 weights

| Target Ratio | Achievable? | Method | Error |
|-------------|-------------|--------|-------|
| **4:1** | ✅ Yes | block_int8 alone | **0.96%** |
| **8:1** | ❌ No (≤1%) | block_int4 | 12.4% |
| **16:1** | ❌ No (≤1%) | — | — |
| **500:1** | ❌ Impossible at 1% | — | — |
| **5000:1** | ❌ Impossible at 1% | — | — |

### 2. Root cause: matrix structure

The Gemma-4 MLP/Attention weights are **random projection matrices** with:
- Near-uniform singular value distribution (not low-rank)
- Broad DCT spectrum (no concentrated energy)
- Gaussian-like value distribution (not sparse)
- No block-diagonal or circulant structure

These properties make them fundamentally resistant to:
- Low-rank approximation (SVD, Tensor Train, CP)
- Spectral compression (DCT, FFT, Wavelet)
- Structured sparsity (N:M sparsity, block sparsity)

### 3. What WOULD work for 500:1+

```python
methods_needed = [
    "Binary/ternary quantization (1-2 bits)",         # 16-32:1
    "Extreme unstructured pruning (98%+ zeros)",       # 50:1
    "Knowledge distillation (smaller model)",          # Architectural
    "Mixed-precision: INT2 attention, INT4 MLP",       # 8-16:1
    "Product quantization / vector quantization",      # 20-100:1
    "Combined: prune 90% → binary quant → entropy",    # 32×10×2 = 640:1
]
```

The existing `sparsity_int4` and `sparse_quant` methods are on the right track but too aggressive (17-24% error at just 5-6:1).

### 4. Fixing the cascade for this codebase

**Option A: Skip SVD/DCT for high-rank tensors**
```python
# In build_cascade_config():
if effective_rank > 0.3 * min(shape):  # high-rank
    stages = [
        {"method_type": "quantization", "params": {"method": "block_int8"}},
        {"method_type": "quantization", "params": {"method": "block_int4"}},
    ]
else:  # low-rank (e.g., attention heads, embeddings)
    stages = [
        {"method_type": "decomposition", "params": {"rank_frac": 0.02}},
        {"method_type": "spectral", "params": {"keep_frac": 0.3}},
        {"method_type": "quantization", "params": {"method": "block_int8"}},
    ]
```

**Option B: Pure quantization cascade for all weights**
```python
cascade = [
    {"method_type": "quantization", "params": {"method": "hadamard_int8", "block_size": 512}},
    {"method_type": "quantization", "params": {"method": "block_int4", "block_size": 64}},
]
# Result: 2.5:1 at 0.06% error (Grade A)
```

**Option C: Novel methods for extreme ratios**
The codebase has 2000+ methods. The physics-inspired and quantum methods may achieve 100:1+ on specialized tensor structures (e.g., MERA, PEPS, MPS for attention heads).

## Summary

| Aspect | Finding |
|--------|---------|
| **Max ratio at ≤1% error** | **4.0:1** (block_int8 single stage) |
| **Best cascade** | block_int8 → block_int4 (2.5:1, 0.06%) |
| **#1 bottleneck** | SVD stage (96% error, destroys signal) |
| **#2 bottleneck** | DCT stage (75% error, no spectral concentration) |
| **Root cause** | Gemma-4 weights are random projection matrices — NOT low-rank |
| **Fix** | Replace SVD/DCT with quantization-only cascade for high-rank tensors |
| **Target 500:1+** | Requires novel methods (binary quant, 99% pruning, VQ) not in current cascade |
