# KV Cache Compression: Formulas, Algorithms & Implementation Guide

## Table of Contents
1. [KIVI — Asymmetric Per-Channel/Per-Token Quantization](#1-kivi-icml-2024)
2. [GEAR — Quantization + Low-Rank + Sparse Error Recovery](#2-gear-2024)
3. [FreqKV — DCT Along Sequence Dimension](#3-freqkv-iclr-2026)
4. [eOptShrinkQ — Optimal Spectral Denoising](#4-eoptshrinkq-2026)
5. [StreamingLLM — Attention Sinks](#5-streamingllm-iclr-2024)
6. [Multiplicative Combination](#6-multiplicative-combination)
7. [Integration with Autoregressive (AR) Models](#7-integration-with-ar-models)

---

## 1. KIVI (ICML 2024)

**Paper:** `arxiv:2402.02750` • **Repo:** `github.com/jy-yuan/KIVI`

### Core Algorithm

KIVI uses asymmetric quantization: **key cache per-channel**, **value cache per-token**.

#### 1.1 Quantization Primitive

Given tensor $X \in \mathbb{R}^{l \times d}$ and bit-width $B$:

**Group-wise asymmetric quantization (per-group):**
$$Q(X) = \left\lfloor \frac{X - z_X}{s_X} \right\rceil, \quad X' = Q(X) \cdot s_X + z_X$$

where per-group $j$:
$$s_j = \frac{\max(X_{\mathcal{G}_j}) - \min(X_{\mathcal{G}_j})}{2^B - 1}, \quad z_j = \min(X_{\mathcal{G}_j})$$

Group size $G$ controls granularity (KIVI default: $G=32$).

#### 1.2 Key Cache: Per-Channel Quantization

For key cache $X_K \in \mathbb{R}^{l \times d}$:

- Group along **channel dimension**: channel $c$'s $l$ tokens form one group
- Rationale: keys have **fixed outlier channels** (consistent across tokens)
- Per-channel quantization **confines error** to each channel, preventing cross-channel contamination

Error comparison (Llama-2-13B, 2-bit):
$$\frac{\|X_K - X_K'\|_F}{\|X_K\|_F} = \begin{cases} 4.55\% & \text{per-channel} \\ 13.67\% & \text{per-token} \end{cases}$$

$$\frac{\|A - A'\|_F}{\|A\|_F} = \begin{cases} 9.60\% & \text{per-channel} \\ 47.00\% & \text{per-token} \end{cases}$$

#### 1.3 Value Cache: Per-Token Quantization

For value cache $X_V \in \mathbb{R}^{l \times d}$:

- Group along **token dimension**: token $t$'s $d$ channels form one group
- Rationale: attention output is $A X_V$, a **weighted sum of token rows**.
  Per-token quantization keeps error confined per-row; attention sparsity ($>84\%$) means most rows contribute $\approx 0$ error weight.

$$[A X_V]_{i*} = \sum_{j=1}^{l} A_{ij} [X_V]_{j*}$$
Per-token error in output:
$$\Delta = \frac{\|A X_V - A X_V'\|_F}{\|A X_V\|_F} = \begin{cases} 3.55 & \text{per-token} \\ 49.89 & \text{per-channel} \end{cases}$$

#### 1.4 Streaming Data Structure

Split KV cache into **grouped** (quantized) + **residual** (FP16):

```
X_K_split:  X_Kg = X_K[:l-r]     (grouped, quantized per-channel)
            X_Kr = X_K[l-r:]     (residual, FP16, ≤ R tokens)
```

- New tokens $\mathbf{t}_K$ appended to $X_{Kr}$
- When $|X_{Kr}| = R$ (residual length, typically 32-128), quantize and merge into $X_{Kg}$
- $R$ must be divisible by $G$

Attention computation with tiled matmul:
$$A_g = \mathbf{t}_Q Q(X_{Kg}^\top), \quad A_r = \mathbf{t}_Q X_{Kr}^\top, \quad A = \text{Concat}([A_g, A_r])$$

Same for value cache (but per-token quantization, so queue-based: oldest token popped, quantized, appended to $X_{Vg}$).

#### 1.5 Algorithm Pseudocode

```
Algorithm: KIVI Prefill + Decode
Input: group size G, residual length R, bit-width B

Prefill:
  for each layer:
    X_K, X_V = compute_kv(input)
    X_Kg = chunk_quantize_per_channel(X_K, G, B)
    X_Kr = empty
    X_Vg = chunk_quantize_per_token(X_V, G, B)
    X_Vr = empty

Decode step t:
  t_K, t_V = compute_kv(token_t)
  X_Kr = concat(X_Kr, t_K)
  X_V  = concat(X_V, t_V)           # V is a deque
  X_Vr.append(t_V)

  if |X_Kr| == R:
    X_Kg = concat(X_Kg, quantize_per_channel(X_Kr, G, B))
    X_Kr = empty

  if |X_Vr| == R:
    oldest = X_Vr.pop()
    X_Vg = concat(X_Vg, quantize_per_token(oldest, G, B))

  # Attention with mixed precision
  A = softmax( t_Q @ dequant(X_Kg, X_Kr)^T / sqrt(d) )
  O = A @ dequant(X_Vg, X_Vr)
```

#### 1.6 Compression Ratio

$$\text{Ratio} = \frac{32 \text{ bits} \times (l \times d \times 2)}{B \times (l \times d \times 2) + \text{metadata}} \approx \frac{32}{B}$$

| Bit-width | From FP32 | From FP16 | Metadata overhead |
|-----------|-----------|-----------|-------------------|
| 2-bit     | 16:1      | 8:1       | ~5% (scales+zero-points) |
| 4-bit     | 8:1       | 4:1       | ~3% |

Peak memory (including model weights): **2.6× reduction** → up to **4× larger batch size** → **2.35-3.47× throughput**.

#### 1.7 AR Model Integration

- **Per-step cost:** Only the *residual* (≤R tokens) needs dequantization per step
- **Grouped portion:** Fused dequantization matmul kernel (Q_MatMul in KIVI's CUDA implementation)
- **No fine-tuning required** — plug-and-play
- Compatible with weight-only quantization (e.g., GPTQ, AWQ)

---

## 2. GEAR (2024)

**Paper:** `arxiv:2403.05527` • **Repo:** `github.com/HaoKang-Timmy/GEAR`

### Core Algorithm

GEAR decomposes KV matrix $X$ into three components:

$$X \approx \hat{D} + L + S$$

Where:
- $\hat{D}$ = quantized backbone (ultra-low precision, e.g., 2-4 bit)
- $L$ = low-rank matrix (captures coherent quantization error)
- $S$ = sparse matrix (captures outlier residuals)

#### 2.1 Step 1: Outlier-Aware Quantization

Remove outliers **before** quantization to reduce dynamic range:

For key cache (per-channel): extract top/bottom $\frac{s}{2}\%$ entries per channel
For value cache (per-token): extract top/bottom $\frac{s}{2}\%$ entries per token

$$S = \text{Filter}_s(X), \quad \hat{D} = \text{Quant}_b(X - S)$$

where:
$$\text{Filter}_s(X)_{ij} = \begin{cases}
X_{ij} & \text{if } X = K \text{ and } X_{ij} \text{ in top/bottom } \frac{s}{2}\% \text{ of channel } j \\
X_{ij} & \text{if } X = V \text{ and } X_{ij} \text{ in top/bottom } \frac{s}{2}\% \text{ of token } i \\
0 & \text{otherwise}
\end{cases}$$

The quantization backbone uses KIVI-style per-channel K / per-token V (denoted KCVT variant).

#### 2.2 Step 2: Low-Rank Approximation of Residual

After removing $S$ and quantizing $D$, the residual is:

$$R = X - (\hat{D} + S)$$

Reshape $R$ into per-head matrices $\{R_h \in \mathbb{R}^{n \times d_H}\}_{h=1}^H$.

**Key observation:** The residual spectrum decays rapidly (Figure 2b of paper) — it has coherent structure.

Apply SVD to each head's residual:
$$R_h = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{m}_i^\top$$

Low-rank approximation (rank $r$):
$$L_h = \text{SVDSolver}_r(R_h) = A_h B_h^\top, \quad A_h \in \mathbb{R}^{n \times r}, B_h \in \mathbb{R}^{d_H \times r}$$

$L = \text{Concat}(L_1, \ldots, L_H)$

#### 2.3 Step 3: Sparse Recovery of Outliers

The sparse matrix $S$ (from step 1) captures individual outlier entries that:
- Dominate the quantization error if included in $\hat{D}$
- Are too few to justify a full low-rank component

Typical density: $s \approx 0.1\text{-}1\%$ of entries.

#### 2.4 Combined Decomposition Algorithm

```
Algorithm: GEAR Compress(X, b, r, s)
  # X ∈ {K, V} from one layer, all heads concatenated
  # b = quantization bit-width, r = low-rank rank, s = outlier fraction

  Step 1: Extract sparse outliers
  S = Filter_s(X)                     # per-channel for K, per-token for V

  Step 2: Quantize remaining
  D_hat = Quant_b(X - S)              # KIVI-style per-channel/per-token

  Step 3: Compute residual
  R = X - (D_hat + S)                 # quantization error residual

  Step 4: Low-rank approximation (per head)
  for each head h:
    R_h = R[:, h*d_H:(h+1)*d_H]
    U_h, Σ_h, V_h = svd(R_h)
    A_h = U_h[:, :r] @ diag(Σ_h[:r])
    B_h = V_h[:, :r]
    L_h = A_h @ B_h^T

  L = concat(L_1, ..., L_H)

  Step 5: Store {D_hat, L, S}
  # D_hat in b-bit, L in FP16 (rank r), S in FP16 (sparse)
```

#### 2.5 Storage & Compression Ratio

Each component's storage:

| Component | Storage per element | Total for $X \in \mathbb{R}^{n \times d}$, $H$ heads |
|-----------|-------------------|-----------------------------------------------------|
| $\hat{D}$ | $b$ bits | $b \times n \times d$ bits |
| $L$ | $r(n + d_H)$ FP16 per head | $2 \times 16 \times H \times r \times (n + d_H)$ bits |
| $S$ | $s \times n \times d$ FP16 | $16 \times s \times n \times d$ bits |

Effective bits per entry:
$$b_{\text{eff}} = b + \frac{16 \times H \times r \times (n + d_H)}{n \times d} + 16s$$

For typical settings ($b=4, r=64, s=0.01, n=4096, d=4096, H=32, d_H=128$):
$$b_{\text{eff}} = 4 + \frac{16 \times 32 \times 64 \times (4096 + 128)}{4096 \times 4096} + 0.16 \approx 4 + 1.03 + 0.16 \approx 5.2$$

Ratio from FP32: $32 / 5.2 \approx 6.15:1$.

#### 2.6 Error Bounds

The paper shows GEAR reduces approximation error by **10-100×** compared to quantization-only methods. For LLaMA3-8B on GSM8K:

| Method | Approximation Error (Frobenius) | GSM8K Accuracy |
|--------|--------------------------------|-----------------|
| FP16 (uncompressed) | 0 | 42.5% |
| KIVI 2-bit | 0.24 | 18.7% |
| GEAR 2-bit | 0.05 | 38.2% |
| GEAR 4-bit | 0.01 | 41.8% |

#### 2.7 AR Model Integration

- **Streaming buffer:** Store last $n_b$ tokens (e.g., 20) in FP16. Compress every $n_b$ steps.
- **Online SVD update:** Power iteration for incremental low-rank update (avoids full SVD each step)
- **Attention with GEAR:** $Q \cdot (\hat{D} + L + S)^\top$ computed tile-wise with fused kernels
- **GEAR-L variant:** Skip sparse $S$ for 2× speed, still 5× error reduction over quantization alone

---

## 3. FreqKV (ICLR 2026)

**Paper:** `arxiv:2505.00570`

### Core Algorithm

FreqKV compresses KV cache **along the sequence dimension** using DCT (Discrete Cosine Transform).

#### 3.1 Why DCT?

KV cache along sequence length $L$ exhibits strong temporal correlation — adjacent tokens produce similar KV vectors. Energy concentrates in **low-frequency DCT components**.

For sequence $x[0], x[1], \ldots, x[L-1]$, DCT-II transforms to frequency domain:
$$X[k] = \sum_{n=0}^{L-1} x[n] \cos\left(\frac{\pi}{L}\left(n + \frac{1}{2}\right)k\right), \quad k = 0, \ldots, L-1$$

Inverse DCT (IDCT):
$$x[n] = \frac{1}{L} \sum_{k=0}^{L-1} w_k X[k] \cos\left(\frac{\pi}{L}\left(n + \frac{1}{2}\right)k\right)$$

where $w_0 = 1$, $w_k = 2$ for $k > 0$.

**Spectral truncation:** retain only top $k$ coefficients (lowest frequencies):
$$k = \left\lfloor \alpha \times L \right\rfloor, \quad \alpha \in (0, 1]$$

Compression ratio along sequence: $1/\alpha$.

#### 3.2 Iterative DCT for AutoRegressive Decoding

The challenge: DCT requires the **entire sequence** — but in AR decoding, tokens arrive one at a time.

FreqKV uses **iterative DCT update**:

**At prefill time** (length $L_P$):
1. Compute DCT of full prefill KV: $X_P = \text{DCT}(K_V\text{ or }V_V)$
2. Retain top $\alpha L_P$ coefficients
3. Store DCT coefficients + basis

**At decode time** (new token at position $L_P + t$):
1. Keep a sliding window of last $W$ tokens in FP16 (residual buffer)
2. Every $N$ steps, re-DCT the full accumulated sequence:
   $$X_{\text{full}} = \text{Concat}(X_{\text{stored\_DCT}}, X_{\text{buffer}})$$
3. Merge by recomputing DCT on full sequence, re-truncate

Computational cost: $O(L \log L)$ per re-DCT, done every $N$ steps amortizes to $O(N \log L)$ per step.

#### 3.3 Update Rule (Approximate)

For efficiency, approximate update without full re-DCT:

For new token $x_{L+1}$ at sequence position $L+1$:
$$X_{\text{new}}[k] \approx X_{\text{old}}[k] + x_{L+1} \cos\left(\frac{\pi}{L+1}\left(L + \frac{1}{2}\right)k\right)$$

This avoids recomputing the full DCT, but introduces approximation error that grows with sequence length. Periodically reset by full DCT.

#### 3.4 Compression Ratios

| $\alpha$ (retained fraction) | Along-sequence ratio | Combined × 2-bit quant |
|------------------------------|---------------------|----------------------|
| 0.25 (25%) | 4:1 | 64:1 |
| 0.125 (12.5%) | 8:1 | 128:1 |
| 0.0625 (6.25%) | 16:1 | 256:1 |

#### 3.5 Quality Impact

- LLaMA-2-7B extended to **256K tokens** with stable perplexity
- Minimal training (8K length) sufficient for the extrapolation
- Perplexity increase at 8:1 spectral compression: <0.5 PPL

#### 3.6 AR Model Integration

```
Decode step t:
  t_K, t_V = compute_kv(token_t)
  
  # Method A: Periodic re-DCT (recommended for quality)
  buffer_K.append(t_K)
  buffer_V.append(t_V)
  
  if len(buffer) >= N:
    merged = concat(last_DCT_full, buffer)
    new_DCT_K = DCT(merged_K)[:k_coeffs]
    new_DCT_V = DCT(merged_V)[:k_coeffs]
    store(new_DCT_K, new_DCT_V)
    buffer = empty

  # Method B: Approximate update (faster, lower quality)
  delta_K = update_DCT(t_K, position)
  DCT_K = merge(DCT_K, delta_K)

  # Attention: reconstruct via IDCT
  K_full = IDCT(DCT_K)             # approximate full K  
  V_full = IDCT(DCT_V)             # approximate full V
  A = softmax(q @ K_full^T / sqrt(d))
  O = A @ V_full
```

**Key limitation:** DCT update adds $O(L \log L)$ complexity. Practical systems use **hybrid**: DCT-compress prefill KV, keep recent tokens in FP16, periodically re-DCT.

---

## 4. eOptShrinkQ (2026)

**Paper:** `arxiv:2605.02905`

### Core Algorithm

Two-stage pipeline:
1. **Spectral denoising** (eOptShrink) — extract low-rank shared structure
2. **Quantization** (TurboQuant) — quantize the isotropic residual

#### 4.1 Spiked Matrix Model for KV Cache

KV cache block $S \in \mathbb{R}^{n \times d}$ (n tokens, d head dim) decomposes as:

$$S = U + Z$$

where:
- $U = \sum_{i=1}^r d_i \mathbf{u}_i \mathbf{v}_i^\top$ = low-rank **shared context** (rank $r$)
- $Z$ = full-rank **per-token residual** (contains token-specific info)

The residual $Z$ satisfies the **thin shell property**: rows have concentrated norms and **delocalized coordinates** (no outliers), making it ideal for scalar quantization.

#### 4.2 eOptShrink: Optimal Singular Value Shrinkage

Given observed SVD: $S = \sum_{i=1}^{n \wedge d} \tilde{\sigma}_i \tilde{\boldsymbol{\xi}}_i \tilde{\boldsymbol{\zeta}}_i^\top$

Apply optimal shrinker $\varphi$ to each singular value:

$$\hat{U}_\varphi = \sum_{i=1}^{n \wedge d} \varphi(\tilde{\sigma}_i) \tilde{\boldsymbol{\xi}}_i \tilde{\boldsymbol{\zeta}}_i^\top$$

For Frobenius norm loss, the optimal shrinker is:

$$\varphi_i^* = \begin{cases}
d_i \sqrt{a_{1,i} a_{2,i}} & d_i > \alpha \text{ (detectable)} \\
0 & d_i \leq \alpha \text{ (undetectable)}
\end{cases}$$

where:
- $d_i$ = estimated signal strength
- $a_{1,i}, a_{2,i}$ = asymptotic singular vector overlaps (Eq. 6 in paper)
- $\alpha$ = BBP phase transition threshold (automatic from spectrum)

**Rank selection** via BBP transition: count singular values above bulk edge $\hat{\lambda}_+$:
$$\hat{r}^+ = |\{i: \tilde{\lambda}_i^2 / \hat{\lambda}_+ - 1 > d^{-1/3}\}|$$

Bulk edge estimate:
$$\hat{\lambda}_+ = \tilde{\lambda}_{k+1}^2 + \frac{1}{2^{2/3} - 1}(\tilde{\lambda}_{k+1}^2 - \tilde{\lambda}_{2k+1}^2)$$
where $k = \lfloor d^c \rfloor$, $c = \min(1/2.01, 1/\log\log d)$.

#### 4.3 TurboQuant for the Residual

After removing $\hat{U}$, the residual $R = S - \hat{U}$ is quantized via TurboQuant:

For each vector $x \in \mathbb{R}^d$ (row of $R$):
1. **Norm separation:** $r = \|x\|$, $u = x/r$
2. **Random rotation:** $z = \Pi u$ (Haar-distributed orthogonal matrix)
3. **Lloyd-Max quantization:** $\hat{z} = \mathcal{C}_b(z)$ (per-coordinate)
4. **Reconstruction:** $\hat{x} = r \cdot \Pi^\top \hat{z}$

**Inner product bias correction** (TQprod variant, optional):
$$\langle y, \hat{x} \rangle = \langle y, \hat{x}_{\text{MSE}} \rangle + \|r_x\| \cdot \frac{\sqrt{\pi/2}}{d} \langle \Phi y, \text{sign}(\Phi r_x) \rangle$$

But eOptShrink's key insight: **spectral denoising eliminates the need for this correction** — the residual is already isotropic, so bias is near-zero. This saves ~1 bit/entry.

#### 4.4 Complete Algorithm

```
Algorithm: eOptShrinkQ Compress
Input: KV block S ∈ ℝ^{n×d}, bit-width b, block size m (≈ d)

1. Partition S into blocks of m consecutive tokens
   For each block:
     Step 1: Compute SVD of S_block
     Step 2: Estimate bulk edge λ̂_+ and rank r̂^+ (Algorithm 2)
     Step 3: For each i ≤ r̂^+, compute φ̂_i via eOptShrink
     Step 4: Û = Σ φ̂_i · ξ̃_i · ζ̃_i^T
     Step 5: R = S_block - Û                       # isotropic residual
     Step 6: R_quant = TurboQuant(R, b)             # per-vector quantization
     Step 7: Store {Û (via SVD factors), R_quant}

2. Attention computation (decode):
   For query q:
     U_reconstructed = reconstructed from stored SVD factors
     R_reconstructed = TurboQuant_dequant(R_quant)
     K_approx = U_reconstructed + R_reconstructed
     score = q @ K_approx^T / sqrt(d)
```

#### 4.5 Compression Ratio

| Component | Bits | Note |
|-----------|------|------|
| Low-rank $U$ | $16 \times r \times (n + d)$ | FP16 SVD factors |
| Quantized $R$ | $b \times n \times d$ | b-bit TurboQuant |
| TurboQuant metadata | $16 \times n$ | scales per vector |

Effective bits per entry: $b_{\text{eff}} \approx b + \frac{16r}{d} + \frac{16r}{n}$

At $b=2.2$, typical effective ratio: **14.5:1 from FP32**.

#### 4.6 Theoretical Guarantees

1. **Automatic rank selection** via BBP phase transition — no hyperparameter
2. **Residual isotropy:** $\|R\|_F \approx \|Z\|_F$ with provably small IP bias
3. **Coordinate delocalization:** $\|r_t\|_\infty \leq C\sqrt{\log d / d} \cdot \|r_t\|_2$
4. **Near-zero inner product bias** — correction factor $1/(1 + \text{SNR}_t)$

#### 4.7 AR Model Integration

- **Prefill cost:** SVD on each block (expensive but one-time)
- **Decode:** New tokens appended to last block; when block fills, re-shrink
- **Online SVD update:** Can use power iteration for incremental update to avoid full SVD
- **Key advantage:** No outlier handling needed → simpler kernel implementation
- Empirically at 2.2 bpe matches FP16 multi-needle retrieval (denoising acts as regularizer)

---

## 5. StreamingLLM (ICLR 2024)

**Paper:** `arxiv:2309.17453` • **Repo:** `github.com/mit-han-lab/streaming-llm`

### Core Algorithm

Not a compression technique per se, but a **token eviction strategy** that enables infinite-length inference.

#### 5.1 Attention Sink Phenomenon

**Observation:** Initial tokens receive disproportionately high attention scores **regardless of semantic content**. These "attention sinks" exist because Softmax attention forces the model to distribute attention mass — even to meaningless tokens.

#### 5.2 Sliding Window + Sinks

StreamingLLM retains:
1. **Attention Sinks** (first 4 tokens) — mandatory
2. **Sliding Window** (last $W$ tokens, typically 2048-8192)
3. **Evicts everything in between**

#### 5.3 Why It Works

For windows attention where only $W$ recent KV are kept:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K_{\text{sink:window}}^\top}{\sqrt{d}}\right) V_{\text{sink:window}}$$

$$K_{\text{sink:window}} = [K_{\text{sink}}; K_{\text{recent}}] \in \mathbb{R}^{(4+W) \times d}$$

Keeping the sink tokens provides a "dumping ground" for attention mass that would otherwise be forced onto the window tokens, distorting their representations.

#### 5.4 Theoretical Explanation

For softmax attention, the sum of attention scores over all keys is 1. When the model processes tokens far from training length:

$$\sum_{j=1}^{4+W} A_{ij} = 1 \implies A_{ij}^{\text{(recent)}} = 1 - \sum_{k \in \text{sink}} A_{ik}$$

Without sinks, the attention mass concentrates on the first window token, causing output corruption. With sinks, the mass is absorbed by the initial placeholder tokens.

#### 5.5 Pre-Training Enhancement

Add a **dedicated sink token** [SINK] at position 0 during pre-training (or SFT with 1K+ tokens). This:
- Provides a cleaner attention sink
- Improves streaming performance
- Adds negligible train-time cost

#### 5.6 Compression Impact

Not directly a KV cache compressor, but enables infinite-length inference with **bounded memory**:

$$\text{Memory} = (4 + W) \times d \times 2 \text{ bytes} \times L_{\text{layers}} \times 2 \text{ (K+V)}$$

At $W = 2048, d = 4096, L=32$: ≈ **1.07 GB** for any sequence length.

Combine with quantization: store sinks + window in 2-4 bits → **0.13-0.27 GB**.

#### 5.7 AR Model Integration

```
StreamingLLM Decode:
  Input: current token t, sinks S (4 tokens), window W (tokens)
  
  t_K, t_V = compute_kv(t)
  window.append(t_K, t_V)
  
  if len(window) > W:
    window.pop_earliest()             # evict oldest (not sink!)
  
  KV_cache = concat(sinks, window)    # in FP16 or quantized
  A = attention(q, KV_cache)
  
  # For DCT-hybrid: evicted tokens can be summarized spectrally
  # (See FreqKV integration below)
```

---

## 6. Multiplicative Combination

### 6.1 Orthogonal Compression Axes

The key insight: each technique compresses along a different (approximately orthogonal) axis:

| Axis | Technique | Factor |
|------|-----------|--------|
| Element depth (bits) | KIVI/GEAR quantization | 8-16× (32→2-4 bits) |
| Sequence length (tokens) | FreqKV DCT truncation | 4-16× (retain 6-25%) |
| Head/neuron dimension | Low-rank (GEAR, eOptShrinkQ) | 2-4× (rank r ≪ d) |
| Token retention | StreamingLLM eviction | 256-1024× (inf ctx bounded) |

**Gains multiply** when axes are orthogonal because they compress independent dimensions of the KV tensor $(L \times d)$.

### 6.2 Production Pipeline

```
SpectralKVCache: Multiplicative Pipeline
══════════════════════════════════════════════

STAGE 1: STREAMINGLLM EVICTION (optional)
  Full sequence → Sinks(4) + Recent Window(W) + Evicted(middle)
  
STAGE 2: DCT ALONG SEQUENCE (FreqKV, optional)
  Sink + Window → DCT along L-dim → retain α fraction of coeffs
  
STAGE 3: SPECTRAL DENOISING (eOptShrinkQ, optional)
  DCT-compressed → SVD per head → extract low-rank U → isotropic residual R

STAGE 4: QUANTIZATION (KIVI + GEAR)
  Key residual → per-channel quantized D̂_K
  Value residual → per-token quantized D̂_V
  → Low-rank L for coherent residual error
  → Sparse S for outlier error

STAGE 5: STORAGE FORMAT
  { sinks(FP16), DCT_coeffs, low_rank_U, quant_D_K, quant_D_V, sparse_S }
```

### 6.3 Combined Ratio Calculation

Let $X \in \mathbb{R}^{L \times d}$ be original KV (32-bit float):

**Step 1: Sequence eviction** (StreamingLLM)
$$\text{Ratio}_1 = \frac{L}{4 + W}$$

**Step 2: Spectral truncation** (FreqKV)
$$\text{Ratio}_2 = \frac{1}{\alpha} \quad (\text{e.g., } \alpha = 0.125 \rightarrow 8:1)$$

*If both Stage 1 and 2: the DCT is applied to the 4+W retained tokens, so* $L_{\text{DCT}} = 4+W$, *truncated to* $\alpha(4+W)$.

**Step 3: Spectral denoising** (eOptShrinkQ) — reduces effective $\sigma^2$ needed for quant:

$$\text{Ratio}_3 = \frac{32}{b_{\text{eff}}} \quad \text{where } b_{\text{eff}} = b + \underbrace{\frac{16 r (n+d_H)}{n d}}_{\text{low-rank overhead}}$$

**Step 4: Quantization** (KIVI/GEAR):
$$\text{Ratio}_4 = \frac{32}{b_{\text{eff}}} = \frac{32}{b + 16 s + \frac{16 H r (n+d_H)}{n d}}$$

**Combined (multiplicative):**
$$\text{Ratio}_{\text{total}} = \text{Ratio}_1 \times \text{Ratio}_2 \times \text{Ratio}_4$$

Example:
- No StreamingLLM (need full context): Ratio₁ = 1
- FreqKV α = 0.125: Ratio₂ = 8
- KIVI 2-bit + GEAR (r=64, s=0.01): b_eff ≈ 5.2, Ratio₄ = 6.15

**Total:** $1 \times 8 \times 6.15 = 49.2:1$

With StreamingLLM (W=4096, L=32768): Ratio₁ = 32768/4100 ≈ 8
**Total:** $8 \times 8 \times 6.15 = 394:1$

With cross-layer redundancy (layer L+1 KV predicted from layer L, 2×):
**Total:** $394 \times 2 = 788:1$

### 6.4 Error Accumulation Model

Each stage adds distortion. Under independence assumption:

$$\text{MSE}_{\text{total}} = \text{MSE}_{\text{DCT}} + \text{MSE}_{\text{shrink}} + \text{MSE}_{\text{quant}} + \text{MSE}_{\text{sparse}}$$

For AR models, error compounds across steps $t$:
$$\text{Logit Error}(t) \approx \sqrt{t} \cdot \sigma_{\text{MSE}} \cdot \|\text{attention Jacobian}\|$$

This is why **GEAR's error reduction** is critical for long chains — it reduces per-step $\sigma_{\text{MSE}}$ by 10-100× vs quantization alone.

### 6.5 Practical Implementation Strategy

**Phase 1** (1 month): KIVI-only (16:1)
**Phase 2** (2 months): KIVI + GEAR L+S (8-16:1, near-lossless)
**Phase 3** (3 months): + FreqKV DCT (64-128:1)
**Phase 4** (4 months): + eOptShrinkQ (100-200:1)
**Phase 5** (6 months): + cross-layer prediction + KVP eviction (200-500:1)
**Phase 6** (9 months): + holographic/HDC extreme compression (500-2000:1 R&D)

---

## 7. Integration with AR Models

### 7.1 The Autoregressive Challenge

AR generation: tokens produced one at a time, each depending on all previous tokens. The KV cache:
- **Grows** by 2 vectors per step
- Must be **read** every step (for attention)
- Error **compounds** across steps

### 7.2 Hybrid Two-Tier Architecture

```
┌─────────────────────────────────────────────────┐
│                ATTENTION COMPUTE                  │
│  Q @ K^T = Q @ [K_grouped_quant, K_residual]^T   │
│         = Q @ K_grouped_quant^T + Q @ K_residual^T│
│                                                   │
│  K_grouped_quant:   quantized, static (old tokens)│
│  K_residual:        FP16, sliding window (recent) │
└─────────────────────────────────────────────────┘
```

### 7.3 Per-Step Operations

| Operation | Cost | Technique |
|-----------|------|-----------|
| Append new K/V | $O(d)$ | FP16 buffer |
| Read K/V for attention | $O(Ld)$ | Fused dequant-matmul |
| Periodic re-compression | $O(L \log L + L d^2)$ | Every $N$ steps |
| SVD update (eOptShrinkQ) | $O(n d \min(n,d))$ | Every $N$ steps (or power iteration) |
| DCT update (FreqKV) | $O(L \log L)$ | Every $N$ steps |

### 7.4 Error Control for AR Models

**Theorem (Error compounding, informal):** For AR model with Lipschitz attention $\kappa$, the output error at step $t$ satisfies:

$$\|\Delta h_t\| \leq \kappa \sum_{s=1}^t \|\Delta \text{KV}_s\| \leq \kappa t \cdot \max_s \|\Delta \text{KV}_s\|$$

where $\Delta \text{KV}_s$ is compression error at step $s$.

**Mitigation strategies:**
1. **Keep recent tokens in FP16** (KIVI residual, StreamingLLM window)
2. **Periodic full recompute** of compressed representation
3. **Error feedback** (GEAR-style): maintain running estimate of cumulative error, correct when re-compressing
4. **Attention-aware bit allocation** (FlashCache, KVP): allocate more bits to tokens with higher attention scores

### 7.5 Predictive Coding Integration

Linear prediction across KV cache tokens:

$$\hat{K}_{t+1} = A \cdot K_t + b$$

where $A$ is learned from the low-rank subspace (captured by eOptShrinkQ's shared context).

Store: $\Delta K_{t+1} = K_{t+1} - \hat{K}_{t+1}$ (prediction residual, lower entropy → fewer bits).

Combined with DCT: predict in frequency domain → $\Delta \text{DCT}(K)$ requires even fewer coefficients.

This is underexplored in current literature and represents a **key research opportunity**.

---

*Sources: KIVI (2402.02750), GEAR (2403.05527), FreqKV (2505.00570, ICLR 2026), eOptShrinkQ (2605.02905), StreamingLLM (2309.17453, ICLR 2024). Prepared for SpectralStream R&D.*
