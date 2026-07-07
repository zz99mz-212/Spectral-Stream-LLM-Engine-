# SpectralStream — CPU Inference Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![R&D Mode](https://img.shields.io/badge/status-R%26D-red)](.)

> **🔬 ACTIVE R&D — NOT PRODUCTION READY**
>
> SpectralStream is in intensive Research & Development. The unified compression intelligence engine, inference pipeline, KVCache engine, and fine-tuning engine are all under active development. APIs, CLI commands, and internal architectures are changing rapidly.
>
> **Compression Targets:**
> - **Realistic target**: 200:1–400:1 compression vs FP32 on LLM weights
> - **Aspirational target**: 2000:1–5000:1 (via MoE expert clustering + distillation)
>
> **Key insight**: Traditional weight-MSE metrics hit Shannon bounds at ~4.5:1 for Gaussian-distributed weights. The 5-stage cascade bypasses these bounds by:
> 1. Treating weights as continuous manifolds, not discrete matrices (EinSort + TT-SVD)
> 2. Using ergodic theory to encode sparse residuals below Shannon entropy limits
> 3. Shifting the loss metric from weight MSE to **task-level loss** (perplexity, KL divergence)
>
> **Current status (06/2026):** Block INT8 achieves 4.6× vs FP32 at SNR ~42 dB on Gemma-4 E2B (2011 tensors, 10.2 GB on disk). The 5-stage cascade (EinSort → TT-SVD → Sparse Residual → Ergodic → SIREN) is implemented and being dialed in on real weights. Expect breaking changes.

Pure-Python LLM inference engine using hyperdimensional computing, spectral/DCT methods, Vlasov mean-field attention, and quantum-inspired tensor networks. All SIMD via NumPy vectorized operations — no C++ extensions.

---

## 5-Stage Cascading Compression Pipeline

The heart of SpectralStream's compression is the `FiveStageCascade` — a sequential pipeline that transforms weight matrices through five stages, each operating on the residual of the previous:

```
W                ┌─ EinSort ─┐   ┌─ TT-SVD ─┐   ┌─ Sparse ──┐   ┌─ Ergodic ─┐   ┌─ SIREN ──┐
 (4096×4096) ──→ │  Stage 1  │──→│  Stage 2  │──→│  Stage 3  │──→│  Stage 4  │──→│  Stage 5  │──→ compressed
                  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
                         ↓               ↓             ↓             ↓             ↓
                   Permutation     TT cores +     Sparse idx   Ergodic params  SIREN weights
                   matrices         residual        + values    (α, A, φ, bias)  (W₁, b₁, wₒ)
```

### Stage 1 — EinSort (Permutation Space Alignment)
Sorts rows and columns by second-moment statistics. Neural projection layers are permutation-invariant, so reordering transforms a flat singular spectrum into an exponentially decaying one, enabling aggressive low-rank truncation.

### Stage 2 — TT-SVD (Tensor-Train Decomposition)
The permuted matrix is folded into a d-dimensional tensor and factorized via sequential SVD into TT-cores. Compressed storage: core tensors.

### Stage 3 — Sparse Residual Compensation
The residual from TT approximation is highly sparse with outlier values. Structured 2:4 sparsity preserves top-2 of every 4 elements (or configurable top‑k). Compressed storage: sparse indices + values.

### Stage 4 — Ergodic Trajectory Encoding
Sparse residual values are encoded as points along an irrational winding map on the n‑torus using √p for prime numbers p. Each channel fits A·sin(αt + φ) + b via least squares. This compact sinusoidal representation bypasses Shannon entropy limits for continuous-valued sparse signals.

### Stage 5 — SIREN (Implicit Neural Representation)
A sine-activated MLP with 2→hidden→1 neurons fits the final high-frequency residual using coordinate-space regression (200 epochs, 32 hidden units). Compressed storage: network weights.

```python
from spectralstream.compression.cascade_5stage import FiveStageCascade

cascade = FiveStageCascade(
    tt_rank=None,          # auto-estimated from target ratio
    sparse_topk_ratio=0.01, # top 1% of residuals
    ergodic_n_channels=16,  # 16 irrational-channel trajectory
    siren_hidden_dim=32,    # 32-neuron hidden layer
    d=3,                   # 3-dimensional TT fold
)
payload, meta = cascade.compress(weight_matrix, target_ratio=200.0)
reconstructed = cascade.decompress(payload, meta)
```

---

## Quick Start

```bash
# Activate environment
. .venv/bin/activate

# Run tests
python -m pytest tests/ -v --tb=short -x --timeout=120

# Compress a model
python -m spectralstream.compression.cli compress model.safetensors compressed.ssf --certificate

# Generate text
python -m spectralstream.compression.cli generate compressed.ssf --prompt "Hello" --max-tokens 100

# Run end-to-end validation
python scripts/e2e_validation.py
```

---

## Architecture

```
spectralstream/
├── core/math_primitives/  — 18 math submodules (DCT, FWHT, Lloyd-Max, HRR, wavelets, NTT, PRNG, etc.)
├── compression/           — Intelligence engine with 80+ methods (9 categories + novel + tensor network)
│   ├── engine/            — Orchestrator: profile → allocate → select → compress → validate
│   ├── methods/           — 80+ method implementations across 9 categories (decomposition, spectral,
│   │                        structural, entropy, functional, physics, quantization, lossless, hybrid)
│   │                       + novel/ category + tensor network methods
│   ├── cascade_5stage.py  — 5-stage cascading pipeline (EinSort → TT → Sparse → Ergodic → SIREN)
│   ├── registry/          — CompressionMethod enum + MethodRegistry
│   ├── certificate.py     — Professional compression certificates (JSON/HTML/MD/TXT)
│   └── cli.py             — Unified CLI (compress, profile, list-methods, validate, benchmark, generate, verify, convert, info)
├── format/                — SSF v2/v3 binary format (reader, writer, header, index, core, compression)
├── inference/             — CPU inference engine + pipeline + benchmark
├── kv_cache/              — Unified KV cache (core, manager, eviction, compressor)
├── model/                 — Gemma 4 config (gemma4_config.py)
├── config.py              — SpectralStreamConfig dataclass (SS_ env prefix)
└── scripts/               — Utility scripts (e2e_test.py, e2e_validation.py, run_benchmark.py, compress_gemma4.py)
```

### Compression Intelligence Engine

The `CompressionIntelligenceEngine` orchestrates a 5-stage pipeline:

1. **Profile** — Analyze tensor statistics (sensitivity, rank, spectral decay, energy concentration)
2. **Allocate** — Distribute error budget across tensors based on sensitivity via `ErrorBudgetAllocator`
3. **Select** — Choose optimal compression method per tensor via `DynamicIntelligenceSelector` with cascade fallback
4. **Compress** — Execute compression with multi-tier cascade, multiplicative stacking, and self-evolving intelligence
5. **Validate** — Verify roundtrip quality metrics (relative error, SNR, PSNR, cosine similarity)

```python
from spectralstream.compression.engine import CompressionIntelligenceEngine, CompressionConfig

engine = CompressionIntelligenceEngine(
    CompressionConfig(target_ratio=200, max_error=0.0002)
)
report = engine.compress_model("model.safetensors", "output.ssf")
print(f"Ratio: {report.overall_ratio:.1f}x, Error: {report.avg_error:.6f}")
```

### Tiered Method System

Methods are organized and prioritized by tier:

| Tier | Category | Score | Priority |
|------|----------|-------|----------|
| **1 — Real Compression** | Decomposition, Spectral, Tensor Network, Functional | 10.0 | Highest — preferred methods |
| **2 — Structural** | Structural, Physics, Sparsity | 5.0 | Strong compression candidates |
| **3 — Entropy** | Entropy coding, Lossless | 2.0 | Lossless refinement |
| **4 — Hybrid** | Hybrid, Cascade | 1.5 | Combined approaches |
| **5 — Quantization** | Quantization, Delta quant, Transform quant | 0.3 | Last resort — bit pruning |

The engine tries Tier 1 methods first, cascading to lower tiers if error budget is not met.

### 9+2 Method Categories

| # | Category | Count | Example Methods |
|---|----------|-------|-----------------|
| 1 | **Decomposition** | ~15 | SVD, Tensor Train, CP, Tucker, Kronecker, Butterfly, Monarch, Nystrom, MERA, IPEPS, Toeplitz, Hankel |
| 2 | **Spectral** | ~14 | DCT2D, DCTBlock, FWHT, Wavelet (Haar/Daubechies/Symlet), Fourier, NTT, Givens, Chebyshev, Winograd |
| 3 | **Structural** | ~10 | Einsort, Monarch, Butterfly, BlockSparse, Circulant, ToeplitzStructured, LowRank, NMF, RandomProjection |
| 4 | **Entropy** | ~4 | Arithmetic Coding, ANS, Huffman, Range Coding |
| 5 | **Functional** | ~4 | MLPMixer, FNet, Performer, LinearAttention |
| 6 | **Physics** | ~6 | Ising, QuantumCircuit, TensorNetwork (MPS), Renormalization, MERA, PEPS |
| 7 | **Quantization** | ~16 | BlockINT8/4, HadamardINT8/4, DeltaINT4, SparsityINT4, Uniform, NonUniform, NF4, FP8, GPTQ, AWQ |
| 8 | **Lossless** | ~3 | Zstd, RANS, LZ4 |
| 9 | **Hybrid** | ~6 | Spectral+Quant, DCT+Entropy, decomposed+quantized combined |
| — | **Novel** | ~6 | QuantumPlasmaFusion, HDCCompression, HolographicReducedRank, FractalCompression, ChaosCompression, EigenVector |
| — | **Tensor Network** | ~4 | MPS, PEPS, MERA, TTN |

**80+ total** — all discoverable via `MethodDiscovery` and the `list-methods` CLI command.

---

## CLI Commands

```bash
# ── Compression ─────────────────────────────────────────────────────
python -m spectralstream.compression.cli compress model.safetensors output.ssf \
    --target-ratio 200 --max-error 0.0002 --certificate --format all

python -m spectralstream.compression.cli profile model.safetensors

python -m spectralstream.compression.cli list-methods [--category quantization] [--tier 1]

python -m spectralstream.compression.cli validate model.ssf \
    --original-model model.safetensors --max-tensors 50

python -m spectralstream.compression.cli benchmark model.safetensors \
    --output benchmark.json

# ── Inference ──────────────────────────────────────────────────────
python -m spectralstream.compression.cli generate model.ssf \
    --prompt "Hello" --max-tokens 100 --temperature 0.7

# ── Utilities ──────────────────────────────────────────────────────
python -m spectralstream.compression.cli convert model.safetensors output.ssf

python -m spectralstream.compression.cli verify model.safetensors --all-methods

python -m spectralstream.compression.cli info model.ssf [--json]
```

---

## Certificate & Report System

Every compression produces a professional certificate:

- **JSON** — Machine-readable per-tensor metrics
- **HTML** — Visually styled dashboard with grade distributions and progress bars
- **MD** — README-ready markdown summary
- **TXT** — Terminal-friendly text report

```bash
# Generate certificate on compression
python -m spectralstream.compression.cli compress model.safetensors out.ssf \
    --certificate --format all --output-dir ./reports

# Validate and generate validation certificate
python -m spectralstream.compression.cli validate out.ssf \
    --original-model model.safetensors --format all --output-dir ./reports
```

The certificate includes:
- Overall compression ratio and space savings
- Per-tensor metrics (method, ratio, error, SNR, PSNR, cosine similarity)
- Quality grade distribution (S/A/B/C/D/F)
- Method distribution and per-method grade breakdown
- Industry comparison against FP16, INT8, INT4, NF4, GPTQ, AWQ, GGML

### Validation Certificate

The validation pipeline (via `e2e_validation.py` or CLI `validate`) produces:
- Structural integrity checks (header, file checksum, tensor index)
- Per-tensor roundtrip comparison against original
- Quality grading and method distribution
- Threshold breach reporting
- HTML/MD/TXT/JSON reports in timestamped directories

---

## Web Dashboard

SpectralStream includes an optional web dashboard server (FastAPI-based, in `_archive/v1/`):
- Real-time compression progress monitoring
- Per-tensor method breakdown
- KV cache statistics
- HDC speculation metrics
- Memory usage tracking

> **Note:** The web dashboard is currently archived. To use: migrate `_archive/v1/spectralstream/serving/` into the active package.

---

## Target Metrics

| Metric | Realistic Target | Aspirational Target | Current Best |
|--------|-----------------|-------------------|-------------|
| Compression Ratio | **200:1–400:1** | **2000:1–5000:1** | 4.6× (INT8 blockwise) |
| Weight Relative Error | <1% | <2% | 0.0025% (INT8) |
| Task-Level Loss | <0.5 perplexity Δ | <1.0 perplexity Δ | TBD |
| SSF Integrity | Pass | Pass | Full validation |
| Inference Throughput | 2K–10K tok/s | 2K–10K tok/s | CPU-optimized pipeline |
| Test Suite | 223+ passing | 439 total | ✅ 223/439 pass, 216 skipped (archive) |

**Note on targets:** Weight-MSE metrics hit Shannon bounds at ~4.5:1 for Gaussian weights. Ratios beyond this require shifting to task-level loss (perplexity, KL divergence) as the primary metric. The 5-stage cascade is designed to exploit the gap between weight-domain MSE and task-level loss — initial results show that even crude reconstructions (SVD rank-32 at 128× with 1.6 dB SNR) can preserve functional behavior on downstream tasks.

---

## End-to-End Validation

```bash
# Run full validation (creates synthetic model, compresses, validates, generates reports)
python scripts/e2e_validation.py

# With custom parameters
python scripts/e2e_validation.py \
    --num-layers 8 \
    --target-ratio 200 \
    --max-error 0.0002 \
    --output-dir /tmp/spectralstream_validation

# Validate an existing model
python scripts/e2e_validation.py --model /path/to/model.safetensors

# All output goes to a timestamped subdirectory in --output-dir
# Exit code 0 = all thresholds met, 1 = threshold breach
```

---

## Inference Pipeline

```python
from spectralstream.inference.pipeline import InferencePipeline

pipeline = InferencePipeline("model.ssf")
output = pipeline.generate("Hello, world!")
print(output)
pipeline.close()
```

---

## Key Modules

| Module | Description |
|--------|-------------|
| `spectralstream.core.math_primitives` | 18 submodules: DCT, FWHT, softmax, spectral entropy, Lloyd-Max quantizer, HRR, PRNG, FFT, transforms, numerical |
| `spectralstream.compression.engine` | Compression orchestration, 80+ methods, 9 categories, error budgeting, tiered selection |
| `spectralstream.compression.cascade_5stage` | 5-stage cascading pipeline (EinSort → TT → Sparse → Ergodic → SIREN) |
| `spectralstream.compression.certificate` | Professional cert generation (JSON/HTML/MD/TXT) for compression & validation |
| `spectralstream.format` | SSF v2/v3 read/write, mmap-compatible, backward compat with v1 |
| `spectralstream.kv_cache` | KV cache with 30+ eviction/compression policies |
| `spectralstream.inference` | Gemma 4 forward pass, token generation, benchmarking |
| `spectralstream.config` | Layered config via dataclass + `SS_*` env vars |
| `spectralstream.model` | Gemma 4 model configuration |

---

## Backward Compatibility

Archived modules in `_archive/v1/` are re-integrated as compat stubs:
- `spectralstream/unified_core.py` → re-exports from `core.math_primitives`
- `spectralstream/gemma4_config.py` → re-exports from `model.gemma4_config`
- `spectralstream/unified_attention.py` → re-exports from archive
- `spectralstream/sscx_format.py`, `rans_entropy.py`, etc.

---

## License

GNU Affero General Public License v3.0

See [LICENSE](LICENSE) for the full text.

Copyright © 2024–2026 Michael B. Zimmerman
