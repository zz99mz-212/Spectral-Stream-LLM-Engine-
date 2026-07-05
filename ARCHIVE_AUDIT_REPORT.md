# Archive Migration Audit Report

Generated: Full systematic comparison of `_archive/v1/spectralstream/compression/methods/` against main codebase.

---

## 1. FILES 100% COVERED (No Migration Needed)

### Transform Files (14 small files) → `methods/spectral/`
All archive transform classes have equivalent or better main equivalents:

| Archive File | Archive Class | Main Equivalent(s) | Status |
|---|---|---|---|
| `butterfly_transform.py` | `ButterflyTransform` | `ButterflySparse` + `Butterfly` (decomp) | ✅ Covered |
| `chebyshev.py` | `ChebyshevTransform` | `Chebyshev` (in `transforms.py`) | ✅ Covered |
| `dct.py` | `DCTTransform` | `DCTQuant`, `DCT2D`, `DCTBlock` (7 classes) | ✅ Main is richer |
| `dct2d.py` | `DCT2DTransform` | `DCT2D`, `DCT2DBlock` | ✅ Covered |
| `fourier.py` | `FourierTransform` | `Fourier`, `FrequencyDomain` | ✅ Covered |
| `givens.py` | `GivensTransform` | `Givens` (in `transforms.py`) | ✅ Covered |
| `hadamard.py` | `HadamardTransform` | `FWHT`, `FWHTQuant`, `RandomizedHadamard` | ✅ Covered |
| `householder.py` | `HouseholderTransform` | (function in `_helpers.py`) | ✅ Covered |
| `ntt.py` | `NTTTransform` | `NTTTransform` (in `transforms.py`) | ✅ Covered |
| `random_rotation.py` | `RandomRotation` | `RandomRotationQuant` | ✅ Covered |
| `wavelet_daubechies.py` | `WaveletDaubechies` | `WaveletDaubechies` (in `wavelet.py`) | ✅ Covered |
| `wavelet_haar.py` | `WaveletHaar` | `WaveletHaar` (in `wavelet.py`) | ✅ Covered |
| `wavelet_symlet.py` | `WaveletSymlet` | `WaveletSymlet` (in `wavelet.py`) | ✅ Covered |
| `winograd.py` | `WinogradTransform` | `Winograd`, `WinogradTransform` | ✅ Covered |

### Decomposition Files (15 small files) → `methods/decomposition/`

| Archive File | Archive Class | Main Equivalent(s) | Status |
|---|---|---|---|
| `block_diagonal.py` | `BlockDiagonalDecompose` | `BlockDiagonal` (in `structured_mat.py`) | ✅ Covered |
| `butterfly.py` | `ButterflyDecompose` | `Butterfly` (in `butterfly.py`) | ✅ Covered |
| `circulant.py` | `CirculantDecompose` | (in `structured_decomposition.py`) | ✅ Covered |
| `cp_decompose.py` | `CPDecompose` | `CPDecomposition` (in `cp.py`) | ✅ Covered |
| `CUR_decomposition.py` | `CURDecompose` | `CURDecomposition` (in `kronecker.py`) | ✅ Covered |
| `hankel.py` | `HankelDecompose` | `Hankel` (in `structured_mat.py`) | ✅ Covered |
| `kronecker.py` | `KroneckerDecompose` | `Kronecker` (in `kronecker.py`) | ✅ Covered |
| `low_rank_residual.py` | `LowRankResidual` | `LowRankStructured` (structural/) | ✅ Covered |
| `nystrom.py` | `NystromDecompose` | `Nystrom` (in `matrix_approx.py`) | ✅ Covered |
| `random_feature.py` | `RandomFeatureDecompose` | `RandomFeature` (in `matrix_approx.py`) | ✅ Covered |
| `svd_lowrank.py` | `SVDLowRankD` | `SVDTruncated` (in `svd.py`) | ✅ Covered |
| `toeplitz.py` | `ToeplitzDecompose` | `Toeplitz` (in `structured_mat.py`) | ✅ Covered |
| `tr_decompose.py` | `TRDecompose` | `TensorRing` (in `tensor_train.py`) | ✅ Covered |
| `tt_decompose.py` | `TTDecompose` | `TensorTrain` (in `tensor_train.py`) | ✅ Covered |
| `tucker.py` | `TuckerDecompose` | `TuckerDecomposition` (in `tucker.py`) | ✅ Covered |

### Entropy Files (basic codecs) → `methods/entropy/`

| Archive File | Archive Class | Main Equivalent(s) | Status |
|---|---|---|---|
| `arithmetic.py` | `ArithmeticCoding` | `AdaptiveArithmeticCoder`, `RangeCoder` | ✅ Covered |
| `bwt_mtf.py` | `BWTMTF` | `bwt_mtf_rle_encode/decode` functions | ✅ Covered |
| `huffman.py` | `HuffmanCoding` | `HuffmanCoder` | ✅ Covered |
| `lz77.py` | `LZ77Coding` | `lz77_encode/decode` functions | ✅ Covered |
| `rans.py` | `RANSCoding` | `RANSEncoder`, `RANSDecoder` | ✅ Covered |
| `tans.py` | `TANSCoding` | `TANSEncoder` (in `ans_table.py`) | ✅ Covered |
| `deflate.py` | `DeflateCoding` | `Deflate` (in `_class_wrappers.py`) | ✅ Covered |
| `zstd_like.py` | `ZstdLike` | `ZstdCodec` (in `lossless/zstd_codec.py`) | ✅ Covered |

### Hybrid Files (basic) → `methods/hybrid/`
| Archive Class | Main Equivalent | Status |
|---|---|---|
| `CascadedCompression` | `Cascade2Stage`, `Cascade3Stage`, `Cascade4Stage` | ✅ Covered |
| `EnsembleCompression` | `AllMethodsEnsemble` + `ensemble_compress` | ✅ Covered |
| `ErrorFeedback` | `ErrorFeedbackQuant` | ✅ Covered |
| `MultiStage` | cascade functions (2/3/4-stage) | ✅ Covered |
| `ProgressiveCompression` | `progressive_compress` function | ✅ Covered |
| `AdaptivePipeline` | `adaptive_cascade` function | ✅ Covered |
| `SparseQuantize` | `QuantizeThenSparsify` | ✅ Covered |
| `SVDQuantize` | `DecomposeThenQuantize` | ✅ Covered |

### Large Archive Engine Files → `engine/` package
| Archive File | Lines | Main Equivalent(s) | Status |
|---|---|---|---|
| `compression/intelligence.py` | 341 | `_orchestrator.py`, `_profiler.py` | ✅ Absorbed |
| `compression/compression_profiler.py` | 529 | `_profiler.py` | ✅ Absorbed |
| `compression/dynamic_tensor_intelligence.py` | 714 | `_unified_intelligence.py`, `dynamic_method_tester.py` | ✅ Absorbed |
| `compression/intelligence_real.py` | 829 | `_unified_intelligence.py` | ✅ Absorbed |
| `compression/compression_intelligence.py` | 859 | `_orchestrator.py` (Lagrangian RD) | ✅ Absorbed |
| `compression/intelligence_engine.py` | 1131 | `_orchestrator.py`, `_profiler.py` | ✅ Absorbed |
| `compression/unified_quant_system.py` | 1240 | `_unified_intelligence.py` | ✅ Absorbed |
| `compression/quantization_engine.py` | 1009 | `_methods.py` + `methods/quantization/` | ✅ Absorbed |
| `compression/optimal_quantizer.py` | 1269 | `_methods.py` + `methods/quantization/` | ✅ Absorbed |

### 7 Top-Level Stubs
Files at `_archive/v1/spectralstream/{compression_intelligence,compression_profiler,...}.py` — all 11-line deprecation stubs. Already re-export. ✅

---

## 2. FILES WITH CLASSES UNDER DIFFERENT NAMES (Verify Equivalence)

### Physics → Cutting Edge (BETTER in cutting edge)

| Archive | Main/Cutting Edge | Completeness |
|---|---|---|
| `AlgebraicGeomCompression` | `AlgebraicGeometryCompression` (cutting_edge/) | 🟢 Cutting edge is FAR superior (actual algebraic variety compression vs. archive's polynomial curve fitting) |
| `CategoryTheoryCompression` | `CategoryTheoryCompression` (cutting_edge/) | 🟢 Cutting edge is true category theory (objects, morphisms, generators); archive was SVD-renamed |
| `DebyeShieldingCompression` | `DebyeShieldingCompression` (cutting_edge/) | 🟢 Cutting edge has actual Debye kernel convolution + SVD fallback |
| `DensityMatrixCompression` | `DensityMatrixCompression` (cutting_edge/) | 🟢 Cutting edge is superior |
| `OptimalTransportCompression` | `OptimalTransportCompression` (cutting_edge/) + `OptimalTransport` (physics_misc.py) | 🟢 Cutting edge is superior |
| `QuantumStateCompression` | `QuantumStateCompression` (cutting_edge/) | 🟢 Cutting edge is superior |
| `TopologicalDataCompression` | `TopologicalDataCompression` (cutting_edge/) | 🟢 Cutting edge implements actual persistent homology; main `TopologicalData` (topology.py) is a **broken delegate** to DensityMatrix |
| `VlasovCompression` | `VlasovDistributionCompression` (cutting_edge/) | 🟢 Cutting edge implements actual Vlasov characteristics; main `VlasovDistribution` (vlasov.py) is just DCT-based, not physics |

### Physics → Main (different name, verify equivalence)

| Archive | Main | Completeness |
|---|---|---|
| `MHDCompression` | `MHDCompression` (physics/mhd.py) | 🟢 Same name, equivalent functionality |
| `PlasmaOscillationCompression` | `PlasmaOscillation` (physics/plasma.py) | 🟢 Equivalent, main has more features |
| `PlasmaTurbulenceCompression` | `PlasmaTurbulence` (physics/plasma.py) | 🟢 Equivalent, main has more features |
| `QuantumEntangleCompression` | `QuantumEntanglement` (physics/quantum.py) | 🟢 Equivalent, main has more features |
| `QuantumTunnelCompression` | `QuantumTunneling` (physics/quantum.py) | 🟢 Equivalent, main has more features |
| `ResonanceModesCompression` | `ResonanceModes` (physics/resonance.py) | 🟢 Equivalent, main has more features |
| `ErrorCorrectionCompression` | `QuantumErrorCorrection` (physics/quantum.py) | 🟢 Equivalent |
| `HamiltonianCompression` | `HamiltonianDynamical` (physics/physics_misc.py) | 🟢 Equivalent |
| `LagrangianCompression` | `Lagrangian` (methods/functional/lagrangian.py) | 🟢 Equivalent |
| `DensityMatrixCompression` | `DensityMatrix` (physics/quantum.py) | 🟢 Equivalent, main has more features |

### key WARNING: `topology.py` (TopologicalData) is WEAKER than archive
Main `physics/topology.py`'s `TopologicalData` and `TopologicalFunctional` classes **delegate to `DensityMatrix`** — they have no independent topological algorithm and lack `estimate_ratio`/`estimate_error`. The archive at least has a statistical feature extraction algorithm. The cutting_edge version is the only correct implementation.

---

## 3. TRULY MISSING CLASSES/FUNCTIONS — Need Migration

### Category A: Novel Inference Operators (CRITICAL — broken imports)
These 5 classes are **imported by `spectralstream/orchestrator.py`** but the file `spectralstream/compression/novel_operators.py` **does not exist**. The import silently fails and all operators are `None`.

| Class | Lines | Description | Location in main? |
|---|---|---|---|
| `HDCWeightedTokenSampling` | ~140 | Fuse HDC similarity scores with model logits | ❌ Absent |
| `SpectralEntropyGating` | ~160 | Gate transformer layers by DCT spectral entropy | ❌ Absent |
| `AdaptiveForwardlessDepth` | ~120 | Dynamic forwardless depth selection | ❌ Absent |
| `GradientFreeFineTuning` | ~180 | Forward-only LoRA fine-tuning (no backward pass) | ❌ Absent |
| `PredictorCorrectorInference` | ~140 | ODE-style predictor-corrector with adaptive step | ❌ Absent |

**Action needed:** Create `spectralstream/compression/novel_operators.py` from `_archive/v1/spectralstream/compression/novel_operators.py` (778 lines). These are pure numpy, no dependencies beyond stdlib + numpy.

### Category B: Manifold Learning Classes (completely absent)
| Class | File | Description |
|---|---|---|
| `ManifoldIsomapCompression` | `methods/physics/manifold_isomap.py` | ISOMAP-based manifold compression |
| `ManifoldLLECompression` | `methods/physics/manifold_lle.py` | LLE-based manifold compression |

Neither class exists anywhere in the main codebase. (Note: `ManifoldLearning` exists in `physics_misc.py` but is a different class.)

### Category C: novel_physics.py — 19 Missing Classes
| Class | Description |
|---|---|
| `FreeEnergyCompression` | Helmholtz free energy decomposition |
| `LagrangianCompression` | Lagrangian mechanics tensor factorization |
| `HamiltonianFlowCompression` | Hamiltonian dynamics phase-space flow |
| `GaugeFieldCompression` | Gauge field theory compression |
| `RenormalizationGroupCompression` | RG flow-based scale decomposition |
| `EntropicForceCompression` | Entropic force compression |
| `MaximumEntropyModelCompression` | MaxEnt model compression |
| `KolmogorovOptimalCompression` | Kolmogorov complexity optimal coding |
| `HolographicEncodingCompression` | Holographic principle encoding |
| `TopologicalCompression` | Topological data analysis compression |
| `CategoryTheoreticCompression` | Category-theoretic morphism compression |
| `OptimalTransportMapCompression` | Optimal transport map compression |
| `ManifoldAlignmentCompression` | Manifold alignment compression |
| `SymplecticCompression` | Symplectic geometry compression |
| `FisherRaoGeodesicCompression` | Fisher-Rao information geometry |
| `QuantumStateCompressionCompression` | Quantum state compression |
| `PlasmaFieldDecompositionCompression` | Plasma field mode decomposition |
| `HolographicReducedRepresentationCompression` | Holographic reduced representation |
| `TimeCrystalPhaseCompression` | Time crystal Floquet compression |

(`NeuralODECompression` is the only one already migrated to cutting_edge/. The remaining 19 are absent.)

### Category D: Hybrid — 2 Missing Combined Methods
| Archive Class | Description | Status |
|---|---|---|
| `HadamardQuantEntropy` | Hadamard + quantization + entropy combined | ❌ No combined method in main hybrid |
| `DeltaQuantize` | Delta/sequential quantization | ❌ Not present in main hybrid |

### Category E: Large Archive Transform Modules (3 files)
| File | Lines | Description | Status |
|---|---|---|---|
| `nn_weight_transforms.py` | 1937 | 20 NN-specific transform techniques with abstract base class | ❌ Not migrated (main spectral/ has generic transforms only) |
| `polynomial_approx.py` | 1661 | 20 polynomial/functional approximation methods | ❌ Not migrated (main has single `PolynomialApprox` class) |
| `working_transforms.py` | 1140 | Transform methods tested on real Gemma-4 weights (FWHT, DCT, etc.) | ❌ Not migrated (uses own FWHT primitives) |

### Category F: advanced_factorization.py — 20 Advanced Decomposition Methods
| Class | Description |
|---|---|
| `SVDResidualDecompose` | SVD with residual compression |
| `CURFullDecompose` | Full CUR decomposition |
| `NystromAdvancedDecompose` | Advanced Nyström approximation |
| `RandomFeatureAdvancedDecompose` | Advanced random feature |
| `BlockSVDDecompose` | Block-wise SVD |
| `TiledLowRankDecompose` | Tiled low-rank approximation |
| `ProgressiveSVDDecompose` | Progressive SVD |
| `IncrementalSVDDecompose` | Incremental SVD |
| `SparseSVDDecompose` | Sparse SVD |
| `OrthogonalProcrustesDecompose` | Orthogonal Procrustes |
| `NonNegativeMatrixFactorize` | NMF |
| `ProbabilisticMatrixFactorize` | Probabilistic matrix factorization |
| `BayesianMatrixFactorize` | Bayesian matrix factorization |
| `TensorTrainAdvancedDecompose` | Advanced tensor train |
| `TensorRingAdvancedDecompose` | Tensor ring decomposition |
| `CPAdvancedDecompose` | Advanced CP decomposition |
| `TuckerAdvancedDecompose` | Advanced Tucker decomposition |
| `HierarchicalTuckerDecompose` | Hierarchical Tucker |
| `BlockDiagonalPlusLowRankDecompose` | Block-diagonal + low-rank |
| `SkeletonDecompose` | Skeleton/CUR decomposition |

**Note:** The main breakthrough_decomposition_massive/ (151 files) may cover some of these, but the named classes above don't directly exist.

### Category G: Archive Top-Level Method Classes (standalone files)
| Archive Class | Description | Status |
|---|---|---|
| `BF16ExploitCompression` | BF16 exponent exploitation | ❌ Absent |
| `DCTSpectralCompression` | DCT spectral compression (with Config class) | ❌ Absent (engine has `_DCTSpectral`, different API) |
| `HadamardTransformCompression` | Hadamard transform compression | ❌ Absent (engine has `_HadamardINT8`/`_HadamardINT4`, different API) |
| `KroneckerApprox` | Kronecker product approximation | ❌ Absent |
| `ManifoldEmbedding` | Manifold embedding compression | ❌ Absent |
| `WaveletThresholdCompression` | Wavelet threshold compression | ❌ Absent |
| `ResonanceDecomposition` | Resonance mode decomposition | ❌ Absent |
| `TRDecomposition` | Tensor ring decomposition | ❌ Absent |
| `TTDecomposition` | Tensor train decomposition | ❌ Absent (engine has `_TensorTrain`, different API) |
| `SparseGPTPruning` | SparseGPT pruning (with Config) | ❌ Absent (methods/sparsity/sparsegpt.py exists but different) |
| `GroupLassoPruning` | Group lasso pruning (with Config) | ❌ Absent (methods/structural/ has GroupLasso, different) |
| `MixedPrecisionAllocation` | Mixed precision allocation (with Config) | ❌ Absent (methods/quantization/mixed_precision.py exists, different) |
| `NF4Quantization` | NF4 quantization (with Config) | ❌ Absent (methods/quantization/nf4.py exists with `NF4`, different) |
| `E8LatticeQuantization` | E8 lattice quantization (with Config) | ❌ Absent (methods/quantization/lattice.py exists with `E8Lattice`, different) |
| `QuantumStateEncoding` | Quantum state encoding (with Config) | ❌ Absent (different from main `QuantumState`) |

### Category H: predictive_coding.py — 20 Entropy Classes
| Class | Status |
|---|---|
| `DeltaRowCoding` | ❌ (coarse `PredictiveCoding` wrapper only) |
| `DeltaColumnCoding` | ❌ |
| `Delta2DCoding` | ❌ |
| `ARPredictCoding` | ❌ |
| `AR2PredictCoding` | ❌ |
| `ARPredictRowCoding` | ❌ |
| `ARPredictColumnCoding` | ❌ |
| `ContextModelCoding` | ❌ |
| `DictionaryCoding` | ❌ |
| `RunLengthCoding` | ❌ |
| `LZ77WeightCoding` | ❌ |
| `BurrowsWheelerCoding` | ❌ |
| `MoveToFrontCoding` | ❌ |
| `HuffmanWeightCoding` | ❌ |
| `RANSWeightCoding` | ❌ |
| `ArithmeticWeightCoding` | ❌ |
| `PredictionErrorCoding` | ❌ |
| `MarkovModelCoding` | ❌ |
| `GaussianMixtureCoding` | ❌ |
| `AdaptiveEntropyCoding` | ❌ |

### Category I: working.py — 12 Quantization Method Functions
These 12 methods (and 10 helper functions) were the **battle-tested reference implementations**. The main engine has class-based equivalents (`_HadamardINT8`, `_BlockINT8`, etc.) but:
- The exact function implementations don't exist
- Key private helpers like `_asym_quantize_int8/4/2`, `_fwht_inplace`, `_estimate_entropy_bits` do NOT exist in main

| Function | Status |
|---|---|
| `_asym_quantize_int8` / `_asym_dequantize_int8` | ❌ Not in main |
| `_asym_quantize_int4` / `_asym_dequantize_int4` | ❌ Not in main |
| `_asym_quantize_int2` / `_asym_dequantize_int2` | ❌ Not in main |
| `_fwht_inplace` | ❌ Not in main (main uses `core.math_primitives.fwht`) |
| `_estimate_entropy_bits` | ❌ Not in main |
| `method_hadamard_int8` (and 11 more) | ❌ Not as functions (exist as classes in engine) |

---

## 4. SUPPLEMENTARY: Cutting-Edge vs Main Physics Quality Assessment

Detailed comparison of 5 paired implementations (archive → cutting_edge vs archive → main physics):

| Class | Archive Quality | Main Physics Quality | Cutting Edge Quality |
|---|---|---|---|
| `TopologicalData`/`TopologicalDataCompression` | Basic (stats + SVD) | **BROKEN** (delegates to DensityMatrix) | ✅ Genuine persistent homology |
| `VlasovCompression`/`VlasovDistributionCompression` | Basic (histogram) | **MISNAMED** (just DCT compression) | ✅ Vlasov characteristics |
| `AlgebraicGeomCompression`/`AlgebraicGeometryCompression` | Fake (SVD + poly fit) | N/A | ✅ Algebraic variety compression |
| `CategoryTheoryCompression` | Fake (SVD renamed) | N/A | ✅ True category theory |
| `DebyeShieldingCompression` | Simple (FFT screening) | N/A | ✅ Debye kernel convolution |

**Recommendation:** The cutting_edge versions are the canonical implementations and should be preferred. Main physics `topology.py` and `vlasov.py` should be audited for correctness — they appear to be placeholder implementations.

---

## 5. SUMMARY COUNTS

| Category | Count | Notes |
|---|---|---|
| ✅ Fully covered files | ~85 | Transforms, decomposition, entropy, hybrid, engine files |
| 🟢 Different names, verified equivalent | ~19 | Physics → cutting edge (superior) or physics → main (equivalent) |
| ❌ Truly missing classes needing migration | ~85+ | See categories A-I above |
| | | |
| **Category A: Novel inference operators** | **5** | CRITICAL — broken imports in orchestrator.py |
| **Category B: Manifold learning** | **2** | Completely absent |
| **Category C: novel_physics.py** | **19** | 1/20 migrated (NeuralODECompression) |
| **Category D: Hybrid methods** | **2** | HadamardQuantEntropy, DeltaQuantize |
| **Category E: Large transform files** | **3** | 60 combined methods (nn_weight, polynomial, working transforms) |
| **Category F: Advanced factorization** | **20** | advanced_factorization.py |
| **Category G: Top-level method classes** | **15** | Config + operation pairs |
| **Category H: Predictive coding** | **20** | Detailed entropy classes |
| **Category I: working.py quantization** | **12** | Functions (engine has class equivalents) |
