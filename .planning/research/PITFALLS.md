# Pitfalls Research

**Domain:** LLM weight / KV-cache compression and local inference (post-training quantization, decomposition, entropy coding)
**Researched:** 2026-07-08
**Confidence:** HIGH (domain theory grounded in 2024–2026 entropy/rate-distortion literature; project-specific claims grounded in `.planning/codebase/` audit + committed result JSONs)

---

## How Spectral-Stream Is Exposed (one-paragraph framing)

This project sits on a pure-NumPy, CPU-only, 2,964-method registry with a 5-stage cascade that only wires 2 stages. That exact combination makes it simultaneously vulnerable to every pitfall below: the cascade *multiplies per-stage ratio estimates and ignores the error budget* (BUG-02), the registry *counts stubs as capability* (TD-01), the SSF container *mistakes 4096-byte alignment padding for compression* (TD-03/PERF-01), the headline number *compares against FP32 to double itself* (BUG-03), and the whole quality story *rests on reconstruction MSE while 71% of tensors are pass-through and functional error is 100× worse* (TD-05, MISS-01). The v1 priority — honesty-first — is the correct antidote, but only if each pitfall is explicitly gated, not just documented.

---

## Critical Pitfalls

### Pitfall 1: The Shannon-Limit Trap — claiming ratios beyond the weight entropy floor

**What goes wrong:**
Numbers are advertised that physically cannot be achieved at the claimed quality. INT8 blockwise quantization reporting `ratio_vs_bf16 = 22.31` (i.e. ~0.7 bits/weight) is one instance; the "research catalog" floats 2000–5000× ratios (quantum/holographic/chaotic) presented anywhere near current capability is another.

**Why it happens:**
Compression is treated as a marketing dial, not a rate-distortion problem. Developers forget that there is a hard floor: at a given reconstruction distortion D, the bits/weight cannot drop below the rate-distortion function R(D). Real measured LLM weight entropy at moderate distortion lands around **4–5.7 bits/weight**; the *lossless* floor is ~11 bits/weight (DFloat11 shows BF16 exponents carry only ~2.6 bits of information out of 8 allocated, so ~30% lossless reduction is the ceiling). INT4 (4 bits/weight) is near the practical lossy floor for weight-only at acceptable quality. Anything claiming 0.7 bits/weight at low error is either lossy beyond usefulness or simply not measuring bytes.

**How to avoid:**
- Route every ratio through `honest_metrics.serialized_nbytes()` — never a derived constant. A 22.31× INT8 claim is a test failure, not a press release.
- Publish the bits/weight implied by any ratio and sanity-check it against R(D): INT8 ≈ 8 b/w (+scale overhead), INT4 ≈ 4 b/w. If bits/weight < floor, the measurement is broken.
- Keep exotic 2000–5000× numbers strictly inside the `experimental/` namespace, labeled "theoretical upper bound, not measured."

**Warning signs:**
- `ratio_vs_bf16` identical across wildly different methods/tensors (wave4_results.json: cascade and block_int8 both = 22.31).
- A "compression ratio" larger than 1/bits-floor (e.g. >2× vs BF16 for INT8 is already suspicious; >4× vs BF16 for INT8 is impossible).
- Ratios quoted without the paired error on the same line.

**Phase to address:** METRICS-01 (error-gate all ratios), METRICS-02 (lead with ratio_vs_disk), DOCS-01 (frame exotic numbers as targets only).

---

### Pitfall 2: Eval Methodology — calibration-set perplexity, overfitting, no downstream accuracy

**What goes wrong:**
Quality is asserted from tensor reconstruction on the same data used to calibrate, or from perplexity measured on the calibration distribution — while zero held-out downstream-task accuracy is reported. The model can look "preserved" on the calibration set and silently degrade on real tasks.

**Why it happens:**
Reconstruction error `‖W−Ŵ‖` is cheap to compute and gets mistaken for task preservation. Literature is explicit that minimizing calibration-set reconstruction does **not** control `‖W−Ŵ‖` deviation and is increasingly unreliable at W2/W3 (SARQC, 2026). Perplexity correlates only weakly with task accuracy, and calibration-set choice alone shifts downstream accuracy by up to **70%** (Generalization Ability of Quantized LLMs, 2024). Reporting only reconstruction/perplexity-on-calibration is the field-standard way to overstate.

**How to avoid:**
- Implement at least one real downstream eval: Wikitext/C4 perplexity of original vs compressed model end-to-end (EVAL-01), plus a zero-shot task or two via lm-eval-harness.
- Always report perplexity on a **held-out** distribution distinct from the calibration set.
- Never lead with reconstruction `rel_mse` when functional/downstream error exists (see Pitfall 9).

**Warning signs:**
- "Quality verified" backed only by per-tensor `rel_mse`/`cosine_sim`, no perplexity JSON artifact in the repo (MISS-01 — none exists).
- Calibration and evaluation use the same 128–256 samples.
- `rel_mse_median = 0.0` reported as a headline without disclosing 71% of tensors are pass-through (TD-05).

**Phase to address:** EVAL-01 (real perplexity), METRICS-04 (test the metric path), DOCS-01.

---

### Pitfall 3: Cascade of Estimate Errors — multiplying per-stage ratios instead of measuring end-to-end bytes

**What goes wrong:**
A multi-stage pipeline reports a ratio computed as the *product of per-stage estimated ratios*, so each stage's optimistic estimate compounds. A method that destroys 85% of the signal still reports the same 22.31× as the working method (BUG-02). This is the single most damaging metric bug in the repo's history.

**Why it happens:**
Per-stage "estimated ratio" is easier to compute than serializing the full payload. The cascade's `stage_diagnosis.json` even cites `ratio_vs_fp32 = 26.06` while `rel_mse = 0.86` — the number and the error are disconnected. The honest-metrics mandate exists (`honest_metrics.py`) but is not enforced by tests (COV-03), so the old `len(dict)`/product behavior leaks back.

**How to avoid:**
- Enforce: ratio = `serialized_nbytes(payload) / original_bytes`, measured once on the final serialized container, never as a product of stage estimates.
- Hard gate: any method with `rel_mse > threshold` must not emit a ratio (METRICS-01). The threshold belongs in config, not in prose.
- Add `tests/test_honest_metrics.py` asserting a high-error payload is flagged/rejected (COV-03 → METRICS-04).

**Warning signs:**
- `ratio_vs_bf16` equal across methods that produce different errors.
- `ratio` computed from `len(dict)` / `len(stages)` / a formula rather than `serialized_nbytes`.
- `honest_metrics.py` has no dedicated test despite being the anti-fabrication safeguard.

**Phase to address:** METRICS-01, METRICS-04 (COV-03), CASCADE-01 (reject cascade if rel_mse > 0.05).

---

### Pitfall 4: GPU-vs-CPU Fallacy — claiming CPU parity while depending on GPU-optimized kernels

**What goes wrong:**
An inference/compression engine is positioned as competitive while its dominant workload (GEMM/decomposition) runs on pure NumPy with no GPU and no vendor BLAS-optimized kernel path. Throughput is orders of magnitude below llama.cpp/ExLlama/vLLM.

**Why it happens:**
The "engine" framing is inherited from GPU tools. Real fast methods need hardware-matched kernels: AQLM's fast path is Triton/CUDA (CPU Numba path is 3–4× slower and marked ❌ for fast GPU); QuIP#'s speed comes from CUDA kernels (3-bit inference is "significantly slower" because no optimized matvec kernel exists). Pure-NumPy gives you none of that. Spectral-Stream's own `PERF-04`/`DEP-01` confirm torch is optional and there is no mandatory accelerated linear-algebra backend.

**How to avoid:**
- Scope the project explicitly as CPU-NumPy research-grade (already in PROJECT.md constraints) — and make that the *default message*, not a footnote.
- Never benchmark "inference speed vs llama.cpp" without disclosing 2–3× slower CPU-only reality. If a speed claim is made, it must be measured on the same hardware class.
- If production viability is ever wanted, commit to an optional BLAS/GPU backend rather than implying the NumPy path is fast.

**Warning signs:**
- README/docs call it an "inference engine" while `pyproject.toml` has no mandatory accelerated dependency.
- Timing claims (`time_ms`) that include model-load overhead presented as pure compute time (PERF-03: block_int8 1–3s/matrix).
- "CPU speed parity" language anywhere near GPU-baseline comparisons.

**Phase to address:** DOCS-01 (honest scoping), DEP-01 resolution (single-source deps, backend decision).

---

### Pitfall 5: Paper-Chasing — implementing the newest method without the hardware to run it

**What goes wrong:**
Exotic SOTA methods (QuIP#, AQLM, HIGGS, ergodic/SIREN cascades) are registered/ported into a pure-NumPy, CPU-only, GPU-less codebase where they cannot execute at the fidelity the papers require, becoming stubs that masquerade as capability.

**Why it happens:**
Novelty reads as progress. But AQLM needs LoRA fine-tuning + CUDA kernels; QuIP# needs incoherence processing + CUDA matvec; SIREN needs 200 epochs of gradient descent that `stage_diagnosis.json` shows captures SNR = 0.00 dB on a near-noise residual. Porting the *math* without the *hardware/kernels/training* yields methods that cannot run or cannot help.

**How to avoid:**
- Gate new methods behind a "runs on real weights in-repo" gate before they enter the active registry.
- Match method to hardware: if you have no GPU, do not register CUDA-kernel-dependent methods as working.
- Keep frontier-method explorations in `experimental/` with an explicit "requires <GPU/fine-tuning/kernel>" label.

**Warning signs:**
- A method module imports `torch`/`scikit-learn` that are only optional extras and raises `ImportError` mid-run (DEP-03).
- `NotImplementedError` in `decompress` paths for "flagship" methods (BUG-06: `compression_intelligence_v2.decompress`).
- A method whose own diagnosis says it adds 0.00 dB (SIREN, ergodic) still listed as a live cascade stage.

**Phase to address:** REGISTRY-01 (validated set vs experimental/), REGISTRY-02 (kill broken auto-discovery), CASCADE-02.

---

### Pitfall 6: The Registry of Aspirational Methods — hundreds of stubs inflate perceived capability

**What goes wrong:**
2,964 methods registered, but the honest full-model run used exactly **2** (`fp16_passthrough` 71% + `int8_blockwise+zlib` 29%). The registry reads as a 2,964-method toolkit; it is a 2-method toolkit with a 2,962-entry catalog (TD-01).

**Why it happens:**
Breadth is mistaken for capability. Large registries look impressive in a README and in `list-methods` output, and the cost (slow startup, maintenance, contributor confusion) is invisible until audited. The danger is not the catalog per se — it is presenting it as working.

**How to avoid:**
- Split into a **validated active set** (methods with passing tests + real-weight results) vs a clearly-labeled `experimental/` namespace (REGISTRY-01).
- Add a test asserting registry size stays bounded and that every active method has a real-weight result artifact.
- Make `list-methods` show maturity level per method; never count the catalog as capability.

**Warning signs:**
- `Discovered and registered 2964 methods` in logs while run artifacts show 2 methods exercised.
- Method count quoted as a strength in docs/README.
- Startup dominated by registration of never-used classes (SCALE-01: ~2s just to register).

**Phase to address:** REGISTRY-01, REGISTRY-02, DOCS-02 (Research Catalog documentation).

---

### Pitfall 7: Calibration Pitfalls — data contamination, tokenizer mismatch, sequence-length effects

**What goes wrong:**
Calibration/eval is contaminated (hardcoded author path), uses the wrong tokenizer, or uses sequence lengths that don't match deployment — producing numbers that don't transfer to real use.

**Why it happens:**
Calibration is fiddly and easy to mock. Spectral-Stream's `benchmark_physics_real_weights.py` hardcodes `/home/mike/Documents/.../gemma-4-E2B/model.safetensors` (SEC-02 — not reproducible off the author's machine). `BaseTokenizer.encode` raises `NotImplementedError` and `token_count` calls it (BUG-04) — so encoding can crash unless a concrete tokenizer with its external lib is present. Literature shows calibration-set choice alone swings task accuracy up to 70%, and sequence length materially affects GPTQ activation quantization.

**How to avoid:**
- Read model/calibration paths from env/CLI args; publish the model hash used (SEC-02 fix).
- Ship one self-contained, tested default tokenizer (MISS-02); make `BaseTokenizer` an ABC so misuse fails at import, not at runtime.
- Document the calibration dataset source, size, and sequence length used for every reported number; use a held-out eval set (Pitfall 2).

**Warning signs:**
- Absolute paths to the author's machine in benchmark scripts.
- `token_count`/`encode` crashing on the base class.
- No statement of calibration data source/size/seq-len alongside a quality claim.

**Phase to address:** SEC-02 (reproducible paths), BUG-04 (tokenizer ABC), MISS-02 (default tokenizer), EVAL-01 (documented calibration).

---

### Pitfall 8: Comparison Fabrication — hardcoded competitor numbers instead of running them

**What goes wrong:**
Competitor ratios/errors are looked-up constants, not measurements. `benchmark_industry_comparison.json` reports GPTQ/AWQ/GGUF with **identical values across different matrices** (proving they're estimated, not measured), and `certificate.py` builds an `industry_comparison` from hardcoded tuples like `("GGML Q8_0", 2.5, ...)`. This is the repo's historical "fabricated metrics" problem recurring (SEC-03).

**Why it happens:**
Running GPTQ/AWQ requires dependencies the project doesn't have (`grep` finds no gptq/awq import; only a local DCT approximation `_awqlike.py`). Rather than admit "we can't measure this," constants are substituted. Identical per-matrix floats are the tell.

**How to avoid:**
- Only compare against methods actually executed in-repo.
- Any external/literature number must be explicitly labeled "literature estimate, not measured here" — never in the same visual style as measured results.
- Remove hardcoded ratio tables; add a lint that fails CI if a competitor number isn't produced by a runner in the repo (MISS-04).

**Warning signs:**
- Same ratio/error float repeated across embedding/output/attention matrices.
- Competitor value equals a clean formula (GGUF = 4/1.125 = 3.5555…).
- `certificate.py` contains a dict of `("name", ratio, ...)` literals.

**Phase to address:** METRICS-03 (replace fabricated comparison), MISS-04 (CI metrics-honesty lint).

---

### Pitfall 9: Reconstruction Error ≠ Functional / Downstream Error (the pass-through illusion)

**What goes wrong:**
Headline quality uses tensor reconstruction `rel_mse`, but the metric that matters for generation — functional error `‖Wx − Ŵx‖ / ‖Wx‖` — is ~100×–560× worse and is buried (TD-05). The reconstruction number is dominated by 1,427 zero-error pass-through tensors, so it looks excellent while real inference error on attention projections reaches 1.4%.

**Why it happens:**
`rel_mse` is cheap and naturally near-zero for the 71% of tensors stored losslessly/pass-through. Averaging over a pass-through-dominated model hides the error that actually changes outputs. This is the same root cause as Pitfall 2 (reconstruction ≠ task), made worse by counting passthrough as compression.

**How to avoid:**
- Lead with functional/downstream error, not reconstruction `rel_mse` of a pass-through-heavy model.
- Report the functional-error distribution prominently; never report `rel_mse_median = 0.0` without the pass-through caveat.
- Treat pass-through tensors as "stored," not "compressed" (Pitfall 11).

**Warning signs:**
- `rel_mse_median = 0.0` alongside a non-zero `functional rel_err` up to 1.4%.
- Quality story rests entirely on `cosine_sim = 0.999988` (dominated by zeros).
- No functional-error column in the certificate by default.

**Phase to address:** METRICS-01 (error-gate + surface functional error), TD-05 fix, EVAL-01.

---

### Pitfall 10: Format Overhead Masquerading as Compression (SSF 4096-byte alignment)

**What goes wrong:**
The SSF container aligns every tensor to 4096-byte pages with 256-byte header + 128-byte footer. With 2,011 tensors (71% small pass-through), alignment/padding dominates stored size. The "compressed 4.443 GB" and the headline "2.3× vs BF16" are carried mostly by format padding, not algorithmic savings (TD-03/PERF-01).

**Why it happens:**
Page-alignment simplifies I/O but is catastrophic for the thousands of small tensors typical in LLM weight maps. The honest-metrics layer measures true bytes (good) but the *design* still inflates them, and the headline ratio is reported without separating overhead from algorithmic ratio.

**How to avoid:**
- Report algorithmic ratio separately from container overhead (FORMAT-01).
- Pack small tensors into shared pages instead of per-tensor 4096-byte alignment (FORMAT-01).
- Stop counting pass-through tensors as "compression" — they should read as "stored losslessly."

**Warning signs:**
- `orig_bf16_bytes / compressed_bytes ≈ 0.077` for pass-through tensors (stored ~13× larger than source).
- Headline ratio with no breakdown of how much is algorithmic vs padding.
- 4096-byte page size applied to tensors far smaller than 4 KB.

**Phase to address:** FORMAT-01 (separate overhead, pack small tensors), DOCS-01 (honest framing).

---

### Pitfall 11: Counting Pass-Through / Lossless as "Compression"

**What goes wrong:**
`fp16_passthrough` and lossless storage are reported in the same "compression" bucket as INT8 quantization. 71% of tensors are pass-through; the aggregate "2.3× compression" is therefore mostly the 29% INT8 tensors doing real work, not a uniform 2.3× across the model.

**Why it happens:**
The compressor selects a method per tensor and the aggregate ratio averages over all tensors, so a few real compressors drag the mean up while most tensors just change container. Presenting the mean as "the model is 2.3× smaller" overstates what compression is actually happening.

**How to avoid:**
- Report per-method and per-category ratios, not just a single model mean.
- Label pass-through/lossless tensors as "stored," with their byte cost shown separately from "compressed."
- Make the default headline the INT8 path's real ratio (~2.3× vs BF16, ~4.6× vs FP32) with the pass-through share disclosed.

**Warning signs:**
- Method distribution shows >50% pass-through yet a single impressive aggregate ratio.
- "Compression ratio" with no per-method breakdown table.

**Phase to address:** FORMAT-01, DOCS-01, METRICS-02.

---

### Pitfall 12: FP32-Baseline Inflation

**What goes wrong:**
The input is already BF16, but the headline ratio is quoted vs FP32, doubling it (4.6× vs the honest 2.3× vs on-disk BF16). BUG-03.

**Why it happens:**
FP32 comparison always looks better and is the number most likely to be quoted. `benchmark_industry_comparison.json` and `final_benchmark_results.json` both compute `ratio_vs_fp32`.

**How to avoid:**
- Make `ratio_vs_disk` (BF16) the default headline; demote `ratio_vs_fp32` to secondary (METRICS-02).
- State the baseline format explicitly next to every ratio.

**Warning signs:**
- Two ratios presented with the larger (FP32) one first or unlabeled.
- "4.6×" quoted without "(vs FP32; vs BF16 it is 2.3×)".

**Phase to address:** METRICS-02, DOCS-01.

---

### Pitfall 13: The Half-Wired Cascade (documented 5-stage, 2 live)

**What goes wrong:**
The flagship 5-stage cascade (EinSort → TT-SVD → Sparse → Ergodic → SIREN) only wires stages 1–2. Stages 3–5 exist as helper functions but are never called in `compress()`. On real Gemma-4 weights it yields **72–92% reconstruction error** (BUG-01), yet is presented as a headline feature (FRAG-04).

**Why it happens:**
The pipeline was described/committed as complete before stages 3–5 were wired. `stage_diagnosis.json` proves it's a *design* flaw, not a tuning issue: TT-SVD folding creates a mode-0 dim of only 32 for a 2048×1536 matrix, capturing ~14% variance; sparse/ergodic/SIREN add ~0 dB. You cannot "tune" a cascade whose first stage throws away 86% of the signal.

**How to avoid:**
- Do not expose a method as default/headline until it beats the INT8 baseline on real weights (CASCADE-01).
- Either wire stages 3–5 honestly (chaining residuals) or update docs to state only EinSort + TT/Quant are live (CASCADE-02).
- Add a test asserting `rel_mse < 0.05` on a real slice; mark cascade `xfail`/disabled until it passes (COV-05).

**Warning signs:**
- `used_stages=[1,2]` in a "5-stage" method.
- Stage-3/4/5 helper functions defined and imported but absent from `compress()`.
- A flagship feature whose own diagnosis says "TT captures only ~14% of variance."

**Phase to address:** CASCADE-01, CASCADE-02, COV-05, DOCS-01.

---

### Pitfall 14: Silent Corruption via dtype overflow (fp16 passthrough NaN)

**What goes wrong:**
`fp16_passthrough` casts BF16→FP16. BF16 exponent range ~3.4e38; FP16 max ≈ 65504. Any out-of-range tensor silently becomes `inf`/`nan` and propagates through generation (FRAG-01). It "works" on Gemma-4 only by coincidence (1427/1429 pass-through tensors happened to be in FP16 range).

**Why it happens:**
The passthrough path assumes all values fit FP16. That holds for the one tested model and breaks on the next model with larger-norm weights/activations. No overflow check, no fallback.

**How to avoid:**
- Detect out-of-FP16-range values; fall back to BF16-passthrough (no cast) or INT8.
- Add a round-trip + overflow-fallback test with large-magnitude inputs (COV-04).

**Warning signs:**
- `rel_mse == 0` for passthrough tensors presented as proof of correctness rather than as "values happened to fit."
- No overflow guard before the BF16→FP16 cast.

**Phase to address:** FRAG-01 fix, COV-04 (round-trip test).

---

### Pitfall 15: Research Prototype vs Real Compression Tool — the credibility gap

**What goes wrong:**
The project reads as a "compression engine" but lacks what separates llama.cpp, ExLlama, and AutoGPTQ from research code: reproducible measured benchmarks on real models, hardware-matched kernels, honest eval via lm-eval-harness + perplexity, a working round-trip from a real format, and the discipline to never claim a ratio it can't measure.

**Why it happens:**
Research code optimizes for novelty and internal demos; production tools optimize for the trust loop (measure → publish → reproduce). Spectral-Stream has the building blocks (honest_metrics, SSF I/O, GGUF/safetensors converters, CLI) but the trust loop is broken by BUG-02/SEC-03/TD-01/MISS-01.

**What real tools do (the bar):**
- **llama.cpp / GGUF:** optimized CPU/GPU kernels, billion-scale adoption, measured perplexity, honest per-quant-type ratios.
- **ExLlama / EXL2:** GPU kernels, measured quality at each bit budget, clear scope.
- **AutoGPTQ:** runs GPTQ end-to-end, publishes perplexity vs baseline, pins its dependency set.

**How to avoid:**
- Treat the v1 honesty work as building the trust loop, not just fixing bugs: every number measured, paired with error, reproducible from a published model hash, never compared to un-run competitors.
- Commit to one credible scope (CPU-NumPy research) and stop implying production serving parity (Pitfall 4).
- Ship a working round-trip on a real model as the definition of "done" for any method.

**Warning signs:**
- No perplexity artifact in the repo while claiming "quality preserved" (MISS-01).
- Divergent dependency manifests (`requirements.txt` vs `pyproject.toml`, DEP-02) so a fresh clone can't reproduce results.
- Hardcoded model paths (SEC-02) so no one else can re-run the benchmark.

**Phase to address:** EVAL-01, SEC-02, DEP-02, METRICS-03, DOCS-01 — collectively the v1 trust-loop milestone.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| 2,964-method registry as "catalog" | Looks comprehensive; preserves exploration value | Slow startup (SCALE-01), misleading capability, maintenance burden | Only if split into validated vs `experimental/` with maturity labels (REGISTRY-01) |
| Per-tensor 4096-byte SSF alignment | Simple mmap I/O | Gigabytes of padding inflate "compressed" size (TD-03/PERF-01) | Never for small tensors; pack into shared pages |
| `ratio` from `len(dict)`/stage product | Easy to compute | Off-by-orders-of-magnitude ratios (BUG-02) | Never — always `serialized_nbytes` |
| Hardcoded competitor numbers | Avoids implementing GPTQ/AWQ | Fabrication recurrence (SEC-03), trust loss | Never; label estimates or omit |
| `BaseTokenizer` stub with `NotImplementedError` | Defers tokenizer decision | Runtime crash in serving path (BUG-04) | Never; use ABC + tested default |
| FP32-baseline ratio as headline | Bigger number | 2× inflation vs honest baseline (BUG-03) | Never as default; BF16 is the truth |
| Archive tests (`test_archive_*.py`) | Preserve old coverage count | Inflate perceived coverage of dead code (COV-06) | Delete or move with `_archive/` |

---

## Integration Gotchas

Common mistakes when connecting to external services / formats.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| GGUF / safetensors load | Hardcoded author model path (SEC-02) | Read path from env/CLI; publish model hash |
| Tokenizer (HF/tiktoken/sentencepiece) | Base class `encode` raises; concrete impls need undeclared libs (BUG-04/DEP-03) | Ship one self-contained default; ABC so misuse fails at import |
| Competitor methods (GPTQ/AWQ) | Report hardcoded constants as measured (SEC-03) | Only compare methods run in-repo; label literature estimates |
| Downstream eval (lm-eval-harness) | Skip it; claim quality from reconstruction only (MISS-01) | Run perplexity + ≥1 zero-shot task; publish JSON |
| CI (GitHub Actions) | None exists; regressions reappear (MISS-04) | Add pytest + metrics-honesty lint gate |
| Deserialization (SSF/GGUF reader) | No offset/size validation → OOB/read-exhaustion (SEC-04) | Validate offsets vs file size; fuzz/negative tests |

---

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Pure-NumPy GEMM/SVD (PERF-04) | Inference 2–3× slower than llama.cpp; 54 min for 10 GB model | Scope as research; optional BLAS/GPU backend if production wanted | Any real serving load |
| Per-tensor 4096-byte alignment (PERF-01) | Stored size dominated by padding | Shared paged allocation; pack small tensors | Models with many small tensors (all LLMs) |
| Pathological slow spike (PERF-02) | 25× slowdown on one tensor class (full SVD) | Per-tensor timeout; per-method time budget; randomized SVD | Larger/varied models |
| Registry startup cost (SCALE-01) | ~2s just to register 2,964 methods | Lazy registration; only register active config methods | Every CLI invocation |
| In-memory materialization of slices (SCALE-03) | OOM on 256×256+ copies | Memory-mapped ops; cap slice size | Larger weight matrices |

---

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| `signal.alarm`/`exec` timeout (SEC-01) | Crashes on Windows; `exec` of string is an anti-pattern | `concurrent.futures`/`multiprocessing` timeouts; remove `exec` |
| Hardcoded absolute model path (SEC-02) | Non-reproducible; leaks author environment | env/CLI arg; document required artifacts |
| No input validation on deserialization (SEC-04) | OOB reads / resource exhaustion from crafted model files | Validate offsets vs file size; fuzz tests; cap allocation |
| Fabricated benchmark JSON committed (SEC-03) | Misleads users/auditors; reputational | CI lint rejecting hardcoded competitor tables |

---

## UX Pitfalls

Common user experience mistakes in this domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| "2,964 methods" in docs | User thinks 2,964 things work; finds 2 | Show validated count + experimental namespace |
| Headline "4.6×" (vs FP32) | User expects 4.6× vs their BF16 file; gets 2.3× | Lead with ratio_vs_disk; state baseline |
| Cascade presented as flagship | User tries it; gets 72–92% error | Gate behind quality; label experimental |
| No working default tokenizer | `infer` crashes on `token_count` | Ship tested default; fail fast with clear message |
| "Compression engine" framing | User expects production serving | Honest "research-grade CPU" scope up front |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- **[Honest metrics]:** `honest_metrics.py` exists but has **no tests** (COV-03) — verify a high-error payload is rejected before trusting any ratio.
- **[Cascade]:** Documented as 5-stage but only 2 stages run (BUG-01) — verify `used_stages` and real-weight `rel_mse < 0.05`.
- **[Compression ratio]:** Appears as 22.31× but is a shared constant, not measured (BUG-02) — verify `serialized_nbytes` drives the number.
- **[Quality]:** `rel_mse_median = 0.0` looks perfect but 71% are pass-through (TD-05) — verify functional error is reported.
- **[Industry comparison]:** Looks like a benchmark table but is hardcoded (SEC-03) — verify each number was produced by an in-repo runner.
- **[Tokenizer]:** `BaseTokenizer` looks usable but `encode` raises (BUG-04) — verify a concrete default resolves.
- **[Coverage]:** 2,758 test functions look strong but `tensor/agents/utils/benchmark` have ~0 refs (COV-01) — verify the critical subsystems are actually covered.
- **[Reproducibility]:** Benchmarks look real but use hardcoded author path (SEC-02) — verify a fresh clone can re-run them.

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Shameless ratio (BUG-02) | LOW | Route ratio through `serialized_nbytes`; add `test_honest_metrics.py` (COV-03) |
| Fabricated comparison (SEC-03) | LOW | Delete hardcoded tables; re-run only in-repo methods; label estimates |
| Half-wired cascade (BUG-01) | MEDIUM | Disable as default; wire stages or document 2-stage; add quality test (COV-05) |
| Registry bloat (TD-01) | MEDIUM | Quarantine 2,962 into `experimental/`; keep 2 validated; bound-size test |
| Format overhead (TD-03) | MEDIUM | Separate overhead; pack small tensors into shared pages (FORMAT-01) |
| No eval (MISS-01) | MEDIUM | Add Wikitext perplexity original vs compressed; publish JSON |
| dtype overflow (FRAG-01) | LOW | Add overflow check + BF16 fallback; round-trip test (COV-04) |
| Trust loss (historical) | HIGH | v1 honesty milestone: measure everything, pair with error, reproducible |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1 Shannon-limit trap | METRICS-01, METRICS-02, DOCS-01 | bits/weight sanity check; no ratio < entropy floor |
| 2 Eval methodology | EVAL-01, METRICS-04 | Held-out perplexity JSON artifact exists |
| 3 Cascade of estimate errors | METRICS-01, METRICS-04, CASCADE-01 | `test_honest_metrics` rejects high-error payload |
| 4 GPU-vs-CPU fallacy | DOCS-01, DEP-01 | Docs state CPU-NumPy scope; no false speed claims |
| 5 Paper-chasing | REGISTRY-01, REGISTRY-02, CASCADE-02 | New method needs real-weight run before active |
| 6 Registry of stubs | REGISTRY-01, REGISTRY-02, DOCS-02 | Bounded registry size test; maturity labels |
| 7 Calibration pitfalls | SEC-02, BUG-04, MISS-02, EVAL-01 | Reproducible path; tested default tokenizer |
| 8 Comparison fabrication | METRICS-03, MISS-04 | CI lint rejects hardcoded competitor tables |
| 9 Recon ≠ functional error | METRICS-01, TD-05, EVAL-01 | Functional error surfaced; no masked median=0 |
| 10 Format overhead | FORMAT-01, DOCS-01 | Overhead reported separately from algorithmic ratio |
| 11 Pass-through as compression | FORMAT-01, DOCS-01, METRICS-02 | Per-method ratio breakdown published |
| 12 FP32 inflation | METRICS-02, DOCS-01 | ratio_vs_disk is default headline |
| 13 Half-wired cascade | CASCADE-01, CASCADE-02, COV-05 | `rel_mse < 0.05` test; or documented 2-stage |
| 14 dtype overflow | FRAG-01, COV-04 | Overflow-fallback round-trip test |
| 15 Prototype vs tool gap | EVAL-01, SEC-02, DEP-02, METRICS-03, DOCS-01 | v1 trust-loop: measure/pair/reproduce |

---

## Sources

- Zhang et al., "70% Size, 100% Accuracy: Lossless LLM Compression via DFloat11 (NeurIPS 2025)" — BF16 exponent entropy ~2.6 bits; ~11 bits/weight lossless floor; 30% lossless reduction. (HIGH confidence)
- Tan et al., "Approaching Shannon Bound with Lossless LLM Weight Compression" (2026) — LLM weight entropy 2–10× lower than stored bitwidth; ANS within 0.01–0.1 bits of Shannon limit. (HIGH)
- Lifar et al., "WaterSIC: information-theoretically (near) optimal linear layer quantization" (2026) — GPTQ has arbitrary gap to IT limit; column-wise waterfilling; rate-distortion floor. (HIGH)
- Young, "Foundations of LLM Compression — Weight Quantization" / "Radio: Rate-Distortion Optimization" (2024–2025) — quantization error halves per bit; R(D) framing. (HIGH)
- Liu et al., "Evaluating the Generalization Ability of Quantized LLMs" (2024) — calibration-set choice swings downstream accuracy up to 70%; IID ≠ optimal. (HIGH)
- SARQC (2026) — reconstruction-only calibration unreliable at W2/W3; doesn't control ‖W−Ŵ‖. (HIGH)
- Williams & Aletras (2023), ACL 2024 calibration study — GPTQ overfits calibration; perplexity weakly correlates with task accuracy (BoolQ 57–71.6%). (HIGH)
- Hugging Face AQLM docs — fast path Triton/CUDA; CPU Numba path slower; needs GPU kernels + LoRA. (HIGH)
- Cornell-RelaxML/quip-sharp — QuIP# needs CUDA matvec kernels; 3-bit "significantly slower" without optimized kernel. (HIGH)
- Hariri, "Entropy of bfloat16: 8 Bits Are Doing 2.6 Bits of Work" (2025) — exponent entropy ~2.6 bits across Llama/Qwen/Mistral/Gemma. (HIGH)
- fergusfinn.com "In search of wasted bits" — Shannon entropy sets lower bound; 7–30% extraneous bits remain per format. (MEDIUM)
- Project-internal: `.planning/codebase/CONCERNS.md` (TD-01/03/05, BUG-01/02/03/04, SEC-01/02/03/04, PERF-01/02/03/04, COV-01/03/04/05/06, MISS-01/02/04, FRAG-01/04, DEP-01/02/03, SCALE-01/03), `ARCHITECTURE.md` (anti-patterns), `stage_diagnosis.json` (TT captures ~14% variance; SIREN/ergodic +0.00 dB), `benchmark_industry_comparison.json` (identical per-matrix competitor floats), `run_full_model_honest_results.json` (2 methods used of 2,964), `PROJECT.md` (v1 honesty + consolidation scope). (HIGH — first-hand)

---

*Pitfalls research for: LLM weight compression & local inference (Spectral-Stream LLM Engine)*
*Researched: 2026-07-08*
