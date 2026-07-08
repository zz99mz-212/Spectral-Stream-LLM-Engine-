# Feature Research

**Domain:** LLM weight compression + local inference tooling
**Researched:** 2026-07-08
**Confidence:** LOW (per seam: Exa provider tier = LOW). Note: findings below are corroborated across many *independent* sources — arxiv papers (AWQ 2306.00978, GPTQ 2210.17323, Marlin 2408.11743, QuIP#, AQLM, GSQ), official library docs (AutoAWQ, AutoGPTQ, llama.cpp, HF transformers), and multiple 2025-2026 comparison/analysis blogs. Corroboration across independent sources raises practical confidence, but since the seam classifies the provider as LOW, treat specific numeric claims (e.g. 741 tok/s) as illustrative, not authoritative, and re-verify before citing in shipped docs.

## Feature Landscape

### Table Stakes (Users Expect These)

Features the external ecosystem treats as non-negotiable. Missing these = product feels incomplete or untrustworthy.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Deterministic quantization at multiple bit widths (INT8, INT4, INT2) | Every serious engine offers ≥2 bit widths; users pick the size/quality tradeoff | MED | Spectral-Stream has INT8 blockwise + FP16 passthrough verified; INT4 path exists but its quality must be proven. "Multiple bit widths" is met only thinly today. |
| Calibration-data-driven quantization | GPTQ/AWQ/llama.cpp(--imatrix) all use a calibration set to protect salient weights; naive scaling fails at low bits | HIGH | Requires activation statistics over a text corpus. Spectral-Stream's INT8 is blockwise; it is unclear whether it is calibration-aware. This is a likely gap. |
| Decompression/dequantization at inference time | Users expect to load a compressed model and run it | MED | Pure-Python dequant is CPU-feasible (slower). Spectral-Stream has `infer` CLI + GGUF dequantizer. |
| HuggingFace / GGUF / safetensors model loading | safetensors is the HF source-of-truth container; GGUF is the local/CPU standard | MED | Spectral-Stream already has GGUF/safetensors converters + pure-Python GGUF dequantizer (no llama.cpp dep) — covers this. |
| Support for common LLM architectures (Llama, Mistral, Gemma, Qwen) | The four dominant open-weight families | MED | Converter must recognize each arch's tensor naming. GGUF converter handles most HF archs incl. MoE. Confirm Spectral-Stream covers Gemma/Qwen explicitly. |
| Perplexity evaluation vs original model | Wikitext-2 perplexity (seq len 2048) is the de-facto intrinsic quality metric; "compare compressed vs FP16" | MED | **This is Spectral-Stream's single biggest gap (MISS-01).** Zero perplexity/eval exists. EVAL-01 in PROJECT.md is the top external-standard gap. |
| Honest, paired ratio + error reporting | Ecosystem norm is loose: tools report "quality preserved" without byte-exact coupling. The gap is *real honesty*, not a feature the competition has | LOW-MED | **This is Spectral-Stream's actual differentiator**, not a table stake. `honest_metrics.py` exists but is unhardened (COV-03, BUG-02). |

### Differentiators (Competitive Advantage)

Features that set a product apart. For Spectral-Stream, "speed" differentiators are largely off-limits (CPU-only, no GPU kernels) — its real wedge is honesty + the spectral thesis.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Byte-exact ratio + paired error, always coupled** | No competitor enforces this. llama.cpp/GPTQ/AWQ report quality loosely. A method with 85% error silently reporting a 22× ratio is exactly the trap that triggered Spectral-Stream's honesty rewrite | LOW-MED | Spectral-Stream's core value. Harden via METRICS-01/02/03. This is the defensible wedge. |
| **CPU-native, zero C++/GPU dependency** | GGUF is the only CPU-friendly format; everything else (AWQ/GPTQ/EXL2/Marlin) is GPU-only. A transparent pure-Python engine runs anywhere NumPy does | MED | Aligns with Spectral-Stream constraints. Differentiator by being the *honest, portable* option, not the fastest. |
| **Multi-redundancy / spectral compression thesis** | Treating weights as manifolds/fields to exploit low-rank + spectral decay + cross-layer correlation is genuinely novel vs flat-tensor quant | HIGH | Stages 1-2 (EinSort, TT-SVD) live but weak (~14% variance captured); stages 3-5 aspirational. RND track. |
| **Transparency / research catalog of maturity-labeled methods** | 2,964-method registry honestly labeled experimental vs validated — unique; competitors don't expose their method zoo | LOW | REGISTRY-01/02. Differentiator as "honest exploration," not capability. |
| **KV-cache compression policy library** | 30+ eviction/compression policies unified under one adapter — ahead of table stakes vs most standalone quant tools | MED | Spectral-Stream already has KVCacheManager. CPU-only means PagedAttention/FlashAttention infra does not apply, but policy breadth is a genuine edge. |
| **Activation-based pruning (APoZ/Wanda/SparseGPT importance)** | The 2025-2026 frontier is joint sparsity+quantization and per-layer importance-driven compression | HIGH | RND-01/02/03. Must run on CPU-only NumPy; validates on real weights. |
| **OpenAI-compatible local inference server** | Usability differentiator; lets users point apps at the engine | MED | Already present (`infer` CLI). Table-stakes-adjacent for "local inference tool." |
| **Format-overhead separation (algorithmic ratio vs container overhead)** | SSF's 4096-byte page alignment inflates ratio; reporting them separately is honest and unusual | LOW | FORMAT-01. Prevents the "fake ratio" failure mode. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems for *this* project specifically.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Yet another GGUF wrapper | "Just support GGUF like everyone else" | The ecosystem already has llama.cpp/Ollama/LM Studio — undifferentiated, and GGUF in vLLM is 93 tok/s (unpack overhead). Spectral-Stream cannot win on GGUF compatibility | Use GGUF as an *interop boundary* (convert to/from), keep SSF as the honest native format |
| GPU-cluster-requiring features (Marlin, vLLM serving) | "Match the speed leaders" | Violates pure-Python/CPU-only constraint; would require CUDA kernels + rewrite. Not the project's lane | Stay CPU-native; compete on honesty/transparency, not throughput |
| Opaque black-box compression (no transparency on quality loss) | "Just give me the smallest file" | Directly contradicts the project's Core Value; this is the exact failure mode that produced the honesty rewrite (c66016e) | Always pair ratio with byte-exact error; require calibration/perplexity evidence |
| Fabricated industry comparisons / aspirational ratios as capability | "Impressive numbers sell" | SEC-03 — fabricated benchmark in `benchmark_industry_comparison.json`. Kills trust permanently | METRICS-03: real measurements or explicit "literature estimate, not measured here" labels |
| Sub-1-bit / 800×+ "achieved" numbers | "Headline ratios" | Require training-from-scratch (BitNet) or are theoretical upper bounds; not post-training. Out of scope per PROJECT.md | Keep in research catalog, labeled as targets/upper bounds, never as achievements |
| Full production serving stack (load balancing, horizontal scaling, SLAs) | "Production-ready" expectation | Research-grade maturity; would dilute the honesty/consolidation v1 focus | Explicitly out of scope; document as such |

## Feature Dependencies

```
[Perplexity eval (EVAL-01)]
    └──requires──> [Calibration-data-driven quant] (needs activations + original model loader)
                        └──requires──> [HF/GGUF/safetensors loading] (table stakes, exists)

[Honest ratio+error coupling (METRICS-01/02)]
    └──requires──> [serialized_nbytes + end_to_end_error] (honest_metrics.py, exists but untested)
    └──enhances──> [All reported ratios] (every advertised number must flow through it)

[Registry pruning (REGISTRY-01/02)]
    └──enhances──> [Honesty story] (separates validated vs experimental)

[SSF overhead separation (FORMAT-01)]
    └──enhances──> [Honest ratio+error coupling] (algorithmic ratio ≠ container ratio)

[Activation-based pruning (RND-01)]
    └──requires──> [HF loading + inference path] (needs real forward passes for activations)
    └──enhances──> [Multi-redundancy thesis] (spectral + pruning combined)

[KV-cache policy library (exists)]
    └──conflicts──> [GPU PagedAttention infra] (CPU-only; block-wise eviction only, not page-mapped)

[Inference-speed differentiators (Marlin/speculative/EXL2)]
    └──conflicts──> [Pure-Python CPU-only constraint] (NOT buildable; explicitly anti-feature)
```

### Dependency Notes

- **Perplexity eval requires calibrated loading + original-model reference:** You cannot measure "compressed vs original" without (a) loading the original FP16/GGUF and (b) a calibration/text corpus. Spectral-Stream's `infer` path + GGUF loader already partially satisfy (b); (a) is the gap.
- **Honest metrics enhance everything:** METRICS-01/02 are the linchpin — once ratio↔error is coupled and enforced, every other feature's numbers become trustworthy. Do this first.
- **Inference-speed features conflict with the core constraint:** Marlin/speculative decoding/EXL2 are GPU-kernel features. Building them would break the pure-Python/CPU-only promise. Treat as anti-features.
- **KV-cache policy library conflicts with GPU paging infra:** Spectral-Stream's 30+ policies run CPU-side; block-wise eviction is compatible but page-mapped PagedAttention-style infra is not.

## MVP Definition (v1 = honesty + consolidation)

### Launch With (v1)

- [x] **Multiple deterministic bit widths (INT8 + FP16)** — already verified; enough to claim "multiple widths" honestly
- [ ] **Honest ratio+error coupling (METRICS-01/02/03)** — error-gate all ratios; BF16 the headline; no fabricated comparisons. *The linchpin.*
- [ ] **Perplexity eval vs original (EVAL-01)** — at least Wikitext-2 perplexity on original vs compressed. *Closes the biggest external-standard gap.*
- [x] **HF/GGUF/safetensors loading + conversion** — already present; confirm Gemma/Qwen arch coverage
- [ ] **Registry split validated vs experimental (REGISTRY-01/02)** — honesty of the method catalog
- [ ] **SSF overhead separation (FORMAT-01)** — report algorithmic ratio separately from 4096-byte page overhead

### Add After Validation (v1.x)

- [ ] **Calibration-driven quantization** — make INT8/INT4 calibration-aware (imatrix-style) to protect salient channels; required to compete at <4 bit
- [ ] **INT4 path with proven quality** — only advertise once perplexity proves rel_mse < threshold
- [ ] **Activation-based pruning (RND-01)** — APoZ/Wanda importance scoring, validated on real weights
- [ ] **KV-cache INT8 quantization policy** — bring KVCacheManager to the INT8-KV production bar

### Future Consideration (v2+)

- [ ] **Joint sparsity+quantization hybrid (RND-02/03)** — per-layer importance → adaptive mixed precision
- [ ] **Spectral multi-redundancy stages 3-5 wired** — only after stages 1-2 proven on real weights (CASCADE-01)
- [ ] **Structured pruning + distillation** — research-grade, GPU-light, CPU-feasible but expensive
- [ ] **Low-bit (<4bit) scalar methods (GSQ-class)** — CPU-feasible scalar quant that closes vector-quant gap

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Honest ratio+error coupling (METRICS-01/02/03) | HIGH | LOW | P1 |
| Perplexity eval vs original (EVAL-01) | HIGH | MED | P1 |
| HF/GGUF/safetensors loading | HIGH | LOW (exists) | P1 |
| Multiple deterministic bit widths (INT8/FP16/INT4) | HIGH | MED | P1 |
| Registry validated vs experimental split | MED | MED | P1 |
| SSF overhead separation (FORMAT-01) | MED | LOW | P1 |
| Calibration-driven quantization | HIGH | HIGH | P2 |
| Activation-based pruning (RND-01) | MED | HIGH | P2 |
| KV-cache INT8 quantization policy | MED | MED | P2 |
| OpenAI-compatible inference server | MED | MED (exists) | P2 |
| Multi-redundancy spectral stages 3-5 | MED | HIGH | P3 |
| Joint sparsity+quantization hybrid | MED | HIGH | P3 |
| Low-bit scalar methods (GSQ-class) | LOW-MED | HIGH | P3 |
| Marlin/speculative/EXL2 GPU kernels | N/A (incompatible) | N/A | ANTI |

**Priority key:**
- P1: Must have for v1 launch (honesty + consolidation)
- P2: Should have, add after validation
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | llama.cpp (GGUF) | AutoGPTQ/AutoAWQ + vLLM | ExLlamaV2 (EXL2) | Spectral-Stream (our approach) |
|---------|------------------|--------------------------|------------------|-------------------------------|
| Bit widths | Q2–Q8 K-quant + IQ | 2/3/4/8 (AWQ 4-bit primary) | Fractional per-layer (2-8) | INT8 + FP16 verified; INT4 partial |
| CPU inference | Excellent (SIMD) | No (GPU only) | No (GPU only) | Yes (pure-Python, slower) |
| Calibration | Optional (--imatrix) | Required | Required | Not yet calibration-aware (gap) |
| Honest ratio+error | No (loose quality) | No (loose quality) | No (loose quality) | **Yes — enforced coupling (value)** |
| Perplexity eval | Via llama-perplexity tool | Via external eval | Via external eval | **Missing (MISS-01) — must build** |
| KV-cache compression | INT8 KV, attention-sink | INT8 KV (vLLM) | INT8 KV | 30+ policies (ahead, CPU-only) |
| Speed differentiators | CPU SIMD | Marlin kernel (741 tok/s) | Fastest single-user | None (CPU, honest instead) |
| Format | Single-file GGUF | safetensors + config | safetensors | SSF (own) + GGUF interop |
| Transparency of method zoo | N/A | N/A | N/A | **Maturity-labeled 2,964 catalog (unique)** |

## Gaps: Spectral-Stream vs External Standard

Specific, project-current gaps the roadmap must close (these feed requirements definition directly):

1. **Perplexity evaluation (CRITICAL).** Zero downstream/perplexity eval exists. External standard = Wikitext-2 perplexity vs original FP16. Maps to EVAL-01. *Highest-priority gap.*
2. **Calibration-driven quantization.** Blockwise INT8 may not be calibration-aware. External tools all use calibration/imatrix to protect salient channels at low bits. Needed before INT4 can be honestly advertised.
3. **INT4 path quality unproven.** INT4 exists but no perplexity evidence; cannot claim "multiple bit widths" with confidence until INT4 passes error gate.
4. **Honest-metrics hardening.** `honest_metrics.py` exists but has no tests (COV-03) and a method with 85% error still reports 22× ratio (BUG-02). METRICS-01/02 close this.
5. **Fabricated comparison removal.** `benchmark_industry_comparison.json` / `certificate.py` contain fabricated industry numbers (SEC-03). METRICS-03.
6. **BF16 as headline ratio.** Currently `ratio_vs_fp32` is default; external context is BF16 disk. METRICS-02.
7. **Registry honesty.** 2,964 methods, but real full-model run used only fp16_passthrough + int8. REGISTRY-01/02 split validated vs experimental.
8. **SSF overhead disclosure.** 4096-byte page alignment inflates ratio; must report algorithmic ratio separate from container. FORMAT-01.
9. **Arch coverage confirmation.** Confirm Gemma-4, Qwen, Mistral explicitly loadable (PROJECT cites Gemma-4 E2B as the verified model).

**Already ahead of / at table stakes:** GGUF/safetensors loading + pure-Python GGUF dequantizer (no llama.cpp dep); 30+ KV-cache policies; OpenAI-compatible inference server; honesty infrastructure. These should be preserved and hardened, not rebuilt.

## Sources

- local-llm.net — GGUF vs GPTQ vs AWQ vs EXL2 format comparison (2026)
- AI/TLDR — GGUF vs GPTQ vs AWQ decision tree (2026)
- oobabooga blog — GPTQ/AWQ/EXL2/llama.cpp perplexity + speed comparison
- RunLocalAI — Quantization formats (GGUF/AWQ/GPTQ/EXL2/FP8/MLX), May 2026
- inventivehq.com — What Is GGUF? format explainer (2026)
- arxiv 2306.00978 (AWQ), 2210.17323 (GPTQ), 2408.11743 (Marlin), 2402.17764 (1.58-bit), 2401.06118 (AQLM), 2402.04396 (QuIP#), 2308.13137 (OmniQuant)
- HuggingFace transformers docs — quantization (awq/gptq), GGUF loading
- AutoAWQ GitHub — quantizer.py, examples.md (calibration, imatrix, GGUF export)
- llama.cpp — quantize.cpp (--imatrix, --pure), convert-hf-to-gguf.py
- DeepResearch Ninja — LLM Quantization Methods Comparative Analysis (2026-07)
- arxiv 2310.01382 — "Compressing LLMs: The Truth is Rarely Pure" (perplexity limitations)
- arxiv 2409.11233, aclanthology 2023.findings-emnlp.349 — perplexity insufficiency, JS divergence
- arxiv 2602.09130 (UniComp) — 13-metric eval framework
- NVIDIA blog — Speculative Decoding (2025); Red Hat — Marlin kernels (2024)
- arxiv 2603.20397, 2410.00161 (KV-Compress), 2403.04643 (QAQ), blog.prompt20.com — KV cache four-lever framework
- HF blog — Common AI Model Formats (safetensors/GGUF)

---

*Feature research for: LLM weight compression + local inference (external ecosystem)*
*Researched: 2026-07-08*
*Confidence: LOW per seam (Exa tier); findings corroborated across independent papers/docs/blogs — re-verify numeric claims before shipping in docs.*
