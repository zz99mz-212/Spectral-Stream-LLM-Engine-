# Project Research Summary

**Project:** Spectral-Stream LLM Engine
**Domain:** Pure-Python (NumPy/SciPy, CPU-only, no torch) LLM weight & KV-cache compression + local inference, with honesty-first v1 mandate
**Researched:** 2026-07-08
**Confidence:** HIGH for stack/pitfalls; MEDIUM for architecture; LOW for feature seams (Exa tier) — though feature claims corroborate across independent papers/docs and are treated as practical-MEDIUM.

---

## Executive Summary

Spectral-Stream is a brownfield re-init of a pure-Python LLM compression engine whose v1 priority is **honesty + consolidation**, not new capability. The dominant finding across every research dimension is that the engine already has the right *building blocks* (`honest_metrics.py`, SSF format, GGUF/safetensors converters, KVCacheManager, registry orchestrator) but the **trust loop is broken**: ratios are disconnected from error (BUG-02), competitor numbers are fabricated (SEC-03), the registry advertises 2,964 methods when only 2 run on real weights (TD-01), and there is **zero perplexity/downstream eval** (MISS-01) — the single biggest external-standard gap. The 5-stage cascade's 72–92% reconstruction error is not a math bug but a *missing calibration → OBS-compensation channel* (ARCHITECTURE); standard INT4 succeeds precisely because it collects activations and propagates quantization error column-by-column.

The recommended approach is to spend v1 building the **trust loop** (measure every number end-to-end, pair it with error, gate it behind a threshold, never compare to un-run competitors) and splitting the catalog into validated vs experimental. The biggest risk is treating the v1 honesty work as bug-fixing rather than infrastructure: without an eval subsystem and an enforced metrics gate, every future method's numbers remain unverifiable and the project stays in "research prototype" territory. Calibration and eval are the **critical path** — neither the differentiators (RND track) nor any credible quality claim can proceed without them.

---

## Key Findings

**1. Eval subsystem is the #1 gap and the linchpin of honesty (MISS-01 / EVAL-01).**
No perplexity or downstream-task eval exists at all. The field's gold standard is WikiText-2 perplexity (ΔPPL < 0.2 at INT4) plus a recovery gate (`compressed/base ≥ 0.95`). Every real tool (llama.cpp, AutoGPTQ, ExLlama) proves quality this way. Until Spectral-Stream runs `lm-evaluation-harness` on original vs compressed weights, it cannot claim "quality preserved" — and it cannot honestly advertise INT4 (only INT8 is verified). *This must land before adding more methods.*

**2. The metrics gate exists but is unenforced (BUG-02 / COV-03 / METRICS-01/02/04).**
`honest_metrics.py` provides byte-exact `serialized_nbytes()` + `end_to_end_error()`, but has no tests and the old `len(dict)`/per-stage-product behavior leaks back. A method with 85% error still reports 22.31×. The fix is small (route every ratio through `serialized_nbytes`, gate `rel_mse > threshold` from emitting a ratio, add `test_honest_metrics.py`) but is the load-bearing honesty safeguard. Lead with `ratio_vs_disk` (BF16), never `ratio_vs_fp32` — the project's honest ~2.3× vs BF16 is exactly half of the advertised 4.6× vs FP32.

**3. The INT4/cascade failure is architectural, not tunable (CASCADE-01 / ARCHITECTURE core diagnosis).**
The 5-stage cascade (72–92% error) and INT8's ~4.6× ceiling both stem from operating on weight tensors with **no activation/calibration context and no OBS error-compensation**. Standard INT4 (GPTQ/AWQ ≈ 5.60–5.63 PPL, ~3.9× vs BF16) works *because* it captures per-layer activations and propagates error column-by-column. Spectral-Stream has no calibration subsystem at all. This is a structural blocker for the entire RND track (APoZ/Wanda/SparseGPT) and any honest INT4 claim — **build `calibration/` first**.

**4. The registry is a 2-method toolkit wearing a 2,964-method catalog (TD-01 / REGISTRY-01/02).**
Honest full-model run used only `fp16_passthrough` (71% of tensors) + `int8_blockwise` (29%). The rest are aspirational stubs, several of which are CUDA/fine-tuning/kernel-dependent and cannot run in a pure-NumPy CPU codebase (QuIP#, AQLM, SIREN = +0.00 dB). Split into a validated active set (tests + real-weight results) vs a labeled `experimental/` namespace; kill the broken walk-based auto-discovery (`_discover_by_walk`).

**5. Format overhead is masquerading as compression (TD-03 / PERF-01 / FORMAT-01).**
SSF aligns every tensor to 4096-byte pages + 256B header + 128B footer. With 2,011 tensors (71% small pass-through), padding dominates stored size — the headline "2.3× vs BF16" is carried mostly by container overhead, not algorithmic savings. Report algorithmic ratio separate from overhead; pack small tensors into shared pages; stop counting pass-through as "compression".

**6. The defensible wedge is honesty + portability, NOT speed (FEATURES differentiators).**
CPU-NumPy is 2–3× slower than llama.cpp at best and has no accelerated backend (PERF-04/DEP-01). The real differentiators are: enforced byte-exact ratio↔error coupling (no competitor does this), CPU-native zero-dependency portability, the maturity-labeled 30+ KV-cache policy library, and transparency of the method catalog. GPU-serving/Marlin/EXL2/speculative-decoding are explicit anti-features per PROJECT.md constraints.

**7. The trust loop requires reproducible, measured baselines (SEC-02/03, DEP-02, MISS-04).**
Hardcoded author model path (`/home/mike/.../gemma-4-E2B`), divergent `pyproject.toml` vs `requirements.txt`, fabricated `benchmark_industry_comparison.json` (identical per-matrix competitor floats), and `BaseTokenizer.encode` crashing (BUG-04) all block a fresh clone from reproducing results. v1 honesty = measure → publish → reproduce, with a CI metrics-lint that rejects hardcoded competitor tables.

---

## Implications for Roadmap

Research converges on a clear critical path. **Calibration and Eval are prerequisites to everything else**; metrics-gating is the linchpin that makes any claim trustworthy; registry/format hygiene can parallelize later; RND pruning depends strictly on calibration; inference promotion depends on a stable format contract.

### Phase 1: Metrics Trust Loop (METRICS-01/02/03/04 + CASCADE-01 gate)
**Rationale:** Every other number in the project is untrustworthy until ratios are error-gated and measured end-to-end. Cheapest, highest-leverage fix; unblocks all downstream validation.
**Delivers:** `serialized_nbytes`-driven ratios, `rel_mse > threshold` gate (config-based), `test_honest_metrics.py`, BF16 headline default, removal/re-labeling of fabricated comparison, cascade disabled-as-default until `rel_mse < 0.05`.
**Avoids:** Pitfall 1, 3, 8, 12.

### Phase 2: Eval Subsystem (EVAL-01, MISS-02, BUG-04, SEC-02)
**Rationale:** Eval must exist *before* any quality claim or new method. Independent "grader" that proves honesty and validates later phases. Closes reproducibility gaps (tokenizer ABC, env/CLI model path, default tokenizer).
**Delivers:** `eval/` subsystem (harness adapter → WikiText-2 perplexity → recovery gate ≥0.95), real original-vs-compressed PPL JSON artifact, tested default tokenizer, reproducible model loading.
**Research flag:** MEDIUM — pure-NumPy perplexity without torch is non-standard; lm-eval adapter API needs validation in a research-phase.

### Phase 3: Calibration Substrate (ARCHITECTURE `calibration/` + RND precursor)
**Rationale:** Critical path for the entire differentiation story. Every RND method and any honest INT4 improvement needs per-layer activation stats (X, Hessian H⁻¹) that no current method receives.
**Delivers:** `calibration/` subsystem (dataset → tokenizer → forward hooks → per-layer activation capture, torch-free), defines the activation-stats protocol consumed by `methods/`.
**Research flag:** MEDIUM — torch-free activation capture on CPU-only is non-trivial; may need external oracle for the base-model forward pass.

### Phase 4: INT4 with OBS Compensation (upgrade verified path, fixes CASCADE-01 honestly)
**Rationale:** Uses calibration (Phase 3) + recovery gate (Phase 2) to build real groupwise INT4 (~3.9× vs BF16 / <0.2 PPL), replacing the broken cascade as headline. Only after this passes perplexity may INT4 be advertised.
**Delivers:** Groupwise INT4 with column-wise error compensation, validated against EVAL-01; de-wire or honestly document the 5-stage cascade.
**Research flag:** MEDIUM — Cholesky H⁻¹ in pure NumPy on CPU may hit PERF limits; validate feasibility.

### Phase 5: Registry & Format Hygiene (REGISTRY-01/02, FORMAT-01, DOCS-01/02)
**Rationale:** Lower-risk, parallelizable with Phase 4. Decouples container overhead from algorithmic ratio; stops catalog masquerading as capability.
**Delivers:** Validated vs `experimental/` split, bounded registry-size test, removed auto-discovery, SSF overhead separation + small-tensor packing, honest README/Research-Catalog docs.

### Phase 6: RND Activation-Pruning Track (RND-01/02/03, prune-first invariant)
**Rationale:** Depends strictly on Phase 3 calibration. Per Architecture pattern, prune on dense weights *before* quantization (never quant-then-prune) — encode as pipeline invariant.
**Delivers:** APoZ/Wanda/SparseGPT importance scoring, neuroanatomy profiling, dynamic per-layer reduction — all gated behind "runs on real weights."

### Phase 7: Inference Subsystem Promotion
**Rationale:** Depends on stable format contract (Phase 5). Promotes thin `infer` CLI to first-class subsystem.
**Delivers:** `inference/` (loader → dequant kernel → model → KV), preserved OpenAI-compatible server.

### Phase Ordering Rationale
- **Calibration + Eval are the critical path:** no differentiators and no credible claim without them.
- **Metrics gate first:** cheapest fix; makes all later validation trustworthy.
- **Registry/format hygiene parallelizes with INT4:** both lower-risk, non-blocking.
- **Pruning strictly after calibration:** APoZ/Wanda/SparseGPT need activations.
- **Inference last:** depends on stable format contract, not on honesty critical path.

---

## Decisions Needed

1. **Tokenizer strategy (MISS-02 / BUG-04).** Which single self-contained default tokenizer ships (HF `transformers`? `tiktoken`? vendored sentencepiece)? Pulls optional dep into core; affects pure-Python boundary. Decide before Phase 2.
2. **Eval execution model.** Native NumPy perplexity (slow, dependency-free) vs external `lm_eval` subprocess oracle (cleaner, adds dev dep). Affects whether EVAL-01 is in-core or CI-only.
3. **Base-model forward pass for calibration (Phase 3).** Pure-Python forward (small CPU models only) vs external oracle producing activation stats for the pure-Python pipeline.
4. **Active-set admission threshold.** Exact `rel_mse`/recovery-gate numbers defining "validated" — research suggests ΔPPL<0.2 (INT4) and recovery ≥0.95, but set the in-repo bar from the INT8 baseline.
5. **GGUF writer vs SSF-only.** Add GGUF *writer* for llama.cpp interop (STACK suggests) or keep GGUF as import/interop boundary only (PROJECT treats SSF as native).
6. **v1 vs v1.x boundary.** FEATURES puts calibration-driven INT4 in "v1.x," but CASCADE-01 implies INT4 work in v1. Clarify whether honest INT4 is v1 or v1.x.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Primary sources: GPTQ/AWQ/QuIP#/AQLM papers, llama.cpp/GGUF spec, lm-eval-harness, GPTQModel (2025–2026). External-oracle recommendation unambiguous given no-torch constraint. |
| Features | LOW (seam) / MEDIUM (practical) | Exa tier = LOW; corroborated across independent arxiv/official docs/blogs. Numeric speed claims illustrative, re-verify before docs. |
| Architecture | MEDIUM | Comparison target well-sourced; project-specific diagnosis grounded in first-hand `stage_diagnosis.json` + CONCERNS.md. |
| Pitfalls | HIGH | Domain theory (2024–2026 entropy/rate-distortion literature) + first-hand `.planning/codebase/` audit + committed JSONs. |

**Overall confidence:** HIGH for core recommendation (build trust loop; calibration+eval critical path). MEDIUM for exact execution of eval/calibration on pure-NumPy CPU (open decisions above).

### Gaps to Address
- **Pure-NumPy perplexity feasibility:** no precedent for torch-free WikiText-2 PPL at scale; validate in Phase 2 research-phase. Mitigate: external `lm_eval` oracle fallback.
- **CPU-only activation capture:** unproven whether pure-NumPy forward is viable; mitigate via external oracle producing activation stats.
- **INT4 OBS in NumPy performance:** Cholesky H⁻¹ per layer on CPU may be slow for large models; validate on Gemma-4 E2B first.
- **Tokenizer dependency boundary:** "no torch" constraint doesn't explicitly forbid `transformers`/`tiktoken`; chosen tokenizer may cross pure-Python line. Resolve in Decision 1.

---

## Sources

### Primary (HIGH confidence)
- GPTQ paper (Frantar et al., ICLR 2023, arXiv:2210.17323) — INT4 g128 = 5.63 PPL.
- AWQ paper (Lin et al., MLSys 2024, arXiv:2306.00978) — INT4 g128 = 5.60.
- QuIP# / AQLM / SpQR papers (IST-DASLab / Vahe1994) — sub-4-bit bpw.
- llama.cpp `tools/quantize` README — GGUF tables (Q4_K_M = 3.27× vs F16).
- lm-evaluation-harness (EleutherAI, v0.4.9) — perplexity/C4/MMLU.
- GPTQModel (ModelCloud, v7.0.0, 2026) — AutoGPTQ/AutoAWQ deprecation.
- DFloat11 (NeurIPS 2025) / Hariri "Entropy of bfloat16" (2025) — ~11 bits/weight lossless floor.
- "Generalization Ability of Quantized LLMs" (2024) — calibration-set swings accuracy up to 70%.
- Spectral-Stream first-hand: `.planning/codebase/CONCERNS.md`, `stage_diagnosis.json`, `run_full_model_honest_results.json`, `benchmark_industry_comparison.json`, `PROJECT.md`.

### Secondary (MEDIUM confidence)
- GGUF format spec (ggml-org) — header/KV metadata/tensor info.
- Safetensors docs — safe serialization, NOT compression.
- AutoAWQ `quantizer.py` — calibration (`max_calib_samples=128`, `seq_len=512`).
- llm-compressor (neuralmagic/vllm) — recovery gate ≥0.95.
- Wanda (locuslab) — `|W|·‖X‖`, prune-first.
- SparseGPT (arXiv:2301.00774) — OBS pruning, joint prune+quantize.

### Tertiary (LOW confidence — re-verify before docs)
- oobabooga / RunLocalAI / inventivehq blogs — format/speed comparisons.
- HQQ repo/bench — calibration-free W4=5.62 (community).
- arXiv 2310.01382 / 2409.11233 — perplexity insufficiency.

---

*Research completed: 2026-07-08*
*Ready for roadmap: yes*
