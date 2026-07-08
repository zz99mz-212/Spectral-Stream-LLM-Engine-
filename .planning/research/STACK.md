# Stack Research

**Domain:** LLM weight compression & CPU-based local inference
**Researched:** 2026-07-08
**Confidence:** HIGH (primary sources: original papers, official docs, HF Transformers quantization overview; dated 2024–2026)

> Purpose of this document: feed roadmap creation for the Spectral-Stream LLM Engine brownfield re-init. The project is a pure-Python (NumPy/SciPy, no C++, no torch, CPU-only) compression + inference engine with a verified INT8/INT4 path and a known honesty problem in its benchmarks. This research surveys the **external ecosystem** so the project can (a) replace fabricated industry comparisons with real data (METRICS-03) and (b) adopt real eval infrastructure (EVAL-01). Recommendations are filtered through the project's hard constraint: **pure Python, CPU, no torch in the core path.**

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **GGUF** (read/dequant via llama.cpp spec) | llama.cpp master (2025–2026) | On-disk format for CPU/local inference; the de-facto standard the project must interoperate with | GGUF is the only widely deployed format whose reference implementation (`llama.cpp`, pure C++) is CPU-first. The project already has a pure-Python GGUF dequantizer — keep it, use it as the gold baseline for "real-world compression ratio vs BF16" comparisons. |
| **safetensors** | v0.8.0 (2025) | Safe, zero-copy, pickle-free tensor serialization | Universal interchange for every PTQ tool. NOT compression (stores FP16/BF16 at full precision). Use as a **lossless validation baseline** and export target, not a compression method. |
| **lm-evaluation-harness** (EleutherAI) | v0.4.9 (2025-06) | Gold-standard eval: perplexity (WikiText2, C4) + zero-shot (MMLU, etc.) | The field's accepted way to measure "compression quality." Replaces fabricated benchmark numbers. The project must add a WikiText2 perplexity eval (EVAL-01) — this is where the real quality signal lives. |
| **GPTQModel** (ModelCloud) | v7.0.0 (2026-04) | Reference GPTQ/AWQ quantization (calibration, kernels) | The actively maintained successor to AutoGPTQ/AutoAWQ; HF Transformers fully deprecated the old libs for it (2025-12). Use as the **reference for what real INT4 ratios/accuracy look like** — but it is CUDA/torch, so only as an external measurement oracle, not a dependency. |
| **NumPy / SciPy** (unchanged) | NumPy ≥1.24, SciPy ≥1.10 | The project's existing core compute | Constraint from PROJECT.md. All compression math stays here. The whole point is a pure-Python engine, so we benchmark *against* torch-based tools, not adopt them. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **transformers** (HF) `Perplexity`/`GPTQConfig` | v5.x (2026) | Reference calibration configs (bits, group_size, dataset) | Source of truth for PTQ hyperparameters (group_size=128, dataset=c4/wikitext2, 128–512 samples). Mirror these in the project's own (torch-free) calibration pipeline. |
| **optimum-quanto** | latest | torch-based int2–int8, QAT, torch.compile | Reference only. Shows the state-of-the-art "flexible PTQ + QAT" design the project's RND track can aspire to (esp. mixed-bit). Not a dependency. |
| **torchao** | PyTorch 2.8+ | Native INT4/INT8 CPU kernels (AMX/AVX-512) | Reference only. Defines the *best-in-class CPU kernel layout* (packed INT4 + group scales, group_size=128). The project's storage layout should be structurally comparable so ratios are apples-to-apples. |
| **AutoRound** (Intel) | latest | INT2–INT8 PTQ via signed gradient descent, 128-sample calibration | Reference for "low-bit accuracy without huge cost." Its calibration recipe (128 samples, pile-10k, seqlen=2048) is a good target for the project's own calibration pipeline. |
| **HQQ** (mobiusml / Dropbox) | latest | Calibration-free on-the-fly INT1–INT8 quantization | Reference for "no-calibration" quality ceiling. Useful comparison point: HQQ W4 = 5.62 PPL on LLaMA-2-7B with zero calibration — a strong honest baseline the project's methods should beat or match. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| **lm-eval-harness CLI** (`lm_eval`) | Run perplexity + MMLU evals | Invoke as an external subprocess/oracle on the dequantized model to produce real PPL deltas. |
| **llama.cpp `llama-quantize` + `imatrix`** | Produce reference GGUF quants for comparison | Use to generate the "what real tools achieve" row in METRICS-03 (external, not imported). |
| **Hugging Face `datasets`** | Load calibration corpora (c4, wikitext, pile) | Even in a torch-free project, the *raw text* from these datasets is what calibration consumes. |
| **GitHub Actions / pytest** | Harden honest_metrics (METRICS-04) | The project's COV-03 gap: honest_metrics.py has no tests. |

---

## The Real Numbers (for METRICS-03 — replace fabricated comparisons)

All perplexity (PPL) figures are **WikiText2 (lower is better)** unless noted. FP16/FP32 reference is the uncompressed model. Ratios are vs BF16/FP16 (≈2× of vs FP32). These are **paper/independent-benchmark claims**, not measurements made by this project — label them as such.

### Per-method claimed results

| Method | Bits | Group | Model | PPL (vs FP16) | Ratio vs BF16 | Source / Confidence |
|--------|------|-------|-------|---------------|---------------|---------------------|
| **RTN** (round-to-nearest, baseline) | INT4 | g128 | LLaMA-2-7B | 5.73 (FP16 5.47) | ~3.9× | AWQ paper / HIGH |
| **GPTQ** | INT4 | g128 | LLaMA-2-7B | 5.63 (FP16 5.47, Δ0.16) | ~3.9× | GPTQ paper (IST-DASLab) / HIGH |
| **GPTQ** | INT3 | g128 | LLaMA-2-7B | 6.61 | ~5.3× | GPTQ paper / HIGH |
| **GPTQ** | INT4 | — | OPT-175B | ≤0.25 PPL loss | ~4× | GPTQ paper / HIGH |
| **AWQ** | INT4 | g128 | LLaMA-2-7B | **5.60** (beats GPTQ 5.63, RTN 5.73) | ~3.9× | AWQ paper (MIT Han Lab) / HIGH |
| **AWQ** | INT4 | g128 | Mistral-7B | 4.30 (FP16 4.14) | ~3.9× | AWQ paper / HIGH |
| **AWQ+GPTQ** | INT2 | g64 | OPT-13B | 13.25 (RTN 17564) | ~8× | AWQ paper / HIGH |
| **QuIP#** | INT2–INT4 | incoherence | LLaMA-2-7B | 8.22 @ 2.02 bpw | ~4.0× @2-bit | QuIP# paper / HIGH |
| **AQLM** | 2–3 bit | 8D codebook | LLaMA-2-7B | 6.65 @ 2.29 bpw (w/ block fine-tune) | ~7× @2-bit | AQLM paper / HIGH |
| **HQQ** | INT4 | g128 | LLaMA-2-7B | 5.62 (**no calibration**) | ~3.9× | HQQ repo/bench / MEDIUM |
| **HQQ** | INT2 | g64 | LLaMA-2-7B | 8.3 | ~8× | HQQ bench / MEDIUM |
| **HQQ** | INT1 | g64 | LLaMA-2-7B | 14.7 | ~16× | HQQ bench / MEDIUM |
| **SpQR** | mixed (outlier FP16 + INT4) | per-row | LLaMA-2-7B | ≈ FP16 (tiny loss) | ~3.5–4× | SpQR paper / HIGH |

### llama.cpp GGUF quants (Llama-3.1-8B, official quantize README)

| Quant | bits/weight | Size (GiB) | vs F16 (14.96 GiB) | Note |
|-------|-------------|------------|--------------------|------|
| F16 | 16.00 | 14.96 | 1.0× | reference |
| Q8_0 | 8.50 | 7.95 | 1.88× | near-lossless |
| Q6_K | 6.56 | 6.14 | 2.44× | |
| Q5_K_M | 5.70 | 5.33 | 2.81× | |
| Q4_K_M | 4.89 | 4.58 | **3.27×** | sweet spot for local CPU |
| Q4_K_S | 4.67 | 4.36 | 3.43× | smallest common 4-bit |

**Gold-standard quality threshold:** acceptable INT4 loss is **<0.1–0.2 PPL on WikiText2 vs FP16**. Real tools sit at Δ0.13–0.16 (GPTQ/AWQ). Anything above ~0.3 PPL is considered degraded. Use these as the honesty bar for the project's INT8/INT4 path.

---

## Installation

The project must NOT adopt torch-based tooling into the core. The recommended "stack" is a set of **external oracles** used only to produce reference measurements for METRICS-03 and EVAL-01. They run as subprocesses, not imports.

```bash
# Core (unchanged — pure Python constraint)
pip install "numpy>=1.24" "scipy>=1.10"

# External eval/reference oracles (dev/CI only, NOT in core path)
pip install "lm-eval==0.4.9"          # perplexity + MMLU harness
pip install "datasets"                # raw calibration corpora (c4, wikitext)
# llama.cpp: build from source or use prebuilt binary for `llama-quantize` + `imatrix`
# GPTQModel/AutoRound: only if a CUDA machine is available for producing reference quants
```

> **Do not** add `torch`, `transformers`, `auto-gptq`, or `autoawq` to the core dependency set. They violate the pure-Python/no-torch constraint and are only used as external measurement references.

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| GGUF as the real-world ratio baseline | EXL2/EXL3 (ExLlama) | Only if targeting GPU ExLlamaV2 inference; EXL3 is GPTQModel-supported but GPU-centric. Not relevant for CPU. |
| lm-eval-harness for PPL | `transformers` `Perplexity` pipeline | Fine for a quick single-model PPL; lm-eval is preferred for reproducibility + MMLU coverage. |
| GPTQModel as reference oracle | AutoGPTQ / AutoAWQ | **Never** — both deprecated/unmaintained (2025-12). GPTQModel is the only current path. |
| safetensors as interchange | Legacy `.bin` / pickle | Never — pickle is unsafe; safetensors is the standard. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **AutoGPTQ** | Officially unmaintained; HF deprecated it for GPTQModel (2025-12). | GPTQModel (as external oracle only) |
| **AutoAWQ** | Officially deprecated (last tested Torch 2.6 / Transformers 4.51). | GPTQModel or AWQ reference numbers from the MIT paper |
| **Fabricated "industry comparison" JSON** | The project's own SEC-03 / METRICS-03 defect. Numbers were invented. | Real paper/independent-benchmark figures in the table above, explicitly labeled "literature estimate, not measured here" |
| **RTN as a quality target** | Round-to-nearest collapses at INT3/INT2 (PPL 25→10000+). Only useful as a lower-bound baseline. | GPTQ/AWQ/HQQ as the honest quality bar |
| **Per-stage product ratios / `len(dict)` as a ratio** | Project's BUG-02/COV-03 history: a method at 85% error reported 22.31×. | `honest_metrics.py` byte-exact `serialized_nbytes()` + paired error (already built — harden it) |
| **torch/torchao/optimum in core** | Violates pure-Python/no-torch constraint. | External subprocess oracles |

---

## Stack Patterns by Variant

**If the goal is "honest INT4 ratio vs real tools":**
- Measure the project's compressed size as `ratio_vs_disk` (BF16) — the default headline per METRICS-02.
- Produce the GGUF Q4_K_M reference (3.27× vs F16) externally and report the project's INT4 against it, labeled as literature/external.

**If the goal is "prove compression didn't hurt quality" (EVAL-01):**
- Add a WikiText2 perplexity eval. Run both original (safetensors/BF16) and project-compressed model through `lm_eval`.
- Acceptable: ΔPPL < 0.2 at INT4. Flag anything above as degraded.

**If building the RND activation-pruning track (APoZ/Wanda/SparseGPT):**
- Use calibration corpora (c4/wikitext, 128–256 samples, seqlen 512–2048) mirrored in pure-Python — load raw text, tokenize, build activation stats without torch.
- HQQ (no calibration) is the honest "lower bound" the pruning track should beat.

**If the goal is a portable on-disk format of the project's own compressed weights:**
- Keep SSF v2/v3 (already built) but **separate algorithmic ratio from container overhead** (per-tensor 4096-byte page alignment + headers) per FORMAT-01. Report both.
- Optionally add a GGUF *writer* so project outputs are consumable by llama.cpp — this is the most interoperable CPU format.

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| GPTQModel 7.0.0 | HF Transformers 5.3.0+ (2026-03) | Auto-defuses fused models via `Defuser`. CPU kernels added in 5.8.0. |
| AutoAWQ (dead) | Torch 2.6.0 / Transformers 4.51.3 (last tested) | Do not build against newer Transformers. |
| lm-eval-harness 0.4.9 | Python 3.10+; MMLU now `cais/mmlu` | C4 perplexity added in this release. |
| safetensors 0.8.0 | float8_e4m3fnuz / e5m2fnuz dtypes added | Page-aligned writes opened up. |
| torchao (CPU INT4) | PyTorch 2.8+, Intel Xeon w/ AMX | Needs `torch.compile` + max-autotune for speed. Not for pure-Python project. |

---

## Key Infrastructure the Project is Missing (roadmap implications)

1. **Perplexity eval harness** (EVAL-01) — the single biggest trust gap (PROJECT.md MISS-01). No eval baseline exists. This is non-negotiable for honest metrics.
2. **Calibration pipeline** — every real PTQ method uses 128–512 samples from c4/wikitext/pile. The project has none; build a torch-free version.
3. **Real comparison data** (METRICS-03) — the table above replaces `benchmark_industry_comparison.json`. Every external number must carry a "paper claim / independent benchmark / theoretical" label.
4. **Error-gated ratio reporting** (METRICS-01) — byte-exact ratio + paired error, already in `honest_metrics.py`; needs tests (METRICS-04).
5. **Format overhead separation** (FORMAT-01) — algorithmic ratio vs SSF container overhead must be reported separately.

---

## Sources

- ModelCloud/GPTQModel (GitHub, v7.0.0 / v5.8.0 release notes, 2026) — actively maintained GPTQ/AWQ successor; deprecation of AutoGPTQ/AutoAWQ. **Confidence: HIGH**
- Hugging Face Transformers quantization overview (v5.0.0rc0 / v4.57) — method/CPU/on-the-fly matrix; AutoGPTQ deprecation commit 8ebfd84 (2025-12-10). **Confidence: HIGH**
- GPTQ paper (Frantar et al., ICLR 2023, arXiv:2210.17323) — LLaMA/OPT PPL tables, INT4 g128 = 5.63 vs 5.47. **Confidence: HIGH**
- AWQ paper (Lin et al., MLSys 2024, arXiv:2306.00978) — INT4 g128 = 5.60 beats GPTQ/RTN; Mistral/Mixtral tables. **Confidence: HIGH**
- QuIP# / AQLM papers (IST-DASLab / Vahe1994) — sub-4-bit bpw results (2.02–2.29 bpw, 6.65–8.22 PPL). **Confidence: HIGH**
- HQQ (mobiusml/Dropbox, GitHub + dropbox.tech 2025-10) — calibration-free W4=5.62, W2=8.3, W1=14.7. **Confidence: MEDIUM** (community benchmark, not peer-reviewed paper)
- llama.cpp `tools/quantize` README (ggml-org, master) — GGUF bpw/size tables for Llama-3.1-8B. **Confidence: HIGH**
- "Which Quantization Should I Use?" (Kurt, arXiv:2601.14277, 2026-01) — unified llama.cpp quant eval. **Confidence: MEDIUM** (preprint)
- oobabooga blog GPTQ/AWQ/EXL2/llama.cpp perplexity comparison (Llama-2-13B). **Confidence: MEDIUM** (independent benchmark)
- safetensors (GitHub, v0.8.0) + HF docs — serialization, not compression. **Confidence: HIGH**
- lm-evaluation-harness (EleutherAI, v0.4.9, Zenodo 2025-06) — perplexity/C4/MMLU harness. **Confidence: HIGH**
- torchao (PyTorch blog, arXiv:2507.16099) + optimum-quanto (HF) — CPU INT4/INT8 infra. **Confidence: HIGH**

---
*Stack research for: LLM weight compression & CPU local inference (external ecosystem)*
*Researched: 2026-07-08 — all versions verified against 2025–2026 sources, not training data*
