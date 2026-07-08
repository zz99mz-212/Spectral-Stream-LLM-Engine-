# Architecture Research

**Domain:** LLM weight compression & local inference engine (architecture dimension)
**Researched:** 2026-07-08
**Confidence:** MEDIUM

> Scope note: This research does NOT re-examine the existing Spectral-Stream system. It documents how *successful* LLM compression engines are architected (GPTQModel/AutoGPTQ, llama.cpp/GGUF, ExLlamaV2, vLLM/llm-compressor, AWQ, SparseGPT/Wanda, lm-evaluation-harness) so the roadmap can compare and repair. Concrete numbers come from the field's published PTQ benchmarks; interpretations are mine.

---

## Standard Architecture

### System Overview (the canonical PTQ + inference stack)

Every production compression system separates **three independent subsystems** that Spectral-Stream currently fuses into one orchestrator pass:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         COMPRESSION TOOLCHAIN                          │
├──────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ Calibration  │  │  Quantizer / │  │    Format / Serializer    │   │
│  │  Data Loader │→│  Pruner Core  │→│  (GGUF / EXL2 / safetensors)│  │
│  │ (tok, batch) │  │ (Hessian, OBS)│  │   (weights + scales + meta)│  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
                                  │  (compressed artifact on disk)
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         INFERENCE ENGINE                               │
├──────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Model     │  │  Dequant │  │   Attention   │  │  Fused GEMM     │  │
│  │  Loader   │→│  Kernel   │→│  (KV cache)   │→│  (weight×act)    │  │
│  │ (mmap)    │  │ (on-fly) │  │              │  │  matmul_q_half  │  │
│  └──────────┘  └──────────┘  └──────────────┘  └─────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         EVALUATION HARNESS                             │
├──────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ Model Adapter │  │  Task / PPL  │  │   Recovery / Threshold    │   │
│  │ (request API) │→│  Runner       │→│   Gate (compressed/base)   │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

The critical architectural insight: **compression is an offline, stateful pass; inference is a separate online path that consumes a self-describing artifact; evaluation is a third independent path that compares compressed vs base through a uniform request interface.** Spectral-Stream's `CompressionIntelligenceEngine` does profile→allocate→select→compress→validate in one in-process flow and has no separate inference or eval subsystem.

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Calibration Loader** | Load raw text dataset, tokenize to fixed seq-len, batch to N samples, feed forward through model to collect layer inputs/activations | HF `datasets` + tokenizer; forward hooks per layer; caches `inps` per module (AutoAWQ `init_quant`, GPTQModel `CalibrationPipeline`) |
| **Quantizer / Pruner Core** | Per-layer (or sequential) optimization: compute Hessian/inverse, select mask, quantize to grid, compensate remaining weights | GPTQ column-wise greedy w/ Cholesky H⁻¹; AWQ activation-aware scaling; SparseGPT OBS update; Wanda saliency `|W|·‖X‖` |
| **Format / Serializer** | Persist quantized weights + scales + metadata so a different engine can reload them | GGUF (versioned binary, KV metadata), EXL2 (safetensors + q_weight/q_scale/q_groups), compressed-tensors (safetensors + quantization config) |
| **Model Loader** | Mmap/load artifact, reconstruct tensors, dispatch to devices, build module tree | llama.cpp `llama_model_load`, ExLlama `ExLlamaV2.load`, HF `from_pretrained` |
| **Dequant Kernel** | Reconstruct FP/FP16 weight from int/scales inside the matmul (AOT storage, on-the-fly reconstruction) | `gemm_half_q_half` (ExLlama), Marlin (FP16×INT4), GGML `ggml_mul_mat` |
| **Attention / KV** | Layer norms, RoPE, KV-cache paging, prefill/decode | engine-specific; KV-cache manager is a first-class component, not an afterthought |
| **Eval Adapter** | Uniform request API (`loglikelihood`, `generate_until`) so tasks don't care about compression | lm-eval-harness `lm_eval.models.*`, vLLM backend |
| **Task / PPL Runner** | Run perplexity (loglikelihood_rolling) + downstream tasks (YAML-defined) | lm-eval-harness tasks; llm-compressor recovery test |
| **Recovery Gate** | Compare compressed vs base metric; fail if recovery < threshold (default 0.95) | llm-compressor `_validate_recovery(compressed/base)` |

---

## Recommended Project Structure (target for Spectral-Stream)

```
src/
├── calibration/          # calibration data loading + activation capture
│   ├── loader.py         # dataset -> tokenized batches (C4/wikitext/alpaca)
│   ├── capture.py        # forward hooks to collect per-layer inputs
│   └── pipeline.py       # calibration pipeline orchestration (independent/sequential)
├── methods/              # individual, VALIDATED compression ops (not 2964 stubs)
│   ├── quant/            # int8_blockwise, int4_groupwise (only what works)
│   ├── prune/            # wanda, sparsegpt (RND track)
│   └── base.py           # CompressionMethod protocol: compress(tensor, calib)->artifact
├── formats/              # serialization, fully separated from compression math
│   ├── ssf.py            # existing SSF v2/v3 read/write
│   ├── gguf.py           # import/export (pure-python already exists)
│   └── safetensors.py    # import/export
├── engine/               # the ORCHESTRATOR (thin, registry-driven)
│   ├── intelligence.py   # profile→allocate→select→compress→validate (keep)
│   └── registry.py       # VALIDATED set vs experimental/ namespace (REGISTRY-01)
├── inference/            # SEPARATE subsystem (currently missing as a real path)
│   ├── loader.py         # load SSF/GGUF -> tensors
│   ├── dequant.py        # on-the-fly dequant kernels (numpy)
│   ├── model.py          # layer stack, KV cache
│   └── server.py         # OpenAI-compatible (exists as CLI only)
└── eval/                 # SEPARATE subsystem (currently MISSING entirely)
    ├── harness.py        # model adapter (loglikelihood interface)
    ├── perplexity.py     # wikitext2 PPL
    └── recovery.py       # compressed/base ratio gate
```

### Structure Rationale

- **`calibration/` separated from `methods/`:** In GPTQModel/AutoAWQ the calibration loader produces per-layer activation stats that *any* method (GPTQ, AWQ, Wanda) consumes. Spectral-Stream has no activation capture at all — its methods operate on weight tensors blind. Capturing activations is a **prerequisite** for every differentiator in the RND track (APoZ, Wanda, dynamic per-layer reduction).
- **`formats/` separated from compression math:** GGUF/EXL2/safetensors are self-describing on-disk formats. The compression algorithm (grid, scales) is independent of how you write bytes. Spectral-Stream's SSF couples container overhead (4096-byte pages) into the ratio — `FORMAT-01` correctly demands algorithmic ratio be separated from container overhead.
- **`inference/` is its own subsystem:** llama.cpp and ExLlama treat dequant as a kernel inside matmul, not a Python pre-pass. Spectral-Stream's `infer` CLI is a thin wrapper; building a real inference subsystem (loader → dequant → model → KV) is what moves it from "prototype" to "tool."
- **`eval/` is its own subsystem and is the #1 missing piece:** Zero perplexity evals exist today. This is the single largest trust gap (PROJECT.md: "No eval baseline"). lm-eval-style recovery gating is how every real project proves honesty.

---

## Architectural Patterns

### Pattern 1: Offline compression → self-describing artifact → separate online dequant

**What:** Compress once (slow, needs calibration + Hessian), write a portable artifact that carries weights, scales, quant-type enum, and metadata. At inference, the engine mmaps the artifact and reconstructs weights *inside the matmul kernel* (AOT storage, on-the-fly reconstruction).

**When to use:** Always, for usable tools. This is the llama.cpp / ExLlama / GPTQModel model.

**Trade-offs:** AOT compression is irreversible (you pay the cost once); but inference is fast and memory-cheap because weights stay 4-bit in RAM/VRAM. The alternative (dynamic/online quantization, bitsandbytes/torchao) skips the artifact but is slower and needs the base model present.

**Example (ExLlama/Marlin pattern):**
```python
# Artifact stores int4 weights + scales (AOT)
q_weights: uint32[k', n]   # packed quantized weights
q_scale:   uint32[g, n/8]  # 4-bit group scales
q_groups:  uint16[2g]      # (group_bits, group_size) per group
# Inference reconstructs in-kernel, never materializing full FP16 W:
out = gemm_half_q_half(hidden_fp16, q_handle)  # fused dequant + GEMM
```

### Pattern 2: Per-layer sequential quantization with Hessian (OBS compensation)

**What:** Process the model layer-by-layer (or true-sequential: sub-modules within a layer). For each linear layer: collect input activations X during calibration, build H = X·Xᵀ (+dampening), compute H⁻¹ via Cholesky. Greedily quantize/prune columns left→right, and after each column **propagate the reconstruction error to the remaining columns** using the Optimal Brain Surgeon formula. This is the GPTQ/SparseGPT shared core.

**When to use:** When you need <0.1 perplexity loss at INT4. The error-compensation step is *why* GPTQ beats naive round-to-nearest (RTN) — RTN alone gives 6.29 PPL at INT4 vs GPTQ's 5.85 on LLaMA-7B.

**Trade-offs:** Computationally heavy (needs full forward passes for calibration + H⁻¹ per layer), but it's *offline*. The compensation is what makes INT4 actually usable. Spectral-Stream's current INT8 path does NOT do this — it uses blockwise RTN-equivalent scaling, which is why it's limited to ~4.6× safely.

**Example (GPTQ column loop, simplified):**
```python
H = torch.linalg.cholesky(Hinv); H = cholesky_inverse(H)
for i1 in range(0, cols, blocksize):
    W1 = W[:, i1:i2]
    Q1 = quantize(W1, scale, zero)              # round to grid
    Err = W1 - dequant(Q1)
    W[:, i2:] -= Err @ Hinv[i1:i2, i2:]         # OBS compensation to remaining cols
    W[:, i1:i2] = dequant(Q1)
```

### Pattern 3: Hybrid prune-then-quantize in a single pass

**What:** SparseGPT shows pruning and quantization can share the GPTQ column-wise framework. The correct order is **prune first (on dense BF16, where saliency `|W|·‖X‖` is meaningful), then quantize the survivors.** Pruning and quantizing jointly (in the same pass) lets later decisions see earlier rounding. Doing AWQ-then-prune on already-quantized weights is worse because the prune criterion is calibrated for BF16.

**When to use:** When chasing max compression (e.g. 70B BF16 140GB → 2:4 sparse 70GB → AWQ INT4 ~17GB, ~8× total).

**Trade-offs:** Additive quality loss (~5% PPL from prune + ~5% from quant ≈ 10% total vs dense). For Spectral-Stream's RND track (APoZ/Wanda/SparseGPT), the ordering rule is the load-bearing architectural decision: **never prune post-quantization.**

### Pattern 4: Uniform request-type eval abstraction

**What:** lm-evaluation-harness isolates model implementation from task logic via four primitive request types: `loglikelihood`, `loglikelihood_rolling`, `generate_until`, `multiple_choice`. Perplexity = `loglikelihood_rolling` (aggregate token logprobs). Tasks are declarative YAML.

**When to use:** Mandatory if you want credible, comparable evals. Spectral-Stream currently has none.

**Trade-offs:** Some boilerplate, but it means a compressed model, a base model, and an API model are all evaluated identically — enabling **recovery testing**: `recovery = compressed_val / base_val`, gate at ≥0.95. This is the exact honesty mechanism PROJECT.md wants (METRICS-01 error-gating, EVAL-01 real downstream eval).

---

## Data Flow

### Compression Flow (offline, stateful)

```
Raw text (C4/wikitext/alpaca)
   ↓ tokenizer(..., max_length=seq_len)
Tokenized batches (128-256 samples × 512-2048 tokens)
   ↓ forward pass through BASE model, capture per-layer inputs via hooks
Per-layer activation stats (X per module)  ──┐
   ↓                                         │
Quantizer/Pruner core  ←─────────────────────┘  (consumes X + H=X·Xᵀ)
   ↓ per-layer: quantize/prune + OBS compensate
Compressed tensors (int codes + scales + group metadata)
   ↓
Format serializer (GGUF/EXL2/SSF)  →  self-describing artifact on disk
```

### Inference Flow (online, stateless w.r.t. compression)

```
Artifact on disk (GGUF/SSF)
   ↓ mmap + parse header (quant-type enum, per-tensor scales/offsets)
Model Loader reconstructs module tree (meta device, lazy)
   ↓
For each token:
   hidden = layer_norm(hidden)
   q,k,v  = dequant_kernel(q_proj, hidden)   # fused reconstruct + matmul
   attn   = softmax(q·kᵀ/√d) · v  + KV cache
   hidden = dequant_kernel(o_proj, attn)
   ↓
logits → sample → next token
```

### Evaluation Flow (independent path)

```
Base model  ──┐
             ├─► Eval Adapter (loglikelihood_rolling) ─► base_ppl / base_acc
Compressed  ─┘
model       ──► Eval Adapter (same interface)        ─► comp_ppl / comp_acc
             ↓
        Recovery Gate: comp/base ≥ 0.95 ?  PASS / FAIL
```

### Key Data Flows

1. **Calibration → method:** Activation stats (X, Hessian) flow from calibration subsystem into *any* method. Spectral-Stream has no such channel today — methods receive only weight tensors. **This is the structural gap blocking the entire RND track.**
2. **Compression → format:** Compressed tensors + scales flow into a serializer that knows nothing about the algorithm. Enables artifact portability and honest ratio measurement (algorithmic vs container).
3. **Format → inference:** The artifact is the contract between compression and inference. If the format is self-describing (quant-type enum + scales), inference needs zero knowledge of how compression ran.
4. **Compressed vs Base → eval:** Both models expose the same request interface; recovery ratio is computed by an independent gate. This is the honesty backbone.

---

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Research (1 model, CPU, NumPy) | Monolith is fine, but the **three-subsystem split** (compress / infer / eval) matters more than perf. Spectral-Stream is here. |
| Larger models (single GPU) | Calibration pipeline needs `sequential` offload (llm-compressor) so activations flow layer-by-layer without holding the whole model. CPU-only research can't reach this. |
| Multi-GPU / serving | Tensor parallelism, fused kernels (Marlin), KV paging. **Out of scope for Spectral-Stream (PROJECT.md: no GPU, no serving).** Do not architect for this. |

### Scaling Priorities (for THIS project)

1. **First bottleneck:** No calibration/activation capture → RND track (Wanda/SparseGPT/dynamic reduction) cannot start. Build `calibration/` first.
2. **Second bottleneck:** No eval subsystem → every ratio claim is unverifiable. Build `eval/` (perplexity + recovery gate) before adding more methods.
3. **Third:** SSF container overhead coupling into ratio (FORMAT-01). Separate algorithmic ratio from 4096-byte page padding.

---

## Anti-Patterns

### Anti-Pattern 1: Fusing compression + eval + inference into one orchestrator

**What people do:** One `engine.run(model)` does profile, compress, measure error, and "infer" inline, as Spectral-Stream's `CompressionIntelligenceEngine` does.

**Why it's wrong:** Compression is offline and stateful; inference is online and latency-sensitive; eval must be *independent* to be credible (you cannot grade your own exam). Coupling them hides the eval-trust gap and prevents the artifact from being reused by a real engine.

**Do this instead:** Three subsystems with the artifact (format) as the only contract between them. The orchestrator picks methods; it does not also serve tokens or grade quality.

### Anti-Pattern 2: Measuring quality by reconstruction error only (no perplexity)

**What people do:** Report `rel_mse` / `snr_db` on weight tensors and call it "compression works."

**Why it's wrong:** Weight reconstruction error is a proxy, not the truth. A method with low weight MSE can still wreck language modeling; conversely small weight changes can cascade. The field's gold standard is **perplexity delta** and **downstream task recovery**.

**Do this instead:** lm-eval-style recovery gate — measure WikiText-2 PPL on the *actual* compressed model vs base, gate at ≥0.95. (PROJECT.md EVAL-01 / METRICS-01.)

### Anti-Pattern 3: Pruning after quantization

**What people do:** Quantize to INT4, then apply Wanda/sparsity on the quantized weights.

**Why it's wrong:** The prune saliency criterion `|W|·‖X‖` and OBS compensation assume BF16-precision weights. Quantized weights give the prune step corrupted input; quality degrades measurably.

**Do this instead:** Prune on dense weights first, then quantize survivors (SparseGPT joint-pass is the ideal). Encode this ordering as an invariant in the pipeline.

### Anti-Pattern 4: Catalog of stubs presented as capability

**What people do:** 2,964 registered methods, only 2 live; advertised ratios imply the catalog works.

**Why it's wrong:** Misrepresents maturity; erodes trust; couples roadmap to unvalidated code.

**Do this instead:** `REGISTRY-01` — split into validated active set (tests + real-weight results) vs labeled `experimental/` namespace. The orchestrator only routes to validated methods by default.

### Anti-Pattern 5: Coupling container overhead into compression ratio

**What people do:** SSF 4096-byte page alignment + header/footer inflate per-tensor cost; reporting `serialized_nbytes / fp32_nbytes` hides this.

**Why it's wrong:** Small tensors pay enormous fixed overhead; the headline ratio is not the algorithmic ratio.

**Do this instead:** `FORMAT-01` — report algorithmic ratio (compressed payload / raw tensor) separate from container overhead; pack small tensors into shared pages.

---

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| HuggingFace `datasets` | Calibration data source (C4, wikitext, etc.) | Streaming to avoid OOM; 128-256 samples sufficient |
| HuggingFace `transformers` | Base model + tokenizer for calibration | Needed to collect activations (CPU-only → small models) |
| lm-evaluation-harness | Eval backend (optional import) | Reuse its task YAMLs; only need `loglikelihood_rolling` adapter |
| GGUF/GGML (llama.cpp) | Export target / import source | Pure-Python GGUF dequant already exists in Spectral-Stream |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `calibration/` ↔ `methods/` | Activation stats object (X, H⁻¹ per layer) | The missing channel; must be defined as a protocol |
| `methods/` ↔ `formats/` | CompressedTensor(weights, scales, quant_type, meta) | Algorithm-agnostic; format writes whatever it's given |
| `formats/` ↔ `inference/` | On-disk artifact (mmap) | Contract = the format spec; inference trusts the enum |
| `eval/` ↔ (`inference/` or external model) | Uniform request interface | Eval never imports compression internals |

---

## Comparison: Spectral-Stream vs Standard Architecture

### What Spectral-Stream gets RIGHT

| Aspect | Assessment |
|--------|------------|
| Registry-driven orchestrator (`profile→allocate→select→compress→validate`) | Correct shape. Matches the field's "method registry + driver" pattern (GPTQModel recipes, llm-compressor modifiers). Keep it. |
| SSF binary format with zstd + read/write/index | A real self-describing format is exactly what the standard demands. The *coupling of container overhead into ratio* is the only flaw (FORMAT-01). |
| Honest-metrics infrastructure (`serialized_nbytes`, `dual_ratio`, `ErrorMetrics`) | Byte-exact ratio + error coupling is the right instinct — this is the recovery-gate idea, pre-formalized. Needs hardening (METRICS-04) and a real eval behind it (EVAL-01). |
| INT8 blockwise verified path (~4.6× vs FP32) | Honest, works. Equivalent to the field's RTN/blockwise baseline (which is *worse* than GPTQ's INT4 — see below). |
| KVCacheManager (30+ policies) | KV-cache as a first-class component matches the standard (llama.cpp/ExLlama treat KV as core). Good. |
| Pure-Python GGUF dequantizer | Real portability win; lets the engine consume external artifacts. |

### What's MISSING or WRONG

| Gap | Standard expects | Spectral-Stream today | Roadmap implication |
|-----|------------------|----------------------|---------------------|
| **Calibration / activation capture** | `calibration/` subsystem feeding per-layer X/H⁻¹ to methods | None. Methods see only weight tensors | **Blocker for RND track.** Build `calibration/` first. |
| **OBS / Hessian compensation** | Per-layer column-wise greedy + error propagation (GPTQ core) | INT8 blockwise = RTN-equivalent (no compensation) | Explainable why INT4 cascade fails: no compensation → 72-92% error. CASCADE-01. |
| **Separate inference subsystem** | `inference/` (loader → dequant kernel → model → KV) | `infer` CLI is thin wrapper, not a subsystem | Promote to first-class; dequant as in-kernel step, not Python pre-pass. |
| **Eval subsystem** | `eval/` with perplexity + recovery gate | Zero perplexity evals | **#1 trust gap (EVAL-01).** Build before more methods. |
| **Method/format separation** | Algorithm independent of serializer | SSF overhead bleeds into ratio | FORMAT-01: split algorithmic vs container ratio. |
| **Validated vs experimental split** | Recipes/modifiers are explicit, versioned | 2,964 stubs, 2 live, advertised as catalog | REGISTRY-01/02: split + label maturity. |
| **Prune-before-quant invariant** | Joint or prune-first, never quant-then-prune | No pruning wired; cascade order undefined | Encode ordering as pipeline invariant (Anti-Pattern 3). |

### The core diagnosis in architectural terms

The 5-stage cascade's 72-92% reconstruction error is **not** a math bug — it's a missing **calibration → OBS compensation** channel. Standard INT4 works (GPTQ ~5.85 PPL on LLaMA-7B, AQLM INT3 ~5.46) *because* it collects activations and propagates quantization error column-by-column. Spectral-Stream's stages operate on weight tensors with no activation context and no error compensation, so residual error compounds across the 5 stages instead of being absorbed. The fix is architectural (add calibration + compensation), not a tuning of stage parameters.

---

## Build Order Implications (for roadmap phasing)

Dependencies drive the order:

1. **Phase A — Calibration substrate (`calibration/` + activation capture).** Everything in the RND track (APoZ, Wanda, SparseGPT, dynamic reduction) and any INT4 improvement depends on it. No calibration → no differentiators.
2. **Phase B — Eval subsystem (`eval/` + perplexity + recovery gate).** Must land early (even before more methods) because it is the only way to *prove* honesty and to validate Phases C/D. This is also PROJECT.md's biggest gap (EVAL-01).
3. **Phase C — INT4 with OBS compensation (upgrade the verified path).** Use calibration from A + recovery gate from B to build a real INT4 (groupwise) method that matches the field's ~4× vs FP16 / <0.1 PPL. Fixes CASCADE-01 honestly.
4. **Phase D — Registry/format hygiene (`REGISTRY-01/02`, `FORMAT-01`).** Split validated vs experimental; decouple container overhead. Lower risk, can parallelize with C.
5. **Phase E — Pruning track (Wanda/SparseGPT), prune-first invariant.** Depends on A (activations). Build only after calibration is solid.
6. **Phase F — Inference subsystem promotion.** Promote `infer` to a real `inference/` path with dequant kernel + KV model. Depends on format contract being stable (D).

Phases A and B are the **critical path** — neither the differentiators nor the honesty claims can proceed without them.

---

## Sources

- GPTQModel (ModelCloud) README/releases — actively maintained AutoGPTQ successor, v7.0.0 (2026-04), CPU kernels, multi-backend. [web, LOW]
- HF Transformers quantization overview — AutoGPTQ/AutoAWQ deprecated, GPTQModel supplants; on-the-fly vs AOT matrix; bits support. [web, LOW]
- AutoAWQ `awq/quantize/quantizer.py` — `AwqQuantizer` calibration pipeline (`init_quant`, `get_calib_dataset`), `max_calib_samples=128`, `max_calib_seq_len=512`. [web, LOW]
- GPTQ calibration conventions (Orchestra-Research, arxiv 2311.09755) — 128×2048=262k tokens; C4/RedPajama/Pile; diminishing returns. [web, LOW]
- GGUF format spec (ggml-org/ggml/docs/gguf.md) — header, KV metadata, tensor info, ggml_type quant enum (~40 types), versioning. [websearch, MEDIUM]
- Safetensors docs (huggingface.co/docs/safetensors) — safe zero-copy serialization, NOT compression; JSON header + raw bytes. [websearch, MEDIUM]
- ExLlamaV2 (DeepWiki, linear.py, issue #494) — `ExLlamaV2Linear`, EXL2 format (q_weight/q_scale/q_groups), `gemm_half_q_half` fused kernel, safetensors container. [web, LOW]
- Marlin / Sparse-Marlin (IST-DASLab) — FP16×INT4 ~4× kernel; 2:4 sparse ~5.3×; used by GPTQModel/vLLM. [web, LOW]
- SparseGPT (IST-DASLab/sparsegpt, arxiv 2301.00774) — OBS pruning, joint prune+quantize in GPTQ column framework, Cholesky H⁻¹. [web, LOW]
- Wanda (locuslab/wanda) — `|W|·‖X‖` saliency, prune-first-then-quantize ordering. [web, LOW]
- Spheron GPU pruning guide (2026) — prune-first then quantize; additive ~10% PPL loss at 8×. [web, LOW]
- lm-evaluation-harness (EleutherAI) — request-type abstraction (loglikelihood/loglikelihood_rolling/generate_until), YAML tasks, perplexity via loglikelihood_rolling. [websearch, MEDIUM]
- llm-compressor (neuralmagic/vllm) — `oneshot` recipe/modifier pipeline, calibration pipelines (independent/sequential/datafree), recovery testing `compressed/base ≥ 0.95`. [web, LOW]
- PTQ benchmark tables (qwopqwop200/GPTQ-for-LLaMa, AQLM/QuIP# papers, model-quantization-lab) — INT4 ~3.4-4× vs FP16 with <0.1 Wiki2 PPL; group quant > algorithm name. [web, LOW]
- torchao (pytorch/ao) — Int4WeightOnlyConfig groupwise, CPU/ARM kernels, dynamic vs weight-only. [web, LOW]

---

*Architecture research for: LLM weight compression & local inference (architecture dimension)*
*Researched: 2026-07-08 — comparison target: Spectral-Stream LLM Engine (brownfield re-init)*
