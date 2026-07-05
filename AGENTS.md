# SpectralStream — Agent Instructions

## Auto-Setup
At session start, run `/index` to build/refresh the intelligence codebase index. The intelligence engine auto-detects `.venv/` and uses tree-sitter AST for 10 languages. Config: 1M context window, `prune=true`, `reserved=100K`.

## Project
Pure-Python LLM inference engine (Python 3.10+) using hyperdimensional computing, spectral/DCT methods, Vlasov mean-field attention, and quantum-inspired tensor networks. All SIMD via NumPy vectorized operations — no C++ extensions.

## Code-Graph Tooling

**ALWAYS run `python -m intelligence.cli <command>` BEFORE writing new code** to check for existing functions, classes, imports, and avoid duplication. All subagents MUST use this tool.

### Available Commands

| Command | Description |
|---------|-------------|
| `index` | Rebuild index after adding/renaming files |
| `stats` | Show index statistics (entities, relationships) |
| `search <term>` | Find functions/classes/files by name |
| `snippet <name>` | Show source code with context lines |
| `suggest <name>` | Auto-import suggestions for a symbol |
| `functions [module]` | List functions in a module |
| `classes [module]` | List classes in a module |
| `imports` | Full import dependency graph |
| `callers <func>` | Find who calls a function |
| `chains <func>` | Call chain analysis (reverse DFS) |
| `related <name>` | Show connected entities |
| `context <query>` | Build context subgraph for a topic |
| `dead` | Dead code analysis (registry-aware, filters dispatched functions) |
| `quality` | Show quality scores for all modules |
| `cycles` | Detect import cycles |
| `todos` | Find TODOs/FIXMEs |
| `test-gaps` | List modules without test files |
| `entrypoints` | List CLI entrypoints with import counts |
| `complexity` | McCabe complexity hotspots |
| `hot-paths` | Most-called functions across the codebase |
| `xboundary` | Cross-package call analysis |
| `api` | Public/private ratio per module |
| `unused-exports` | Public names never imported externally |
| `import-groups` | Files with overlapping import patterns |
| `impact <name>` | Dependency impact — what breaks if X is deleted |
| `cluster <name>` | Group callers of a function by package |
| `insight` | Full repo insight report (all above combined) |
| `mermaid` | Generate Mermaid JS dependency graph |

### Quick Reference
```bash
# Before writing new code — check if it exists
python -m intelligence.cli search "my_function_name"
python -m intelligence.cli suggest "BlockINT8"
python -m intelligence.cli snippet "compress_tensor"

# Understand call relationships
python -m intelligence.cli callers "compress_tensor_with_validation"
python -m intelligence.cli chains "apply_compression"
python -m intelligence.cli related "CompressionIntelligenceEngine"

# Find gaps
python -m intelligence.cli test-gaps
python -m intelligence.cli dead | head -30
python -m intelligence.cli quality | head -20

# Architecture understanding
python -m intelligence.cli insight
python -m intelligence.cli hot-paths | head -10
python -m intelligence.cli complexity | head -10
python -m intelligence.cli xboundary
python -m intelligence.cli import-groups | head -10
```

## Commands
- **Run tests:** `python -m pytest tests/ -v --tb=short -x --timeout=120`
- **Run single test:** `python -m pytest tests/test_compression_engine.py -v --tb=short -x --timeout=120`
- **Run e2e test:** `python scripts/e2e_test.py`
- **Run e2e validation:** `python scripts/e2e_validation.py`
- **Run e2e validation (custom):** `python scripts/e2e_validation.py --num-layers 8 --target-ratio 5000 --output-dir /tmp/e2e`
- **Generate docs:** `python -m spectralstream.compression.cli list-methods --verbose`
- **Generate certificates:** `python -m spectralstream.compression.cli compress model.safetensors out.ssf --certificate --format all`
- **Run CLI:** `python -m spectralstream.compression.cli <command> [options]`
- **Activate venv:** `. .venv/bin/activate` (always do this first)

## Package Layout
```
spectralstream/
  core/math_primitives/  — 18 math submodules (prng, fft, transforms, spectral, numerical, etc.)
  compression/           — intelligence engine with 80+ methods (9 categories + novel + tensor network)
    advanced/            — v1 archive reintegration: turboquant_codec, sparsity_engine, tt_pq_engine,
                           quantum_tensor_net, hyper_compression_v2, advanced_sparsity
    cutting_edge.py      — 25 novel compression methods (v1 archive reintegration)
    novel_compression_library.py — 70+ methods across 6 categories (v1 archive reintegration)
    physics_compression.py      — Physics-inspired compression (Hamiltonian, Topological, StateSpace)
    noise_aware_compressor.py   — Noise-adaptive compression with floor detection
    unified_compression_pipeline.py — End-to-end compression pipeline (SSCX format)
    unified_quantizer.py — 5-stage quantizer (DCT+TT+VBQ+entropy+quality tables)
    engine/              — orchestrator (profile → allocate → select → compress → validate)
    methods/             — 80+ method implementations across 9+2 categories
      decomposition/     — SVD, Tensor Train, CP, Tucker, Kronecker, Butterfly, Monarch, etc.
      spectral/          — DCT2D, FWHT, Wavelets, Fourier, NTT, Givens, Chebyshev, Winograd, etc.
      structural/        — Einsort, Monarch, BlockSparse, Circulant, LowRank, NMF, etc.
      entropy/           — Arithmetic Coding, ANS, Huffman, Range Coding
      functional/        — MLPMixer, FNet, Performer, LinearAttention
      physics/           — Ising, QuantumCircuit, TensorNetwork (MPS), Renormalization, MERA, PEPS
      quantization/      — BlockINT8/4, HadamardINT8/4, DeltaINT4, SparsityINT4, Uniform, NF4, etc.
      lossless/          — Zstd, RANS, LZ4
      hybrid/            — Spectral+Quant, DCT+Entropy, combined methods
      novel/             — QuantumPlasmaFusion, HDCCompression, HolographicReducedRank, etc.
    registry/            — CompressionMethod enum + MethodRegistry
    certificate.py       — Professional certificates (CompressionCertificate, ValidationCertificate)
    cli.py               — unified CLI (compress, profile, list-methods, validate, benchmark, generate, verify, convert, info)
  format/                — SSF v2/v3 binary format (reader, writer, header, index, core, compression)
  inference/             — CPU inference engine + pipeline + benchmark
  kv_cache/              — unified KV cache (core, manager, eviction, compressor)
  model/                 — gemma4_config
  config.py              — SpectralStreamConfig dataclass (SS_ env prefix)
scripts/
  e2e_test.py            — Legacy end-to-end test
  e2e_validation.py      — Full validation pipeline with cert generation and threshold enforcement
  run_benchmark.py       — System benchmark script
  compress_gemma4.py     — Gemma 4 compression helper
```

## Import Pattern
All subpackages use relative/absolute imports that work correctly:
```python
from spectralstream.compression.engine import CompressionIntelligenceEngine
from spectralstream.format.reader import SSFReader
from spectralstream.inference.pipeline import InferencePipeline
from spectralstream.core.math_primitives import dct, fwht, softmax
from spectralstream.compression.certificate import (
    CompressionCertificate, ValidationCertificate, CertificateBuilder,
)
```
Top-level `__init__.py` is minimal (just `__version__`) to avoid circular imports.

## Backward Compatibility
Archived modules in `_archive/v1/` have been re-integrated as compat stubs:
- `spectralstream/unified_core.py` → re-exports from `core.math_primitives`
- `spectralstream/gemma4_config.py` → re-exports from `model.gemma4_config`
- `spectralstream/unified_attention.py` → re-exports from `attention.unified_attention` (archive copy)
- `spectralstream/sscx_format.py`, `rans_entropy.py`, etc.

## Key Source Files
- `spectralstream/core/math_primitives/` — 18 submodules with all math primitives
- `spectralstream/compression/engine/_orchestrator.py` — CompressionIntelligenceEngine (1095 lines)
- `spectralstream/compression/engine/_methods.py` — core compression methods (BlockINT8/4, HadamardINT8/4, etc.)
- `spectralstream/compression/engine/method_tiers.py` — Tier assignment (Tier 1-5) by category
- `spectralstream/compression/engine/method_discovery.py` — Auto-discovers all 80+ methods
- `spectralstream/compression/certificate.py` — Certificate generation (CompressionCertificate, ValidationCertificate)
- `spectralstream/compression/cutting_edge.py` — 25 novel compression methods (v1 archive reintegration)
- `spectralstream/compression/novel_compression_library.py` — 70+ methods across 6 categories (v1 archive reintegration)
- `spectralstream/compression/physics_compression.py` — Physics-inspired compression (v1 archive reintegration)
- `spectralstream/compression/noise_aware_compressor.py` — Noise-adaptive compressor (v1 archive reintegration)
- `spectralstream/compression/unified_compression_pipeline.py` — End-to-end pipeline (v1 archive reintegration)
- `spectralstream/compression/unified_quantizer.py` — 5-stage quantizer (v1 archive reintegration)
- `spectralstream/compression/advanced/turboquant_codec.py` — PolarQuant+QJL (v1 archive reintegration)
- `spectralstream/compression/advanced/sparsity_engine.py` — Advanced pruners (v1 archive reintegration)
- `spectralstream/compression/advanced/tt_pq_engine.py` — TT+PQ pipeline (v1 archive reintegration)
- `spectralstream/compression/advanced/quantum_tensor_net.py` — Quantum-inspired tensor networks (v1 archive reintegration)
- `spectralstream/compression/advanced/hyper_compression_v2.py` — FrequencyDomain/TT/VQ (v1 archive reintegration)
- `spectralstream/compression/advanced/advanced_sparsity.py` — SparseGPT/structured sparsity (v1 archive reintegration)
- `spectralstream/inference/pipeline.py` — InferencePipeline (558 lines)
- `spectralstream/format/reader.py` — SSFReader
- `spectralstream/format/writer.py` — SSFWriter
- `spectralstream/compression/cli.py` — Unified CLI with 9 commands

## Testing
- `pytest` with `-x` fail-fast, `--timeout=120`
- 223 core tests pass, 216 skipped (archive modules), 5 warnings
- `tests/conftest.py` controls skip markers for archive-module tests
- Core test files: `test_compression_engine.py`, `test_comprehensive.py`, `test_complete_system.py`
- Skipped until migration: `test_rans_hadamard.py`, `test_sscx_format.py`, `test_supreme_quant_engine.py`, `test_finetune.py`, `test_audit.py`, `test_pipeline.py`
- Archive reintegration tests: `test_archive_combined_pipeline.py`, `test_archive_extreme_compression.py`, `test_archive_intelligence_engine.py`, `test_archive_architecture_compressor.py`, `test_archive_extreme_compressor.py`

## CLI Commands
```
compress      Compress safetensors → SSF (supports --certificate, --format, --quick, --streaming)
profile       Profile model tensors with sensitivity analysis
list-methods  List compression methods (filter by --category, --tier, --verbose)
validate      Verify SSF file integrity (supports --original-model for comparison, --format for certs)
benchmark     Benchmark compression methods (supports --synthetic, --output)
generate      Generate text from SSF model (supports --temperature, --top-k, --top-p)
verify        Test compression methods on tensors (--all-methods)
convert       Alias for compress with defaults
info          Show SSF file metadata (--json for machine-readable)
```

## E2E Validation
```bash
# Full pipeline: synthetic model → compress → validate → report
python scripts/e2e_validation.py

# Options:
#   --model PATH            Use existing safetensors instead of synthetic
#   --num-layers N          Layers for synthetic model (default: 4)
#   --target-ratio R        Target compression ratio (default: 5000)
#   --max-error E           Max relative error (default: 0.0002)
#   --output-dir DIR        Output directory (default: /tmp/spectralstream_validation)
#   --max-error-threshold E Exit threshold for avg error (default: 0.01 = 1%)
#   --min-ratio-threshold R Exit threshold for ratio (default: 100:1)

# Exit code 0 = all thresholds met, 1 = threshold breach
# Reports saved to timestamped subdirectory: validation_report_{timestamp}/
```

## Document Generation
```bash
# List all methods with categories, tiers, and descriptions
python -m spectralstream.compression.cli list-methods --verbose

# Generate compression certificate (JSON, HTML, MD, TXT)
python -m spectralstream.compression.cli compress model.safetensors out.ssf --certificate --format all

# Generate validation certificate
python -m spectralstream.compression.cli validate out.ssf --original-model model.safetensors --format all

# Generate profile report
python -m spectralstream.compression.cli profile model.safetensors --report

# Generate benchmark report
python -m spectralstream.compression.cli benchmark model.safetensors --output benchmark.json --report
```

## Conventions
- Type hints required for all function signatures
- NumPy docstrings for public APIs
- NumPy vectorized ops over Python loops
- `SpectralStreamConfig` dataclass for config (env vars `SS_*` override file config)

## Anti-Duplication
Before writing any new function, class, or module:
1. Use `code-graph` tool with `search <term>` to check if something similar exists
2. Use `code-graph` tool with `functions [module]` or `classes [module]` to list definitions in a package
3. Use `code-graph` tool with `imports` to see the dependency graph
4. Check `__init__.py` and `method_registry.py` before adding new exports or compression methods
5. Use LSP tool's `findReferences` and `goToDefinition` to verify existing callers and implementations
6. Never rewrite existing functionality — extend or import it

## Intelligence Engine (`intelligence/`)
Python codebase graph engine (inspired by Saguaro Intelligence). 32K+ entities indexed from the live codebase.
- **`/search <term>`** — find existing functions/classes before writing new code
- **`/dead`** — dead code analysis: dead functions, classes, files, unused imports, duplicates
- **`/index`** — re-index after adding/renaming files

## Available Tools
- **code-graph** — query Python code structure: `search <term>`, `functions [module]`, `classes [module]`, `imports`, `callers <func>`, `related <name>`, `dead`, `context <query>`, `stats`, `quality`, `cycles`, `todos`, `test-gaps`. ALWAYS use before writing new code.
- **LSP tool** (experimental, enable via `OPENCODE_EXPERIMENTAL_LSP_TOOL=true`) — goToDefinition, findReferences, workspaceSymbol, documentSymbol, call hierarchy.

## Intelligence Architecture
- **Entity system**: Everything is an entity (files, functions, classes, imports, methods, symbols). Unified type system.
- **AST-analyzed call graph**: Detects bare calls, `self.method()`, `cls.method()`, `ClassName.static()`, and `ClassName()` → `__init__`. No heuristics — purely graph-based dead code analysis.
- **N-gram search index**: Fuzzy name + path search across all 32K entities.
- **Concurrent indexer**: ThreadPoolExecutor batch processing with shared-mutex state.
