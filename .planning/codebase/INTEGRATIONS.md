# External Integrations

**Analysis Date:** 2026-07-07

This is a CPU-targeted, research-grade mathematics/compression codebase. The **core engine performs no network I/O**: all model weights, caches, and results are read/written to the **local filesystem**. External integrations exist only as optional bridges to local model-serving tools and as standalone config templates. There are **no cloud services, no managed databases, no message queues, no webhooks, no CI/CD, no telemetry**.

## APIs & External Services

**Local model-serving bridges (not cloud):**
- **LM Studio** — `spectralstream/serving/lmstudio.py` connects to a *local* LM Studio OpenAI-compatible server using the Python **standard library** `http.client.HTTPConnection` (no `requests`/`openai` dependency). Default target `127.0.0.1:1234` (`spectralstream/config.py:73`). It also scans local disk for `*.gguf` files under `~/.lmstudio/models/...` (`lmstudio.py:14`).
- **vLLM** — `configs/vllm.json` is a launch-config template for a local vLLM server (`host: 0.0.0.0`, `port: 8080`, `gpu_memory_utilization: 0.0`, `served_model_name: spectralstream`). No vLLM library is imported in code; it is a target the engine can sit behind.
- **OpenAI-compatible API** — `configs/openai_sdk.py` is a *sample client* (uses the `openai` package, which is **not declared** as a dependency) pointing at `http://localhost:8080/v1`. The FastAPI server itself (`spectralstream/serving/api/`) exposes an OpenAI-style chat/completion interface for local consumption.
- **Ollama** — `configs/ollama.Modelfile` is an Ollama model definition (`FROM spectralstream`, `PARAMETER num_ctx 4096`) to wrap the compressed model as an Ollama model.
- **Claude Desktop / Codex / OpenCode** — `configs/claude_desktop.json`, `configs/codex.json`, `configs/opencode.json` are MCP/agent integration config stubs pointing at the local SpectralStream server. These are configuration artifacts, not code integrations.

**LLM provider / cloud API:** None. No Anthropic, OpenAI, or other hosted SDK is wired into the engine. The only `openai` reference is the standalone sample in `configs/openai_sdk.py`.

**Network calls in code:** Only `spectralstream/utils/multimodal_prompt.py` optionally imports `requests` (guarded by `HAS_REQUESTS`, `:68`) for an optional multi-modal fetch feature; this path is **disabled by default** and the dependency is undeclared. LM Studio bridging uses stdlib `http.client`, not external SDKs.

## Data Storage

**Databases:**
- **None.** No SQL, NoSQL, or embedded DB (no SQLite, no DuckDB, no Redis client) is used.

**File Storage:**
- **Local filesystem only.** Primary formats:
  - `.safetensors` — model weights (`safetensors.safe_open`), loaded from local files (`scripts/compress_gemma4.py`, `spectralstream/compression/world_model/dial_in_engine.py`).
  - `.gguf` — alternative model format, parsed via `gguf.GGUFReader` (`spectralstream/utils/tokenizer_engine.py`, `spectralstream/benchmark/benchmark_suite.py`, several `tests/benchmark_*.py`).
  - `.ssf` (SpectralStream Serialized Format) — custom compressed output using `zstandard` (`spectralstream/format/compression.py`, `spectralstream/format/ssf_format_pipeline.py`).
  - `.json` — configs, compression certificates, benchmark results, audit logs (e.g. `tests/*.json`, `tests/output/*.json`, root `*.json` logs).
  - State/checkpoints — `~/.spectralstream/state/` (`spectralstream/config.py:90`), auto-saved/loaded checkpoints.
- `models/` directory is git-ignored (`.gitignore`).

**Caching:**
- **None** (no Redis, no in-memory cache server). `psutil` is used for *monitoring*, not caching.

## Authentication & Identity

**Auth Provider:**
- **Custom / optional, local only.** The FastAPI server (`spectralstream/serving/api/_spectralstreamserver.py`, `spectralstream/serving/unified_server.py`) uses `fastapi.security.HTTPBearer` + `HTTPAuthorizationCredentials` bearer-token middleware, but it is optional and operates on a local token (`configs/lm_studio.json` uses `"api_key": "not-needed"`). No external identity provider (OAuth/OIDC/Auth0) is integrated.

## Monitoring & Observability

**Error Tracking:**
- **None.** No Sentry, no Rollbar, no telemetry SDK.

**Metrics:**
- A **self-contained Prometheus text-exposition exporter** exists (no `prometheus_client` dependency): `spectralstream/orchestrator.py:811` `get_metrics_prometheus()` and `spectralstream/serving/production_stack.py:729/772/1956` `to_prometheus()`. It emits Prometheus-format plain text on demand; a `MonitoringConfig.enable_prometheus` flag exists (`spectralstream/config.py:82`, default `False`) but no scraper is configured.

**Logs:**
- Python `logging` module + optional `rich` console output. `spectralstream/logging_config.py` configures logging; `MonitoringConfig.log_file` (`config.py:85`) and `console_report_interval` (`config.py:84`) control output. Plain text logs / JSON result files only.

## CI/CD & Deployment

**Hosting:**
- **None / self-hosted local process.** The engine runs as a local Python process or a local FastAPI (`uvicorn`) server. No container orchestration, no cloud deploy manifests (no `Dockerfile`, no `Makefile`, no `docker-compose` in repo). `configs/ollama.Modelfile` and `configs/vllm.json` describe how to wrap it behind Ollama/vLLM locally.

**CI Pipeline:**
- **None detected.** No `.github/`, `.gitlab-ci.yml`, `.travis.yml`, or `tox.ini`. Tests are run manually via `python -m pytest tests/ -v --tb=short -x --timeout=120` (`README.md:76`) or `tests/run_all_tests.py`.

## Webhooks & Callbacks

**Incoming:**
- **None.** (The FastAPI server exposes REST endpoints, but these are interactive API calls, not webhook subscriptions.)

**Outgoing:**
- **None** to external systems. The only outbound HTTP is to a *local* LM Studio/vLLM server via stdlib `http.client` (`spectralstream/serving/lmstudio.py`).

## Environment Configuration

**Required env vars:**
- None are strictly required to run the core engine. Optional `SS_`-prefixed vars override config (`spectralstream/config.py:120`): `SS_HOST`, `SS_PORT`, `SS_STATE_DIR`, `SS_KV_COMPRESSION`, `SS_K_BITS`, `SS_V_BITS`, `SS_SPECTRAL_RANK`, `SS_HDC_DIM`, `SS_HDC_NGRAM_ORDER`, `SS_GATE_LR`, `SS_TARGET_FPR`, `SS_MIN_BLOCK`, `SS_MAX_BLOCK`, `SS_CANDIDATES`, `SS_COHERENCE`, `SS_MAX_BUFFER`, `SS_BATCH_SIZE`, `SS_LMSTUDIO_URL`, `SS_MAX_CONNECTIONS`, `SS_TIMEOUT`, `SS_CHECKPOINT_INTERVAL`.

**Secrets location:**
- None. No secret store, no `.env` parsing in code. LM Studio config uses a literal `"api_key": "not-needed"` (`configs/lm_studio.json`, `configs/vllm.json`). (Forbidden-secret files were intentionally not read.)

## Model Hub / External Format Access

**HuggingFace Hub:** Not used. No `huggingface_hub` import; no network model download. The string `.lmstudio/models/huggingface` (`spectralstream/serving/lmstudio.py:16`) is only a *local disk search path*, not a hub call.

**Torch Hub / Model registries:** None.

**External file formats read:**
- `safetensors` (local) — `safetensors` dep.
- `gguf` (local) — `gguf` extra.
- `.ssf` custom (local) — `zstandard` dep.
- `xaml` / code samples used as test fixtures in `tests/data/` (e.g. `sample_main_window.xaml`, `sample_main_window.xaml.cs`) — local only.

## Integration Surface Summary

| Integration | Type | Mechanism | Required? | Declared dep? |
|-------------|------|-----------|-----------|---------------|
| LM Studio (local) | Local HTTP server | stdlib `http.client` | Optional | No (stdlib) |
| vLLM (local) | Config template | `configs/vllm.json` | Optional | No |
| Ollama (local) | Modelfile template | `configs/ollama.Modelfile` | Optional | No |
| OpenAI-compatible (local) | Sample client | `configs/openai_sdk.py` | Optional | No (`openai` undeclared) |
| Claude Desktop/Codex/OpenCode | MCP config stubs | `configs/*.json` | Optional | No |
| safetensors weights | Local file | `safetensors.safe_open` | Core | Yes |
| gguf weights | Local file | `gguf.GGUFReader` | Optional | Yes (`gguf` extra) |
| ssf format | Local file | `zstandard` | Core | Yes |
| Prometheus metrics | Local text exporter | custom `to_prometheus()` | Optional | No (stdlib) |
| HTTP bearer auth | Local token | `fastapi.security.HTTPBearer` | Optional | Yes (`fastapi` extra) |
| requests/matplotlib multimodal | Optional code path | guarded import | Disabled | No (undeclared) |

*Integration audit: 2026-07-07*
