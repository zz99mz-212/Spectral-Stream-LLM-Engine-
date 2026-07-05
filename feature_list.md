# Python Code-Graph Intelligence Engine — Feature Specification

## 1. CODE UNDERSTANDING (AST analysis, symbol extraction, type inference)

| # | Feature | Source | Description | Pri | Cplx |
|---|---------|--------|-------------|-----|------|
| 1.1 | Tree-Sitter AST Parsing (multi-language) | Quasar-V5 | Parse Python, C++, Rust, JS, TS, Go, Java, C#, Ruby, PHP via tree-sitter grammars. Produce uniform CST with node types, positions, and role assignments. | P0 | L |
| 1.2 | Symbol Extraction (functions, classes, methods) | Saguaro (entity_store) | Extract all named declarations: functions, classes, methods, variables, parameters, decorators, async functions, lambdas. Store with qualified name, file, line, column, docstring. | P0 | M |
| 1.3 | Import/Export Relationship Extraction | Quasar-V5 (graph-intelligence) | Resolve all import/require/include statements to concrete file paths. Build bi-directional import graph (imports + imported_by). Handle relative/absolute/package resolution. | P0 | M |
| 1.4 | Call Graph Construction | Saguaro (code_intelligence) | Resolve call sites to callee definitions. Handle bare calls, self.method(), cls.method(), staticmethod, classmethod, property, decorator wrapping, super(). | P0 | L |
| 1.5 | AST Fingerprinting | Saguaro (ast_fingerprint) | Compute structural hash of AST subtrees for clone detection and change tracking. FNV-1a hash over normalized AST with language-aware skipping of trivia. | P1 | M |
| 1.6 | Spectral Symbol Analysis | Saguaro (code_intelligence) | Map identifier tokens to frequency bins via universal hashing. Compute energy per symbol, detect dominant symbols per file, entropy of symbol distribution. | P2 | M |
| 1.7 | Type Inference (basic) | Research (engine.py) | Resolve variable types through assignment tracking and parameter hints. Track isinstance checks and type annotations. Return types from docstrings. | P2 | L |
| 1.8 | Architecture Boundary Detection | Quasar-V5 (gates) | Auto-detect subsystem boundaries from directory structure and package layout. Assign each entity to a subsystem. Detect cross-subsystem imports and dependency directions. | P1 | M |
| 1.9 | Decorator/Annotation Extraction | Quasar-V5 (graph-intelligence) | Extract decorators, annotations, and pragmas. Classify by type (e.g., @property, @staticmethod, @dataclass, @abstractmethod, custom decorators). Track decorator chains. | P2 | S |
| 1.10 | Docstring & Comment Parsing | Saguaro (disparate_relation_miner) | Extract and structure docstrings (NumPy, Google, reStructuredText, plain). Parse :param / :return / :raises. Detect TODO, FIXME, HACK, XXX, NOTE comments. | P2 | S |
| 1.11 | Exception Flow Analysis | Research (software_eng) | Track raise/try/except/finally chains. Determine which exceptions a function can raise. Build exception propagation graph across callers. | P3 | L |
| 1.12 | Async/Await Call Graph | Saguaro (code_intelligence) | Build separate call subgraph for async functions. Track await chains, event loop patterns, asyncio.gather, asyncio.create_task. | P2 | M |
| 1.13 | Lambda & Closure Analysis | Research | Detect lambda definitions, closure variable captures, binding scope. Track which variables are closed over and where lambdas are passed. | P3 | M |
| 1.14 | Class Hierarchy Resolution | Saguaro (entity_store) | Build MRO for each class (C3 linearization). Track base classes, abstract methods, mixins, metaclass usage, __init_subclass__. | P1 | M |
| 1.15 | Override Detection | Quasar-V5 (graph-refactor) | Detect method overrides in class hierarchies. Flag missing super().__init__() calls. Check signature compatibility with overridden methods. | P2 | M |

## 2. GRAPH QUERIES (search, navigation, relationships)

| # | Feature | Source | Description | Pri | Cplx |
|---|---------|--------|-------------|-----|------|
| 2.1 | Entity Find by Name | Quasar-V5 (graph.md) | Search all entities by name with fuzzy matching and n-gram indexing. Return file:line:type for each match. Support partial and case-insensitive matching. | P0 | M |
| 2.2 | N-Hop Neighbor Traversal | Quasar-V5 (graph-navigate) | Given an entity or file, traverse up to N hops through import/call/inheritance edges. Return all reachable entities with depth and path info. | P0 | M |
| 2.3 | Shortest Path Between Entities | Quasar-V5 (graph.md) | Compute shortest path between any two entities through the call/import/inheritance graph using BFS or Dijkstra. | P1 | M |
| 2.4 | Blast Radius Analysis | Quasar-V5 (impact.md) | Given a file or entity, compute transitive closure of affected entities at depth N. Assign risk score based on count, depth, and cross-subsystem contamination. | P0 | M |
| 2.5 | Caller/Callee List | Quasar-V5 (graph-navigate) | For any function/method, list all callers (incoming) and callees (outgoing). Show call sites with file:line. | P0 | S |
| 2.6 | Reverse Import Chain | Quasar-V5 (graph.md) | For any file, show which files import it (transitively). Used for determining if a file is a "leaf" or "hub". | P1 | S |
| 2.7 | Full-Text Search | Quasar-V5 (graph-search) | Index source text for keyword search with TF-IDF scoring. Support phrase matching, file-type filtering, and regex patterns. | P1 | L |
| 2.8 | Subsystem Breakdown | Quasar-V5 (graph.md) | List all subsystems with file/entity/edge counts. Show cross-subsystem dependency matrix. | P1 | M |
| 2.9 | Language Distribution | Quasar-V5 (graph.md) | Show file/line/entity counts by programming language across the project. | P2 | S |
| 2.10 | Code Clone Detection | Quasar-V5 (graph-intelligence) | Find structurally similar code blocks using AST fingerprint hashing and normalized token sequences. Report similarity scores and locations. | P2 | L |
| 2.11 | Bridge Detection (cross-language) | Quasar-V5 (graph.md) | Detect cross-language call bridges (e.g., Python calling C++ via ctypes, FFI, Cython, pybind11). Map interface boundaries between languages. | P2 | L |
| 2.12 | Entry Point Discovery | Quasar-V5 (graph.md) | Find all entry points: main(), CLI handlers, signal handlers, thread targets, async entry points, web routes, cron jobs, registered callbacks. | P1 | M |
| 2.13 | Natural Language Query Routing | Quasar-V5 (query.md) | Route NL queries like "untested functions in plasma" to appropriate graph query modes using keyword matching and intent classification. | P1 | M |
| 2.14 | Similarity Search | Quasar-V5 (graph-intelligence) | Find entities similar to a given entity by embedding similarity (cosine over extracted features). | P2 | L |
| 2.15 | Entity Relationship Graph | Saguaro (graph) | Export full entity-relationship graph as adjacency list, CSR matrix, or JSON dump. Support edge type filtering (calls, imports, inherits, contains). | P1 | M |

## 3. DEAD CODE ANALYSIS (unused code, orphans, unwired)

| # | Feature | Source | Description | Pri | Cplx |
|---|---------|--------|-------------|-----|------|
| 3.1 | Unused Function Detection | Quasar-V5 (graph-deadcode) | Detect functions/methods that are defined but never called (zero callers). Assign alive_score based on call graph analysis. | P0 | M |
| 3.2 | Unused Import Detection | Quasar-V5 (graph-import-optimizer) | Detect imports that are never referenced in the importing file. Suggest removal with confidence scoring. | P1 | M |
| 3.3 | Unused Class Detection | Quasar-V5 (graph-deadcode) | Detect classes that are never instantiated, subclassed, or referenced by qualified name. | P1 | M |
| 3.4 | Unused Variable Detection | Saguaro (code_intelligence) | Detect assigned variables that are never read. Distinguish between intended unused (e.g., _ convention) and accidental. | P2 | M |
| 3.5 | Unused Parameter Detection | Quasar-V5 (graph-deadcode) | Detect function parameters that are never used in the function body. Flag callback signatures where params are required by protocol. | P2 | M |
| 3.6 | Unwired Code Detection (WM Isolation) | Quasar-V5 (graph-intelligence) | Detect code that is not reachable from any entry point or subsystem hub. Used to find code not wired into the application's runtime graph. | P1 | M |
| 3.7 | Dead Export Detection | Saguaro (entity_utils) | Detect exported/public names that are never imported by any other module in the project. | P2 | S |
| 3.8 | Orphan File Detection | Quasar-V5 (graph-deadcode) | Detect files not reachable from any entry point via import chains. Files that exist on disk but are not part of the live codebase. | P2 | M |
| 3.9 | Dead Code Confidence Scoring | Quasar-V5 (graph-deadcode) | Assign confidence to each dead code finding based on graph certainty. Dynamic dispatch, decorator registration, and plugin patterns reduce confidence. | P1 | M |
| 3.10 | Dead Enum/Constant Detection | Research | Detect enum members and constants that are never referenced. Filter out self-referential enum patterns. | P3 | S |

## 4. QUALITY & METRICS (complexity, test coverage, health)

| # | Feature | Source | Description | Pri | Cplx |
|---|---------|--------|-------------|-----|------|
| 4.1 | McCabe Cyclomatic Complexity | Saguaro (code_intelligence) | Compute cyclomatic complexity for each function/method: count of if/while/for/and/or/case/except with branching. Report per-file, per-subsystem aggregates. | P1 | M |
| 4.2 | Cognitive Complexity | Research (Clean Architecture) | Compute cognitive complexity: nesting depth, break/continue scope, recursion, boolean logic chains. Weighted score per function. | P2 | M |
| 4.3 | Test Coverage Estimation | Quasar-V5 (graph-test-hints) | For each entity, determine if corresponding test files exist. Estimate behavioral coverage: function-to-test ratio. Track test_coverage per entity. | P1 | M |
| 4.4 | Test Gap Detection | Quasar-V5 (graph-test-hints) | List all functions/classes that lack test coverage. Filter by subsystem, complexity tier, or risk score. | P1 | M |
| 4.5 | Health Score Dashboard | Quasar-V5 (graph-health-monitor) | Aggregate health metrics: node count, edge count, annotation count, avg complexity, test coverage %, dead code count. Compute 0-100 health score. | P0 | M |
| 4.6 | Code Smell Detection | Quasar-V5 (graph-intelligence) | Detect 16+ code smells: god class, long method, excessive params, data clump, primitive obsession, switch statements, lazy class, shotgun surgery. | P1 | L |
| 4.7 | Risk Score per File | Quasar-V5 (graph-intelligence) | Compute risk score (0-100) per file from violation counts, severity weights, complexity, cross-subsystem imports. | P0 | S |
| 4.8 | Annotation Count by Severity | Quasar-V5 (graph-intelligence) | Maintain running counts of annotations by severity (critical, high, medium, low, info). Show trends over time. | P1 | S |
| 4.9 | Multi-Dimensional Quality Score | Quasar-V5 (graph-review) | Compute quality, security, maintainability, test coverage, and overall scores per file and per subsystem. | P1 | M |
| 4.10 | Behavioral Coverage (function-level) | Quasar-V5 (graph-test-hints) | For each entity, determine if it's called/referenced from test files. Compute ratio of tested entities to total entities. | P2 | M |
| 4.11 | Hot-Path Complexity Scoring | Saguaro (hot-paths) | Identify most-called functions across the codebase. Weight complexity by call frequency. Flag hot functions with high complexity for refactoring. | P2 | M |
| 4.12 | Freshness Tracking | Research (engine.py) | Track last-modified timestamps per file. Detect stale entities (not modified in N days) vs actively churned entities. | P2 | S |

## 5. SECURITY & COMPLIANCE (vulnerabilities, secrets, standards)

| # | Feature | Source | Description | Pri | Cplx |
|---|---------|--------|-------------|-----|------|
| 5.1 | Secret/Token Detection | Quasar-V5 (graph.md) | Scan for hardcoded API keys, passwords, tokens, certificates, SSH keys. Use regex patterns and entropy detection. | P1 | M |
| 5.2 | CVE Matching | Saguaro (cve_matcher) | Match dependency names and versions against known CVE database. Flag vulnerable packages with CVE IDs and severity. | P2 | L |
| 5.3 | SAST Pattern Detection | Quasar-V5 (gate_config) | Detect common security anti-patterns: eval/exec usage, SQL injection vectors, shell injection via os.system/subprocess, unsafe deserialization (pickle). | P1 | M |
| 5.4 | Compliance Gap Analysis (NIST/SOC2/ISO) | Quasar-V5 (graph.md) | Check codebase against compliance frameworks. Map entities to control requirements. Report gaps by severity. | P2 | L |
| 5.5 | License Compliance Scan | Quasar-V5 (graph.md) | Detect SPDX license headers. Check dependency licenses against allowlist. Report missing or incompatible licenses. | P2 | M |
| 5.6 | Memory Safety Analysis (Python) | Research | Detect unsafe patterns: buffer manipulation, ctypes misuse, __del__ reference cycles, weakref patterns, large object accumulation. | P2 | M |
| 5.7 | Supply Chain Risk Assessment | Quasar-V5 (graph.md) | Analyze dependency tree depth, number of maintainers per dependency, repo freshness, deprecated packages. Score supply chain risk. | P3 | L |
| 5.8 | SBOM Generation | Quasar-V5 (graph.md) | Generate Software Bill of Materials from dependency graph. Format: SPDX 2.3 JSON. Include transitive dependencies. | P2 | M |
| 5.9 | Code Injection Detection | Saguaro (ml_security) | Detect patterns where user input reaches dangerous sinks: eval, exec, compile, __import__, os.system, subprocess with shell=True. | P1 | M |
| 5.10 | PII Scanning | Quasar-V5 (graph.md) | Detect hardcoded email addresses, phone numbers, SSN, credit card numbers, IP addresses. Flag for removal or externalization. | P1 | M |
| 5.11 | Sanctions Screening | Quasar-V5 (graph.md) | Check dependency origins against OFAC/EU/UN sanctions lists based on developer location data and repo jurisdiction. | P3 | XL |
| 5.12 | Seccomp/Capability Audit | Quasar-V5 (gate_config) | Audit required system capabilities. Check for unnecessary privilege escalation patterns (setuid, capabilities, namespace escapes). | P3 | M |
| 5.13 | FIPS Compliance Checker | Quasar-V5 (gate_config) | Verify crypto usage against FIPS 140-3 requirements. Detect non-approved algorithms (MD5, RC4, DES) and non-compliant RNG usage. | P3 | L |

## 6. PERFORMANCE (hot paths, energy, profiling)

| # | Feature | Source | Description | Pri | Cplx |
|---|---------|--------|-------------|-----|------|
| 6.1 | Hot Path Detection | Saguaro (hot-paths) | Identify most-called functions across the call graph. Rank by cumulative invocation count. Report top 10/50/100 hot functions. | P1 | M |
| 6.2 | Complexity-Performance Correlation | Saguaro (code_intelligence) | Correlate cyclomatic complexity with hot-path frequency. Flag hot functions with high complexity as refactoring candidates. | P2 | M |
| 6.3 | Energy Profile Estimation | Quasar-V5 (graph-energy) | Estimate per-file and per-subsystem energy consumption based on call frequency, complexity, IO operations, and string operations. | P2 | L |
| 6.4 | Carbon Footprint Estimate | Quasar-V5 (graph-energy) | Convert energy estimates to carbon grams per operation. Use regional grid intensity factors. | P3 | L |
| 6.5 | Performance Hotspot Detection | Quasar-V5 (graph-intelligence) | Flag files with excessive loops, recursion, repeated computations, large list comprehensions, nested iterations. | P2 | M |
| 6.6 | N+1 Query Detection | Research | Detect ORM patterns that cause N+1 query problems: iterating over relation and accessing per-row attributes. | P3 | L |
| 6.7 | Cache Analysis (cache-line-aware) | Hardware Research (RAPL) | For data-intensive code, estimate cache miss rates based on data structure access patterns. Flag loops with poor spatial locality. | P3 | XL |
| 6.8 | NUMA Placement Analysis | Hardware Research (CXL) | Analyze NUMA-aware code patterns. Detect cross-NUMA memory access in data-parallel algorithms. | P3 | L |
| 6.9 | Concurrency Hotspot Scan | Quasar-V5 (graph.md) | Detect lock contention patterns, thread pool oversubscription, unnecessary serialization. | P3 | L |
| 6.10 | Per-Operation Energy Breakdown | Quasar-V5 (graph-telemetry-enhanced) | Estimate energy per operation type (compute, IO, network, serialization). Show dominant energy consumers. | P3 | L |

## 7. REFACTORING (monolithic files, extract method, rename)

| # | Feature | Source | Description | Pri | Cplx |
|---|---------|--------|-------------|-----|------|
| 7.1 | Extract Method Candidates | Quasar-V5 (graph-refactor) | Identify inline code blocks with high internal cohesion and low coupling to surrounding function. Suggest extract method with confidence score. | P1 | L |
| 7.2 | Extract Class Candidates | Quasar-V5 (graph-refactor) | Detect classes with too many responsibilities. Identify groups of methods/fields that form a coherent subdomain. | P2 | L |
| 7.3 | Monolithic File Detection | Quasar-V5 (graph-refactor) | Flag files exceeding complexity/line count thresholds. Suggest file splitting with subsystem-aware boundary detection. | P1 | M |
| 7.4 | Dead Code Removal Suggestions | Quasar-V5 (graph-refactor) | For each dead code finding, suggest the exact lines to remove with confidence score and reasoning. | P1 | S |
| 7.5 | Abstraction Leak Fix Suggestions | Quasar-V5 (graph-refactor) | Detect when a low-level implementation detail leaks through a high-level API. Suggest encapsulating or abstracting. | P2 | M |
| 7.6 | Rename Refactoring Impact | Quasar-V5 (graph-refactor) | Given a rename candidate, show all files/lines that would need updating. Compute rename complexity score. | P1 | M |
| 7.7 | Import Optimization | Quasar-V5 (graph-import-optimizer) | Suggest removing unused imports, adding missing imports, replacing wildcard imports with specific ones. Offer auto-fix. | P2 | M |
| 7.8 | Forward Declaration Suggestions | Quasar-V5 (graph-import-optimizer) | For C++/Python type annotations, suggest forward declarations to reduce import dependencies and circular imports. | P3 | M |
| 7.9 | Self-Healing Fix Suggestions | Quasar-V5 (graph-intelligence) | For known violation patterns (hardcoded values, weak exception handling, missing logging), generate concrete fix suggestions. | P1 | L |
| 7.10 | Code Pattern Standardization | Saguaro (llvm_analysis) | Detect non-idiomatic code patterns and suggest project-standard alternatives. Enforce style guide rules automatically. | P2 | M |
| 7.11 | C++23 / Modernization Suggestions | Quasar-V5 (graph-refactor) | For legacy code patterns, suggest modern equivalents (e.g., f-strings over %, pathlib over os.path, dataclasses over manual __init__). | P2 | M |
| 7.12 | Parameter Object Introduction | Research (Refactoring) | Detect function clusters with >3 parameters that share a common subset. Suggest introducing parameter objects. | P3 | M |

## 8. ARCHITECTURE (layers, boundaries, cycles, stability)

| # | Feature | Source | Description | Pri | Cplx |
|---|---------|--------|-------------|-----|------|
| 8.1 | Import Cycle Detection | Quasar-V5 (graph-cycles) | Detect circular import chains between modules. Show cycle path with depth and risk score. Support filtering by cycle length. | P0 | M |
| 8.2 | Dependency Layer Violation | Quasar-V5 (rules) | Enforce layer rules (e.g., adapters must not depend on UI). Detect upward and sideways dependency violations. | P1 | M |
| 8.3 | Architecture Stability Score | Quasar-V5 (graph-intelligence) | Compute architectural stability: ratio of stable (no incoming changes) vs volatile components. Instability = Ce / (Ca + Ce). | P1 | M |
| 8.4 | Abstraction/Stability (Main Sequence) | Research (Clean Architecture) | Compute D' = |A + I - 1| distance from main sequence. Plot components on A-I graph. Flag components far from main sequence. | P2 | M |
| 8.5 | Cross-Package Call Analysis | Saguaro (xboundary) | Analyze all cross-package calls. Count cross-package edges per package. Identify packages with excessive external coupling. | P1 | M |
| 8.6 | Import Group Overlap | Research | Find files with overlapping import patterns that may benefit from shared abstractions. | P2 | M |
| 8.7 | Fan-In/Fan-Out Analysis | Quasar-V5 (graph.md) | For each module, count incoming (fan-in) and outgoing (fan-out) dependencies. High fan-in = utility/abstraction, high fan-out = orchestrator/module. | P1 | S |
| 8.8 | Layer Diagram Generation | Quasar-V5 (graph-intelligence) | Generate architectural layer diagram: presentation → application → domain → infrastructure. Classify each module to a layer. | P2 | M |
| 8.9 | Dependency Impact Scoring | Quasar-V5 (impact.md) | For each file/entity, compute what would break if it were deleted. Score by count of affected downstream entities. | P1 | M |
| 8.10 | API Stability Analysis | Quasar-V5 (graph-intelligence) | Track changes to public API signatures over time. Detect breaking changes: removed params, changed types, removed exports. | P2 | L |
| 8.11 | Package Dependency Matrix | Saguaro (imports) | Generate N×N matrix showing which packages import which. Identify hub packages (most imported) and sink packages (import everything). | P2 | M |
| 8.12 | Mermaid JS Export | Saguaro (mermaid) | Export dependency graph as Mermaid JS flowcharts. Support subsystem, layer, and entity-level granularity. | P2 | S |
| 8.13 | TLA+ Invariant Export | Quasar-V5 (tools/tla_to_cpp) | Export architectural invariants (e.g., "no circular imports") as TLA+ formulas for formal verification. | P3 | XL |
| 8.14 | Bus Factor Estimation | Quasar-V5 (graph.md) | Estimate bus factor: number of developers who know each subsystem. Based on git blame distribution. Flag files with single-author risk. | P2 | M |

## 9. LLM INTEGRATION (context building, NL queries, suggestions)

| # | Feature | Source | Description | Pri | Cplx |
|---|---------|--------|-------------|-----|------|
| 9.1 | Token-Budgeted Context Extraction | Quasar-V5 (graph-intelligence) | Given a file, extract entities, relationships, and violations within a token budget. Replace full file reads with structured summary. | P0 | M |
| 9.2 | LLM Context Packet Builder | Research (engine.py) | Build structured context packets for queries: corpus stats, top matches, source-text snippets. Limit to max_chars. | P1 | S |
| 9.3 | NL Query → Graph Query Routing | Quasar-V5 (query.md) | Route natural language queries like "untested functions in plasma" to appropriate graph query modes using keyword matching. | P1 | M |
| 9.4 | Pre-Edit File Check + Block | Quasar-V5 (graph-intelligence) | Before any edit, check file for violations. If risk_score ≥ 70, block the edit and return violation details. | P0 | M |
| 9.5 | Post-Edit Violation Diff | Quasar-V5 (graph-intelligence) | After edit, compare violations before/after. Show delta per violation category. Alert on new violations introduced. | P1 | M |
| 9.6 | Autocomplete from Graph Index | Quasar-V5 (graph-autocomplete) | Suggest function names, class names, method completions from graph entity index. Suggest #include headers from observed imports. | P2 | L |
| 9.7 | Context-Aware Read Alternative | Quasar-V5 (graph-file-resolver) | Intercept read() calls for source files. Return graph context (entities, relationships, violations) instead of raw content. | P1 | M |
| 9.8 | LLM Bus Integration | Saguaro (llm_bus) | Bridge between code graph and LLM backend. Send structured requests with system prompts from graph data. Parse structured responses. | P2 | M |
| 9.9 | Commit Message Generation | Quasar-V5 (graph-commit-msg) | From git diff + graph entity relationships, generate structured commit messages: type(scope): description. | P2 | M |
| 9.10 | HDC Vector Export for LLM | Research (engine.py) | Export entity embeddings as hyperdimensional computing vectors for LLM tokenization. Support HDC similarity search. | P3 | L |

## 10. LSP REPLACEMENT (goToDef, references, hover, completion)

| # | Feature | Source | Description | Pri | Cplx |
|---|---------|--------|-------------|-----|------|
| 10.1 | Go To Definition | Saguaro (cursor_engine) | Given a symbol name at a file:line, resolve to its definition location. Use entity index for fast lookup. | P0 | M |
| 10.2 | Find All References | Quasar-V5 (graph-navigate) | Find all references to a symbol across the codebase. Show file:line for each reference. Classify as read/write/call. | P0 | M |
| 10.3 | Hover Information | Quasar-V5 (graph-diagnostics) | For a symbol at file:line, return its type, docstring, parameters, callers count, complexity score, and recent changes. | P1 | M |
| 10.4 | Autocomplete from Entity Index | Quasar-V5 (graph-autocomplete) | Given prefix, suggest symbol completions from the entity index. Rank by import proximity and usage frequency. | P1 | L |
| 10.5 | Syntax Diagnostics from Graph | Quasar-V5 (graph-diagnostics) | Report inline diagnostics: violations at specific file:line positions. Cache with 30s TTL. Merge graph + LSP diagnostics. | P1 | M |
| 10.6 | Workspace Symbols | Quasar-V5 (graph-intelligence) | List all symbols matching a query across the workspace. Group by file and symbol type. | P1 | M |
| 10.7 | Document Symbols | Quasar-V5 (graph-tui) | List all symbols in a file: classes, functions, methods, variables with line ranges. Used for file outline/navigation. | P1 | S |
| 10.8 | Code Lens (Callers Count Inline) | Quasar-V5 | Show inline annotation: "3 callers" or "10 references" above function definitions. | P2 | M |
| 10.9 | Signature Help | Research | Given function name, show its signature with parameter names, types, defaults, and docstrings. Triggered on '('. | P2 | M |
| 10.10 | Inline Fix Suggestions | Quasar-V5 (graph-diagnostics) | For diagnostics with fix information, present inline fix suggestions that can be applied with one action. | P2 | M |

## 11. VISION SYSTEM (GUI code understanding, XAML, HTML, CSS)

| # | Feature | Source | Description | Pri | Cplx |
|---|---------|--------|-------------|-----|------|
| 11.1 | View Definition Extraction | Quasar-V5 (graph-vision) | Scan AXAML/HTML files for view definitions. Extract x:Class, DataContext bindings, xmlns references. | P1 | M |
| 11.2 | View → ViewModel Edge Mapping | Quasar-V5 (graph-vision) | Create graph edges from views to their ViewModels via DataContext binding analysis. | P1 | M |
| 11.3 | ViewModel → Service Dependency Chain | Quasar-V5 (graph-vision) | Scan ViewModel files for service/manager/repository constructor injections. Map full dependency chain: view → VM → service → backend. | P1 | M |
| 11.4 | HTML Templating Detection | Quasar-V5 (graph-vision) | Detect HTML templates, Angular controllers, Vue components, React patterns. Map controller/component to service dependencies. | P2 | L |
| 11.5 | CSS Class Usage Analysis | Quasar-V5 (graph-vision) | Track CSS class definitions against usage in HTML templates. Detect unused CSS classes and missing class references. | P3 | M |
| 11.6 | JavaScript/TypeScript Call Graph | Quasar-V5 (graph-vision) | Build JS/TS call graph from AST analysis. Handle ES module imports, require(), dynamic imports, and webpack chunks. | P2 | L |
| 11.7 | Full-Stack Traceability Graph | Quasar-V5 (graph-vision) | Connect frontend views → API routes → backend handlers → database queries in a unified graph. | P2 | XL |
| 11.8 | i18n String Coverage | Quasar-V5 (graph-telemetry-enhanced) | Scan locale JSON files for translated vs. untranslated strings. Compute coverage % per locale and per subsystem. | P2 | M |
| 11.9 | a11y Compliance Scan | Quasar-V5 (graph-telemetry-enhanced) | Check AXAML/HTML for accessibility attributes: AutomationProperties, aria-*, keyboard navigation, contrast ratios. | P2 | L |
| 11.10 | GUI Test Coverage | Quasar-V5 (graph-vision) | Map UI components to their test files. Flag untested views and ViewModels with no corresponding test file. | P2 | M |

## 12. COLLABORATION (PR readiness, code review, git churn)

| # | Feature | Source | Description | Pri | Cplx |
|---|---------|--------|-------------|-----|------|
| 12.1 | PR Readiness Score | Quasar-V5 (graph-pr-dashboard) | Compute 0-100 PR readiness score from critical issues, dead code, compliance gaps, test coverage, and complexity issues. | P0 | M |
| 12.2 | Pre-Commit Gate | Quasar-V5 (graph-precommit) | Before every commit, check PR readiness. Score < 60 BLOCK. Score < 80 WARN. Score ≥ 80 ALLOW. | P0 | M |
| 12.3 | Code Review Summary (Multi-Layer) | Quasar-V5 (graph-review) | Generate review: quality + security + maintainability scores, top blocking issues, warnings, recommendations. | P1 | M |
| 12.4 | Blast Radius per Commit | Quasar-V5 (impact.md) | For each changed file in a commit, compute blast radius. Show all files that would need re-testing. | P1 | M |
| 12.5 | Git Co-Change Mining | Quasar-V5 (impact.md) | Analyze git history for files that change together frequently. Suggest which files to review/test together. | P2 | L |
| 12.6 | Code Churn Heatmap | Quasar-V5 (graph.md) | Per-file change frequency over time. Flag high-churn files for refactoring. Show churn per subsystem. | P2 | M |
| 12.7 | Contributor Impact Analysis | Quasar-V5 (graph-intelligence) | Analyze which files/subsystems each contributor touches. Show expertise matrix: contributor → subsystem. | P2 | M |
| 12.8 | PR Change Scope Analysis | Quasar-V5 (graph-pr-dashboard) | Given a PR diff, classify changes: feature, fix, refactor, test, docs. Show affected entities and risk level. | P2 | M |
| 12.9 | Review Bottleneck Detection | Research (DevOps) | Flag files with many pre-existing violations being touched in a PR. Suggest they be cleaned up in a separate PR. | P2 | S |
| 12.10 | Temporal Metrics Capture | Quasar-V5 (orchestrator-agent) | Capture before/after snapshots of graph metrics for trend analysis. Store in temporal metrics table. | P2 | M |

## 13. PERSISTENCE & CACHING (snapshots, trends, evolution)

| # | Feature | Source | Description | Pri | Cplx |
|---|---------|--------|-------------|-----|------|
| 13.1 | Snapshot Serialization | Saguaro (persistence) | Serialize full graph state to binary snapshot format. Magic header, versioned sections: entities, edges, annotations, metadata. | P1 | L |
| 13.2 | Snapshot Delta Comparison | Quasar-V5 (graph.md) | Compare two graph snapshots. Show added/removed/changed entities, edges, annotations. Compute delta score. | P2 | L |
| 13.3 | Metric Trend Tracking | Quasar-V5 (graph.md) | Track health_score, annotation_count, complexity metrics over time. Store 30/90/365 day rolling history. | P1 | M |
| 13.4 | Incremental Indexing | Saguaro (parallel_indexer) | On file change, re-index only the changed file and entities that depend on it. No full rebuild needed. | P0 | L |
| 13.5 | Multi-Level Result Cache | Saguaro (cache_manager) | LRU cache for query results with TTL. In-memory + mmap'd cache levels. Content-addressed (SHA256 keyed on query + args). | P1 | M |
| 13.6 | Cache Warmup on Session Start | Quasar-V5 (tool-registry) | Pre-warm result cache on startup by pre-computing common queries (stats, health, subsystem breakdown). | P2 | M |
| 13.7 | On-Disk Manifest Cache | Quasar-V5 (tool-registry) | Cache tool/manifest discovery to JSON file. Validate cache freshness via directory mtime comparison. Avoid re-scan at every session. | P1 | S |
| 13.8 | Tool Result Caching | Quasar-V5 (tool-registry) | Cache tool execution results by SHA256 hash of tool name + args. TTL configurable. Clear on tool reload. | P1 | M |
| 13.9 | Graph DB Size Management | Quasar-V5 (graph-intelligence) | Monitor graph DB size. Auto-prune stale entities. Support configurable max size with FIFO eviction. | P2 | M |
| 13.10 | Checkpoint-Based Persistence | Saguaro (persistence) | Periodic checkpoints to disk during long-running indexing. Crash recovery from last valid checkpoint. | P2 | M |

## 14. NOVEL INVENTIONS (cross-session memory, multi-repo, self-healing)

| # | Feature | Source | Description | Pri | Cplx |
|---|---------|--------|-------------|-----|------|
| 14.1 | Cross-Session Entity Memory | Saguaro (holographic_entity) | Store entities across sessions with holographic vector embeddings. FFT-based binding of entity features into HD vectors for similarity recall. | P2 | XL |
| 14.2 | Context Diffusion (Heat Equation) | Saguaro (context_diffusion) | Solves heat equation on entity graph using conjugate gradient. Context flows from high-information entities to low-information ones. CSR Laplacian matvec. | P2 | XL |
| 14.3 | Holographic Attention | Saguaro (holographic_attention) | Attention mechanism using holographic reduced representations. Query → key binding via circular convolution. Used for entity relevance ranking. | P3 | XL |
| 14.4 | Bifurcation Memory (Binary Splitting) | Saguaro (bifurcation_memory) | Entity storage that bifurcates when memory pressure exceeds threshold. Hot entities stay in fast path, cold entities are compressed/evicted. | P3 | XL |
| 14.5 | Causal Inference Engine | Saguaro (causal_engine) | Granger causality test between entity time series. F-statistic + p-value computation. Directional causal edges between entities. | P3 | XL |
| 14.6 | Disparate Relation Mining | Saguaro (disparate_relation_miner) | Mine relationships between entities that have no explicit connection. Use naming similarity, comment co-mentions, proximity in AST, git co-change patterns. | P2 | L |
| 14.7 | Quantum Walk Based Search | Saguaro (quantum_scheduler) | Implement continuous-time quantum walk on entity graph for search. Use Hamiltonian evolution instead of classical random walk. | P3 | XL |
| 14.8 | MPS Tensor Network Embedding | Saguaro (tensor_train) | Encode entity relationships as Matrix Product State (tensor train). Compress large graphs via SVD-based truncation. | P3 | XL |
| 14.9 | Predictive Maintenance | Quasar-V5 (graph.md) | From trend data (health score, annotation count, complexity), predict when quality will degrade below threshold. Proactive alerting. | P2 | L |
| 14.10 | Multi-Repo Graph Merge | Saguaro (search/distributed) | Import entity graphs from multiple repositories. Merge conflict resolution for entities with same qualified name. Cross-repo call chain analysis. | P3 | XL |
| 14.11 | Self-Healing Actions | Quasar-V5 (graph-intelligence) | For auto-fixable violations (unused imports, missing logging, hardcoded values), generate and apply fix automatically. | P1 | L |
| 14.12 | World Model Wiring Audit | Quasar-V5 (wm-agent) | Check that all subsystems are wired into the application's evolution/update cycle. Detect unwired code paths and suggest wiring hooks. | P2 | L |
| 14.13 | Entropy Anomaly Detection | Saguaro (entropy_detector) | Track entropy of the entity graph over time. Sudden drops in entropy indicate concentrated changes that may destabilize architecture. | P3 | L |
| 14.14 | Graph Rewiring (Dynamic Reorganization) | Saguaro (graph_rewiring) | Periodically rewire entity graph for better locality: cluster entities that are frequently queried together. Reduce average query path length. | P3 | XL |
| 14.15 | Symplectic Runtime Integration | Saguaro (symplectic_runtime) | Connect graph query execution to energy-preserving symplectic integrator. Query scheduling with guaranteed energy/delay tradeoff. | P3 | XL |

## 15. PLUGINS & EXTENSIBILITY (tool registry, gates, hooks)

| # | Feature | Source | Description | Pri | Cplx |
|---|---------|--------|-------------|-----|------|
| 15.1 | Tool Auto-Discovery from Scripts | Quasar-V5 (tool-registry) | Scan tools/ directories for Python/bash scripts with header comments. Parse # tool:, # description:, # args: annotations. Register as callable tools. | P0 | M |
| 15.2 | Tool Registry with Manifest Cache | Quasar-V5 (tool-registry) | Maintain tool registry with name, description, arg schema. Cache manifest to JSON. Validate freshness by directory mtime. | P0 | M |
| 15.3 | Tool Execution with Retry/Cache | Quasar-V5 (tool-registry) | Execute discovered tools via spawn. Support retry with exponential backoff, result caching (SHA256 key), timeout, max buffer. | P1 | M |
| 15.4 | Tool Health Tracking | Quasar-V5 (tool-registry) | Track per-tool execution count, failure rate, avg duration, consecutive failures. Report flaky tools. Auto-disable tools with >5 consecutive failures. | P1 | S |
| 15.5 | Binary Tool Discovery (ELF) | Quasar-V5 (tool-registry) | Detect ELF executables in build directories. Auto-register as tools. Receive args via TOOL_ARGS env variable. | P1 | M |
| 15.6 | Sideload Tool Support (external repos) | Quasar-V5 (tool-registry) | sideload.json manifest listing external tool repos. Auto-clone (git clone --depth=1) and register tools from them. Lower priority than built-in. | P2 | M |
| 15.7 | build_tool — Create New Tools | Quasar-V5 (tool-registry) | Create new Python/bash tool scripts with proper header. Write to tools/ directory. Generate arg schema from JSON description. | P2 | S |
| 15.8 | reload_tools — Hot-Reload Registry | Quasar-V5 (tool-registry) | Rescan all tool directories, C++ binary dirs, and sideload manifests. Re-register all tools without restarting. | P1 | S |
| 15.9 | Graph-Native CI Gates (YAML) | Quasar-V5 (gates) | Define gates as YAML with SQL query + pass_if expression. Register gates into graph DB. Execute via mode=gate. | P0 | M |
| 15.10 | Gate Execution by Pipeline | Quasar-V5 (gate.md) | Run quality/security/performance/compliance gates. Gate_runner iterates over gates, evaluates queries, checks pass_if, collects results. | P1 | M |
| 15.11 | Zero-Tolerance Gate Mode | Quasar-V5 (gate_config) | Gates marked zero_tolerance block merge/release pipelines on failure. Non-blocking gates allow merge with warnings. | P1 | S |
| 15.12 | Pre-Edit Plugin Hook | Quasar-V5 (graph-intelligence) | Hook into graph_intelligence.ts: before every edit, run checkFileGraphFull. If risk ≥ 70, return blocking message. | P0 | M |
| 15.13 | Post-Edit Plugin Hook | Quasar-V5 (graph-intelligence) | After edit, run checkFileGraphFull again. Compare violations before/after. Log violation delta. | P1 | M |
| 15.14 | Read Intercept Hook (Context Suggestions) | Quasar-V5 (graph-intelligence) | On read() of source file, suggest graph_context as alternative. Show token savings estimate. | P1 | S |
| 15.15 | File Edit Event → Auto Reindex | Quasar-V5 (graph-intelligence) | On file.edited event, trigger background reindex of the changed file. Update entity graph incrementally. | P1 | M |
| 15.16 | TUI Plugin Integration | Quasar-V5 (graph-tui) | Provide sidebar content: health score, violations count, entities, subsystem. Status indicator next to prompt. KV storage for widget state. | P2 | L |
| 15.17 | Subagent Orchestration | Quasar-V5 (orchestrator-agent) | Coordinate tool-agent (diagnose), fix-agent (fix), test-agent (verify), wm-agent (wire). Enforce DISCOVER → DIAGNOSE → FIX → VERIFY → TEST sequence. | P1 | M |
| 15.18 | Graph-First Rule Enforcement | Quasar-V5 (rules) | Zero-tolerance rule: grep/rg forbidden. Pre-edit graph_check mandatory. Plugin auto-blocks edits with risk > 70. | P0 | S |
| 15.19 | Tool Catalog for Agent Discovery | Quasar-V5 (orchestrator-agent) | Provide tool_catalog() function listing all available tools with descriptions. Support search filtering by keyword. | P2 | S |
| 15.20 | Schema Migration Support | Quasar-V5 (query.md) | Support schema versioning and migration. Commands: /query schema-status, /query schema-migrate. | P2 | M |

---

## Total: 119 features across 15 categories

### Priority distribution:
- **P0 (critical):** 18 features
- **P1 (high):** 40 features
- **P2 (medium):** 48 features
- **P3 (low):** 13 features

### Complexity distribution:
- **S (small):** 16 features
- **M (medium):** 60 features
- **L (large):** 30 features
- **XL (extra large):** 13 features
