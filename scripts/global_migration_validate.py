"""End-to-end validation of code-graph system."""

import sys, json, os, subprocess, tempfile


def test(name, condition, detail=""):
    status = "✓" if condition else "✗"
    print(f"  {status} {name} {detail}")
    return condition


errors = 0

# 1. Config validation
with open("/home/mike/.config/opencode/opencode.jsonc") as f:
    gc = json.load(f)
errors += not test("Global config valid JSON", True)
errors += not test("compaction.prune=True", gc["compaction"]["prune"] == True)
errors += not test("compaction.reserved=100000", gc["compaction"]["reserved"] == 100000)

# 2. Intelligence imports tests
sys.path.insert(0, ".")
from intelligence import EntityKind, CodeGraph, SearchIndex
from intelligence.dead_code_analyzer import DeadCodeAnalyzer
from intelligence.quality_scanner import QualityScanner
from intelligence.repo_insight import RepoInsight
from intelligence.graph_intelligence import GraphIntelligence
from intelligence.vision.gui_analyzer import AvaloniaAnalyzer
from intelligence.vision.web_analyzer import HtmlAnalyzer

errors += not test("All intelligence modules import", True)

# 3. Index loads
from intelligence.parallel_indexer import IndexerConfig, ParallelIndexer

idx = ParallelIndexer(IndexerConfig(root_dir=".", persist_path="intelligence/index"))
loaded = idx.load()
errors += not test("Index loads", loaded)
if loaded:
    stats = idx.stats
    errors += not test(
        f"Index has entities ({stats['total_entities']})", stats["total_entities"] > 0
    )
    errors += not test(
        f"Index has relationships ({stats['graph_relationships']})",
        stats["graph_relationships"] > 0,
    )

# 4. Core features work
ri = RepoInsight(idx.graph, idx.search_index)
errors += not test("Registry dispatch", len(ri.find_registry_dispatch()) > 0)
errors += not test(
    "Source snippets", len(ri.get_source_snippets("compress_tensor")) > 0
)
errors += not test(
    "Auto-import", len(ri.suggest_imports("CompressionIntelligenceEngine")) > 0
)
errors += not test("Entrypoints", len(ri.trace_entrypoints()) > 0)
errors += not test("Complexity hotspots", len(ri.complexity_hotspots(5)) > 0)

# 5. Quality features
qr = QualityScanner(idx.graph, idx.search_index).scan()
errors += not test(
    f"Quality scores ({len(qr.module_scores)} modules)", len(qr.module_scores) > 0
)

# 6. Dead code analysis
dr = DeadCodeAnalyzer(idx.graph, idx.search_index).analyze()
errors += not test(
    f"Dead code analysis ({dr.total_issues()} issues)", dr.total_issues() > 0
)

# 7. Vision system
aa = AvaloniaAnalyzer()
errors += not test("Vision system loads", aa is not None)
ha = HtmlAnalyzer()
errors += not test("Web vision system loads", ha is not None)

# 8. LSP server
from intelligence.lsp_server import LSPDispatcher

dispatcher = LSPDispatcher()
caps = dispatcher.get_capabilities()
errors += not test("LSP capabilities loaded", len(caps) > 0)

# 9. Global files exist
for f in [
    "intelligence-cli",
    "opencode.jsonc",
    "init.sh",
    "instructions/setup.md",
    "tui.json",
    "tools/code-graph.ts",
]:
    errors += not test(
        f"Global ~/.config/opencode/{f} exists",
        os.path.exists(f"/home/mike/.config/opencode/{f}"),
    )

for d in ["agents", "commands", "rules", "tools", "intelligence"]:
    errors += not test(
        f"Global ~/.config/opencode/{d}/ exists",
        os.path.isdir(f"/home/mike/.config/opencode/{d}"),
    )

# 10. Wrapper script works
result = subprocess.run(
    ["/home/mike/.config/opencode/intelligence-cli", "stats"],
    cwd="/home/mike/Documents/Github/SpectralStream",
    capture_output=True,
    text=True,
    timeout=30,
)
errors += not test("Wrapper script produces stats output", "entities" in result.stdout)

print(f"\n{'=' * 50}")
print(f" VALIDATION COMPLETE: {errors} errors")
print(f"{'=' * 50}")
if errors == 0:
    print("✓ All systems operational. Ready for global-only deployment.")
else:
    print(f"✗ {errors} checks failed. Review above.")
