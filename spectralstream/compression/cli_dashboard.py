"""Real-time TUI dashboard for compression progress.

Usage: embed in any compression loop via:
    dashboard = CompressionDashboard(total_tensors=100)
    for name, tensor in tensors:
        result = compress(tensor)
        dashboard.update(name, result)
    dashboard.finish()

Uses Textual framework for full-screen rendering.
Falls back to ANSI-based rendering if textual is unavailable.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────
# Shared helpers (used by both implementations)
# ─────────────────────────────────────────────

_GRADE_COLORS: Dict[str, str] = {
    "S": "green",
    "A": "green",
    "B": "cyan",
    "C": "yellow",
    "D": "red",
    "F": "red",
}


def _format_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    elif n < 1024**2:
        return f"{n / 1024:.1f}KB"
    elif n < 1024**3:
        return f"{n / 1024**2:.1f}MB"
    else:
        return f"{n / 1024**3:.2f}GB"


def _grade_error(err: float) -> str:
    if err <= 0.0002:
        return "S"
    elif err <= 0.001:
        return "A"
    elif err <= 0.005:
        return "B"
    elif err <= 0.01:
        return "C"
    elif err <= 0.05:
        return "D"
    else:
        return "F"


def _grade_color(grade: str) -> str:
    return _GRADE_COLORS.get(grade, "white")


def _method_tier(method_name: str) -> int:
    name = method_name.lower().replace(" ", "_").replace("-", "_")
    if any(x in name for x in ("int8", "int4", "int2", "uniform", "nf4", "delta_int")):
        return 5
    if any(x in name for x in ("hybrid", "cascade", "combined")):
        return 4
    if any(
        x in name
        for x in (
            "arithmetic",
            "ans",
            "huffman",
            "range",
            "zstd",
            "lz4",
            "rans",
            "entropy",
        )
    ):
        return 3
    if any(
        x in name
        for x in (
            "svd",
            "tucker",
            "cp_decompose",
            "kronecker",
            "low_rank",
            "nmf",
            "monarch",
            "butterfly",
            "circulant",
            "block_sparse",
            "einsort",
        )
    ):
        return 2
    return 1


@dataclass
class TensorResult:
    name: str
    method: str
    original_bytes: int
    compressed_bytes: int
    ratio: float
    error: float
    snr: float
    time_s: float


# ─────────────────────────────────────────────
# Try Textual; fall back to ANSI
# ─────────────────────────────────────────────

_HAS_TEXTUAL: bool
try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical
    from textual.widgets import (
        Header,
        Footer,
        Static,
        DataTable,
        RichLog,
        ProgressBar,
        Label,
        Rule,
    )
    from textual.binding import Binding
    from textual.reactive import reactive
    from rich import box
    from rich.bar import Bar
    from rich.panel import Panel
    from rich.style import Style
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align
    from rich.columns import Columns

    _HAS_TEXTUAL = True
except ImportError:
    _HAS_TEXTUAL = False


# ═════════════════════════════════════════════
# ANSI Fallback (same interface)
# ═════════════════════════════════════════════

if not _HAS_TEXTUAL:

    class CompressionDashboard:
        """ANSI-based fallback dashboard."""

        def __init__(
            self,
            total_tensors: int,
            title: str = "SpectralStream Compression Intelligence",
        ):
            self.total = total_tensors
            self.title = title
            self.results: List[TensorResult] = []
            self.start_time = time.time()
            self._last_render = 0.0
            self._min_render_interval = 0.1

        def update(self, name: str, result: Dict[str, Any]) -> None:
            tr = TensorResult(
                name=name,
                method=result.get("method", "unknown"),
                original_bytes=result.get("original_bytes", 0),
                compressed_bytes=result.get("compressed_bytes", 0),
                ratio=result.get("ratio", 0.0),
                error=result.get("error", 1.0),
                snr=result.get("snr", 0.0),
                time_s=result.get("time_s", 0.0),
            )
            self.results.append(tr)
            self._render()

        def _render(self) -> None:
            now = time.time()
            if now - self._last_render < self._min_render_interval:
                return
            self._last_render = now
            elapsed = now - self.start_time
            done = len(self.results)
            remaining = self.total - done
            speed = done / max(elapsed, 0.001)
            eta = remaining / max(speed, 0.001)
            if self.results:
                ratios = [r.ratio for r in self.results]
                errors = [r.error for r in self.results]
                avg_ratio = sum(ratios) / len(ratios)
                avg_error = sum(errors) / len(errors)
                total_orig = sum(r.original_bytes for r in self.results)
                total_comp = sum(r.compressed_bytes for r in self.results)
                overall_ratio = total_orig / max(total_comp, 1)
            else:
                avg_ratio = avg_error = overall_ratio = 0.0
                total_orig = total_comp = 0
            pct = done / max(self.total, 1) * 100
            bar_w = 40
            filled = int(bar_w * done / max(self.total, 1))
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            _R = "\033[0m"
            _B = "\033[1m"
            _CY = "\033[96m"
            _GR = "\033[92m"
            _GY = "\033[90m"
            lines = ["\033[H\033[J"]
            lines.append(f"{_B}{_CY}{self.title}{_R}")
            lines.append(f"{'=' * 60}")
            lines.append(
                f"  {_B}Progress:{_R} [{bar}] {done}/{self.total} ({pct:.1f}%)"
            )
            lines.append(
                f"  {_B}Ratio:{_R} {avg_ratio:.1f}x "
                f"({_GR}{overall_ratio:.1f}x{_R} overall)  "
                f"{_B}Error:{_R} {avg_error:.6f}  "
                f"{_B}Speed:{_R} {speed:.0f} tensors/s  "
                f"{_B}ETA:{_R} {timedelta(seconds=int(eta))}"
            )
            lines.append(
                f"  {_B}Memory:{_R} {_format_size(total_orig)} \u2192 {_format_size(total_comp)} "
                f"({_format_size(total_orig - total_comp)} saved)"
            )
            lines.append("")
            lines.append(
                f"  {_B}{'Tensor':<30} {'Method':<18} {'Ratio':>8} {'Error':>10} {'SNR':>8} {'Grade':>6}{_R}"
            )
            lines.append(f"  {'-' * 80}")
            for r in self.results[-15:]:
                grade = _grade_error(r.error)
                cc = (
                    "\033[92m"
                    if grade in ("S", "A")
                    else "\033[96m"
                    if grade == "B"
                    else "\033[93m"
                    if grade == "C"
                    else "\033[91m"
                )
                name = r.name[:29] if len(r.name) > 29 else r.name
                lines.append(
                    f"  {name:<30} {r.method:<18} {r.ratio:>8.1f}x {r.error:>10.6f} {r.snr:>8.1f} "
                    f"{cc}{grade:>6}{_R}"
                )
            if remaining > 0:
                lines.append(f"\n  {_GY}Compressing... press Ctrl+C to stop{_R}")
            else:
                lines.append(f"\n  {_GR}{_B}\u2713 Complete!{_R}")
            sys.stdout.write("\n".join(lines))
            sys.stdout.flush()

        def finish(self) -> Dict[str, Any]:
            self._render()
            elapsed = time.time() - self.start_time
            if not self.results:
                return {"error": "no results"}
            ratios = [r.ratio for r in self.results]
            errors = [r.error for r in self.results]
            total_orig = sum(r.original_bytes for r in self.results)
            total_comp = sum(r.compressed_bytes for r in self.results)
            summary = {
                "total_tensors": len(self.results),
                "elapsed_s": elapsed,
                "avg_ratio": sum(ratios) / len(ratios),
                "avg_error": sum(errors) / len(errors),
                "overall_ratio": total_orig / max(total_comp, 1),
                "total_original_bytes": total_orig,
                "total_compressed_bytes": total_comp,
                "bytes_saved": total_orig - total_comp,
            }
            _B = "\033[1m"
            _GR = "\033[92m"
            _R = "\033[0m"
            print(f"\n\n{_B}{_GR}Compression Complete{_R}")
            print(f"  {_B}Overall Ratio:{_R} {summary['overall_ratio']:.1f}x")
            print(f"  {_B}Avg Error:{_R} {summary['avg_error']:.6f}")
            print(f"  {_B}Elapsed:{_R} {timedelta(seconds=int(elapsed))}")
            print(
                f"  {_B}Size:{_R} {_format_size(total_orig)} \u2192 {_format_size(total_comp)}"
            )
            return summary


# ═════════════════════════════════════════════
# Textual Dashboard
# ═════════════════════════════════════════════

else:

    class DashboardApp(App):
        """Internal Textual application for the dashboard."""

        TITLE = "SpectralStream"
        SUB_TITLE = "Compression Intelligence Dashboard"

        CSS = """
        Screen {
            layout: vertical;
        }

        #main-row {
            height: 1fr;
            layout: horizontal;
        }

        #right-col {
            width: 1fr;
            layout: vertical;
        }

        #bottom-row {
            height: 14;
            layout: horizontal;
        }

        #tensor-panel {
            width: 42;
            min-width: 42;
            border: solid $primary;
            height: 100%;
        }

        #method-panel {
            height: 1fr;
            border: solid $primary;
        }

        #tier-panel {
            height: 1fr;
            border: solid $primary;
        }

        #cascade-panel {
            width: 1fr;
            border: solid $primary;
        }

        #memory-panel {
            width: 1fr;
            border: solid $primary;
        }

        #topresults-panel {
            width: 1fr;
            border: solid $primary;
        }

        #status-bar {
            dock: bottom;
            height: 3;
            background: $panel;
            padding: 0 1;
        }

        DataTable {
            height: 100%;
        }

        .chart-static {
            height: 100%;
        }
        """

        BINDINGS = [
            Binding(key="ctrl+c", action="quit", description="Quit", priority=True),
            Binding(key="q", action="quit", description="Quit", priority=True),
        ]

        def __init__(
            self,
            total_tensors: int,
            dashboard_title: str,
            start_time: float,
            pending: deque,
            results_list: List[TensorResult],
            error_list: List[str],
            ready_signal: threading.Event,
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)
            self._total = total_tensors
            self._dashboard_title = dashboard_title
            self._start_time = start_time
            self._pending = pending
            self._results = results_list
            self._error_list = error_list
            self._ready_signal = ready_signal
            self._finished = False
            self._completion_summary: Optional[Dict[str, Any]] = None
            self._last_result_count = 0

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="main-row"):
                with Container(id="tensor-panel"):
                    yield DataTable(id="tensor-table")
                with Vertical(id="right-col"):
                    with Container(id="method-panel"):
                        yield Static(id="method-chart")
                    with Container(id="tier-panel"):
                        yield Static(id="tier-chart")
            with Horizontal(id="bottom-row"):
                with Container(id="cascade-panel"):
                    yield Static(id="cascade-chart")
                with Container(id="memory-panel"):
                    yield Static(id="memory-content")
                with Container(id="topresults-panel"):
                    yield Static(id="topresults-content")
            yield Static(id="status-bar")

        def on_mount(self) -> None:
            self._ready_signal.set()
            table = self.query_one("#tensor-table", DataTable)
            table.add_column("Tensor", width=20)
            table.add_column("Method", width=14)
            table.add_column("Ratio", width=7)
            table.add_column("Error", width=10)
            table.add_column("SNR", width=7)
            table.add_column("Grade", width=6)
            self.set_interval(0.5, self._refresh)

        # ── Process pending results ──

        def _process_pending(self) -> None:
            while self._pending:
                try:
                    item = self._pending.popleft()
                except IndexError:
                    break
                if item is None:
                    self._finished = True
                    continue
                name, result = item
                tr = TensorResult(
                    name=name,
                    method=result.get("method", "unknown"),
                    original_bytes=result.get("original_bytes", 0),
                    compressed_bytes=result.get("compressed_bytes", 0),
                    ratio=result.get("ratio", 0.0),
                    error=result.get("error", 1.0),
                    snr=result.get("snr", 0.0),
                    time_s=result.get("time_s", 0.0),
                )
                self._results.append(tr)
                self._add_tensor_row(tr)
                err = result.get("error", 0)
                if err > 0.1:
                    self._error_list.append(name)

        def _add_tensor_row(self, tr: TensorResult) -> None:
            try:
                table = self.query_one("#tensor-table", DataTable)
            except Exception:
                return
            grade = _grade_error(tr.error)
            color = _grade_color(grade)
            ratio_str = f"{tr.ratio:.1f}x" if tr.ratio < 1000 else f"{tr.ratio:.0f}x"
            name = tr.name if len(tr.name) <= 20 else tr.name[:17] + "..."
            table.add_row(
                Text(name),
                Text(tr.method[:14]),
                Text(ratio_str, style=Style(color="green")),
                Text(f"{tr.error:.6f}", style=Style(color=color)),
                Text(f"{tr.snr:.1f}"),
                Text(grade, style=Style(color=color, bold=True)),
            )
            if table.row_count > 200:
                try:
                    table.remove_row(next(iter(table.ordered_rows)).key)
                except Exception:
                    pass

        # ── Periodic refresh ──

        def _refresh(self) -> None:
            self._process_pending()

            elapsed = time.time() - self._start_time
            done = len(self._results)
            remaining = self._total - done
            speed = done / max(elapsed, 0.001)
            eta = remaining / max(speed, 0.001)

            self._update_method_chart(done)
            self._update_tier_chart(done)
            self._update_cascade(done)
            self._update_memory(done, elapsed)
            self._update_top_results()
            self._update_status(done, elapsed, speed, eta)

            if self._finished:
                self._update_all_final()
                self.exit(return_code=0)

        def _update_all_final(self) -> None:
            """Show completion summary on all panels."""
            elapsed = time.time() - self._start_time
            if not self._results:
                self._completion_summary = {"error": "no results"}
                return
            ratios = [r.ratio for r in self._results]
            errors = [r.error for r in self._results]
            total_orig = sum(r.original_bytes for r in self._results)
            total_comp = sum(r.compressed_bytes for r in self._results)
            summary = {
                "total_tensors": len(self._results),
                "elapsed_s": elapsed,
                "avg_ratio": sum(ratios) / len(ratios),
                "avg_error": sum(errors) / len(errors),
                "overall_ratio": total_orig / max(total_comp, 1),
                "total_original_bytes": total_orig,
                "total_compressed_bytes": total_comp,
                "bytes_saved": total_orig - total_comp,
            }
            self._completion_summary = summary
            grades = [_grade_error(r.error) for r in self._results]
            grade_counts = {g: grades.count(g) for g in ("S", "A", "B", "C", "D", "F")}
            grade_table = Table(box=box.ROUNDED, padding=(0, 2))
            grade_table.add_column("Grade", justify="center", style="bold")
            grade_table.add_column("Count", justify="center")
            for g in ("S", "A", "B", "C", "D", "F"):
                gc = _grade_color(g)
                grade_table.add_row(
                    Text(g, style=Style(color=gc, bold=True)),
                    str(grade_counts.get(g, 0)),
                )
            summary_text = Text.assemble(
                (f"Compression Complete!\n\n", "bold green"),
                (f"Overall Ratio:  ", "bold"),
                (f"{summary['overall_ratio']:.1f}x\n", "green bold"),
                (f"Avg Error:      ", "bold"),
                (f"{summary['avg_error']:.6f}\n", "yellow"),
                (f"Total Tensors:  ", "bold"),
                (f"{summary['total_tensors']}\n", "white"),
                (f"Original:       ", "bold"),
                (f"{_format_size(summary['total_original_bytes'])}\n", "white"),
                (f"Compressed:     ", "bold"),
                (f"{_format_size(summary['total_compressed_bytes'])}\n", "white"),
                (f"Saved:          ", "bold"),
                (f"{_format_size(summary['bytes_saved'])}\n", "green"),
                (f"Elapsed:        ", "bold"),
                (f"{timedelta(seconds=int(elapsed))}\n", "white"),
            )
            self.query_one("#method-chart", Static).update("")
            self.query_one("#tier-chart", Static).update(Align.center(grade_table))
            self.query_one("#cascade-chart", Static).update("")
            self.query_one("#memory-content", Static).update(summary_text)
            self._update_status(
                done=len(self._results), elapsed=elapsed, speed=0, eta=0
            )

        # ── Chart builders ──

        def _update_method_chart(self, done: int) -> None:
            widget = self.query_one("#method-chart", Static)
            if done == 0:
                widget.update(
                    Panel(
                        Align.center(Text("No data yet", style="dim")),
                        title="Method Distribution",
                        border_style="blue",
                    )
                )
                return
            counts: Dict[str, int] = defaultdict(int)
            for r in self._results:
                counts[r.method] += 1
            sorted_methods = sorted(counts.items(), key=lambda x: -x[1])[:12]
            max_count = max(c for _, c in sorted_methods) if sorted_methods else 1
            table = Table(box=box.SIMPLE, padding=(0, 1))
            table.add_column("Method", style="cyan", no_wrap=True)
            table.add_column("Count", justify="right", style="bold")
            table.add_column("", width=20)
            bar_colors = [
                "green",
                "yellow",
                "magenta",
                "blue",
                "cyan",
                "red",
                "white",
            ]
            for i, (method, count) in enumerate(sorted_methods):
                color = bar_colors[i % len(bar_colors)]
                mn = method if len(method) <= 14 else method[:11] + "..."
                bar = Bar(size=max_count, begin=0, end=count, width=20, color=color)
                table.add_row(mn, str(count), bar)
            widget.update(
                Panel(table, title="Method Distribution", border_style="blue")
            )

        def _update_tier_chart(self, done: int) -> None:
            widget = self.query_one("#tier-chart", Static)
            if done == 0:
                widget.update(
                    Panel(
                        Align.center(Text("No data yet", style="dim")),
                        title="Tier Breakdown",
                        border_style="green",
                    )
                )
                return
            tier_counts: Dict[int, int] = defaultdict(int)
            for r in self._results:
                tier_counts[_method_tier(r.method)] += 1
            tier_labels = {
                1: "Tier 1",
                2: "Tier 2",
                3: "Tier 3",
                4: "Tier 4",
                5: "Tier 5",
            }
            tier_colors = {
                1: "green",
                2: "cyan",
                3: "yellow",
                4: "magenta",
                5: "red",
            }
            max_count = max(tier_counts.values()) if tier_counts else 1
            table = Table(box=box.SIMPLE, padding=(0, 1))
            table.add_column("Tier", style="bold", width=7)
            table.add_column("Count", justify="right", style="bold")
            table.add_column("", width=20)
            for t in range(1, 6):
                count = tier_counts.get(t, 0)
                color = tier_colors[t]
                bar = Bar(size=max_count, begin=0, end=count, width=20, color=color)
                table.add_row(tier_labels[t], str(count), bar)
            widget.update(Panel(table, title="Tier Breakdown", border_style="green"))

        CASCADE_STAGE_LABELS = [
            "Profile",
            "Allocate",
            "Select",
            "Compress",
            "Validate",
        ]

        def _update_cascade(self, done: int) -> None:
            widget = self.query_one("#cascade-chart", Static)
            total = self._total
            pipe: List[Text] = []
            for i, stage in enumerate(self.CASCADE_STAGE_LABELS):
                if i == 0:
                    color = "green" if done > 0 else "dim"
                    status = "\u2713" if done > 0 else "\u25cb"
                elif i == 1:
                    color = "green" if done > 0 else "dim"
                    status = "\u2713" if done > 0 else "\u25cb"
                elif i == 2:
                    color = "green" if done > 0 else "dim"
                    status = "\u2713" if done > 0 else "\u25cb"
                elif i == 3:
                    if done >= total:
                        color = "green"
                        status = "\u2713"
                    elif done > 0:
                        color = "yellow bold"
                        status = "\u25b6 ACTIVE"
                    else:
                        color = "dim"
                        status = "\u25cb PENDING"
                else:
                    if done >= total:
                        color = "green"
                        status = "\u2713"
                    else:
                        color = "dim"
                        status = "\u25cb PENDING"
                if i > 0:
                    pipe.append(Text("     \u2502"))
                pipe.append(Text(f"  {status} {stage}", style=color))
            pipeline_text = Text("\n").join(pipe)
            widget.update(
                Panel(
                    Align.center(pipeline_text),
                    title="Cascade Stage",
                    border_style="yellow",
                )
            )

        def _update_memory(self, done: int, elapsed: float) -> None:
            widget = self.query_one("#memory-content", Static)
            if done == 0:
                widget.update(
                    Panel(
                        Align.center(Text("Waiting for data...", style="dim")),
                        title="Memory Usage",
                        border_style="magenta",
                    )
                )
                return
            total_orig = sum(r.original_bytes for r in self._results)
            total_comp = sum(r.compressed_bytes for r in self._results)
            saved = total_orig - total_comp
            overall_ratio = total_orig / max(total_comp, 1)
            saved_pct = (saved / max(total_orig, 1)) * 100
            table = Table(box=box.SIMPLE, padding=(0, 1))
            table.add_column("Metric", style="bold")
            table.add_column("Value", justify="right")
            table.add_row("Original", f"{_format_size(total_orig)}")
            table.add_row("Compressed", f"{_format_size(total_comp)}")
            table.add_row("Saved", f"{_format_size(saved)} ({saved_pct:.1f}%)")
            table.add_row("Ratio", f"{overall_ratio:.1f}x")
            progress_pct = min(overall_ratio / 10000.0, 1.0)
            bar = Bar(
                size=100,
                begin=0,
                end=int(progress_pct * 100),
                width=20,
                color="magenta",
            )
            table.add_row("Progress", bar)
            widget.update(
                Panel(Align.center(table), title="Memory Usage", border_style="magenta")
            )

        def _update_top_results(self) -> None:
            widget = self.query_one("#topresults-content", Static)
            if len(self._results) < 1:
                widget.update(
                    Panel(
                        Align.center(Text("No data yet", style="dim")),
                        title="Top Results",
                        border_style="cyan",
                    )
                )
                return
            top_ratio = sorted(self._results, key=lambda r: -r.ratio)[:5]
            top_low_error = sorted(self._results, key=lambda r: r.error)[:5]
            table = Table(box=box.SIMPLE, padding=(0, 1))
            table.add_column("Top Ratio", style="green bold", width=18)
            table.add_column("Ratio", justify="right", style="bold", width=8)
            table.add_column("Top Error", style="cyan bold", width=18)
            table.add_column("Error", justify="right", width=10)
            max_rows = max(len(top_ratio), len(top_low_error))
            for i in range(max_rows):
                left_name = ""
                left_ratio = ""
                right_name = ""
                right_error = ""
                if i < len(top_ratio):
                    r = top_ratio[i]
                    left_name = r.name if len(r.name) <= 16 else r.name[:13] + "..."
                    left_ratio = f"{r.ratio:.1f}x"
                if i < len(top_low_error):
                    r = top_low_error[i]
                    right_name = r.name if len(r.name) <= 16 else r.name[:13] + "..."
                    right_error = f"{r.error:.6f}"
                table.add_row(left_name, left_ratio, right_name, right_error)
            widget.update(Panel(table, title="Top Results", border_style="cyan"))

        def _update_status(
            self, done: int, elapsed: float, speed: float, eta: float
        ) -> None:
            widget = self.query_one("#status-bar", Static)
            pct = done / max(self._total, 1) * 100
            eta_str = str(timedelta(seconds=int(eta)))
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            total_orig = (
                sum(r.original_bytes for r in self._results) if self._results else 0
            )
            total_comp = (
                sum(r.compressed_bytes for r in self._results) if self._results else 0
            )
            overall_ratio = total_orig / max(total_comp, 1) if total_comp else 0.0
            bar_filled = int(pct / 5)
            bar_str = "=" * bar_filled + " " * (20 - bar_filled)
            text = Text.assemble(
                (f"  ETA: {eta_str}  ", "bold cyan"),
                (f"|  Speed: {speed:.1f} t/s  ", "white"),
                (f"|  Elapsed: {elapsed_str}  ", "white"),
                (f"|  Ratio: {overall_ratio:.1f}x  ", "bold green"),
                (f"|  ", "white"),
                (f"[{bar_str}]", "yellow"),
                (f"  {done}/{self._total} ({pct:.1f}%)", "bold"),
            )
            widget.update(text)

    # ─────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────

    class CompressionDashboard:
        """Textual-based full-screen TUI dashboard.

        Thread-safe: call update() and finish() from any thread.
        The Textual app runs in a background daemon thread with signal.signal
        patched to avoid ValueError (signal handlers require main thread).
        """

        def __init__(
            self,
            total_tensors: int,
            title: str = "SpectralStream Compression Intelligence",
        ):
            self._total = total_tensors
            self._title = title
            self._start_time = time.time()
            self._pending: deque = deque()
            self._results: List[TensorResult] = []
            self._error_list: List[str] = []
            self._app: Optional[DashboardApp] = None
            self._thread: Optional[threading.Thread] = None
            self._started = False

        def _ensure_app(self) -> None:
            if self._app is not None:
                return
            ready = threading.Event()
            self._app = DashboardApp(
                total_tensors=self._total,
                dashboard_title=self._title,
                start_time=self._start_time,
                pending=self._pending,
                results_list=self._results,
                error_list=self._error_list,
                ready_signal=ready,
            )

            def _run_textual():
                import signal as _sig

                _orig = _sig.signal
                _sig.signal = lambda *a, **kw: None
                try:
                    self._app.run()
                finally:
                    _sig.signal = _orig

            self._thread = threading.Thread(target=_run_textual, daemon=True)
            self._thread.start()
            if not ready.wait(timeout=15):
                self._app = None
                self._thread = None
                raise RuntimeError("Dashboard failed to start within 15 seconds")

        def update(self, name: str, result: Dict[str, Any]) -> None:
            if not self._started:
                self._ensure_app()
                self._started = True
            self._pending.append((name, result))

        def finish(self) -> Dict[str, Any]:
            elapsed = time.time() - self._start_time
            if self._app is not None:
                self._pending.append(None)
                if self._thread is not None:
                    self._thread.join(timeout=15)
                if self._app._completion_summary:
                    return self._app._completion_summary
            if not self._results:
                return {"error": "no results"}
            ratios = [r.ratio for r in self._results]
            errors = [r.error for r in self._results]
            total_orig = sum(r.original_bytes for r in self._results)
            total_comp = sum(r.compressed_bytes for r in self._results)
            return {
                "total_tensors": len(self._results),
                "elapsed_s": elapsed,
                "avg_ratio": sum(ratios) / len(ratios),
                "avg_error": sum(errors) / len(errors),
                "overall_ratio": total_orig / max(total_comp, 1),
                "total_original_bytes": total_orig,
                "total_compressed_bytes": total_comp,
                "bytes_saved": total_orig - total_comp,
            }
