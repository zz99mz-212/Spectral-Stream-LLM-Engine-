from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.benchmark.benchmark_runner import (
    BenchmarkRunResult,
    BenchmarkTensorResult,
    TARGET_RATIOS,
)
from spectralstream.compression.benchmark.loss_calculator import (
    ModelLossMetrics,
    TensorLossMetrics,
)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


def _human_size(n: int) -> str:
    nf = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if nf < 1024:
            return f"{nf:.1f}{unit}"
        nf /= 1024
    return f"{nf:.1f}TB"


def _format_snr(snr: float) -> str:
    return "∞" if snr == float("inf") or math.isinf(snr) else f"{snr:.1f}"


def _grade_from_error(err: float) -> str:
    if err < 0.0002:
        return "S"
    if err < 0.001:
        return "A"
    if err < 0.005:
        return "B"
    if err < 0.01:
        return "C"
    if err < 0.05:
        return "D"
    return "F"


class ReportGenerator:
    def __init__(self):
        self._console = Console() if _HAS_RICH else None

    def generate_rich_report(
        self, result: BenchmarkRunResult, per_type: bool = True
    ) -> str:
        if not _HAS_RICH:
            return self._text_report(result, per_type)

        lines: List[str] = []
        console = self._console

        title = f"Benchmark: {os.path.basename(result.model_name)}"
        console.print(Panel(Text(title, style="bold cyan"), box=box.ROUNDED))

        summary = Table(box=box.SIMPLE)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="green")
        summary.add_row("Tensors", str(result.num_tensors))
        summary.add_row("Failures", str(result.num_failures))
        summary.add_row("Run Time", f"{result.run_time:.2f}s")
        summary.add_row(
            "Ratios Tested", ", ".join(f"{r}x" for r in result.target_ratios)
        )
        console.print(summary)

        if per_type and result.per_type_results:
            self._print_per_type_table(console, result)

        if result.per_method_results:
            self._print_method_table(console, result)

        if result.cascade_results:
            self._print_cascade_table(console, result)

        if result.streaming_results:
            self._print_streaming_table(console, result)

        return "\n".join(lines)

    def generate_text_report(self, result: BenchmarkRunResult) -> str:
        return self._text_report(result, per_type=True)

    def _text_report(self, result: BenchmarkRunResult, per_type: bool = True) -> str:
        lines = [
            "=" * 72,
            f"  Benchmark: {os.path.basename(result.model_name)}",
            f"  Ratios: {', '.join(f'{r}x' for r in result.target_ratios)}",
            f"  Tensors: {result.num_tensors}  Failures: {result.num_failures}",
            f"  Run Time: {result.run_time:.2f}s",
            "=" * 72,
        ]

        if per_type and result.per_type_results:
            lines.append("")
            lines.append("Per-Tensor-Type Results:")
            lines.append("-" * 72)
            header = f"{'Type':<16} {'Ratio':<8} {'Method':<20} {'RelErr':<10} {'SNR':<10} {'Grade':<6} {'Time':<8}"
            lines.append(header)
            lines.append("-" * 72)
            for ttype in sorted(result.per_type_results.keys()):
                for tr in sorted(result.per_type_results[ttype].keys()):
                    tresults = result.per_type_results[ttype][tr]
                    avg_err = float(
                        np.mean([r.metrics.relative_error for r in tresults])
                    )
                    avg_snr = float(
                        np.mean(
                            [
                                r.metrics.snr_db
                                for r in tresults
                                if r.metrics.snr_db != float("inf")
                            ]
                        )
                    )
                    avg_time = float(np.mean([r.compression_time for r in tresults]))
                    methods = set(r.method for r in tresults)
                    method_str = ", ".join(sorted(methods))[:20]
                    grade = _grade_from_error(avg_err)
                    lines.append(
                        f"{ttype:<16} {tr:<8}x {method_str:<20} {avg_err:<10.6f} {avg_snr:<10.1f} {grade:<6} {avg_time * 1000:<8.1f}ms"
                    )

        if result.per_method_results:
            lines.append("")
            lines.append("Per-Method Results:")
            lines.append("-" * 72)
            for method in sorted(result.per_method_results.keys()):
                for tr in sorted(result.per_method_results[method].keys()):
                    mresults = result.per_method_results[method][tr]
                    avg_err = float(
                        np.mean([r.metrics.relative_error for r in mresults])
                    )
                    avg_ratio = float(np.mean([r.achieved_ratio for r in mresults]))
                    avg_snr = float(
                        np.mean(
                            [
                                r.metrics.snr_db
                                for r in mresults
                                if r.metrics.snr_db != float("inf")
                            ]
                        )
                    )
                    lines.append(
                        f"  {method:<20} @ {tr:<6}x: ratio={avg_ratio:<8.1f} err={avg_err:.6f} snr={avg_snr:.1f}dB"
                    )

        return "\n".join(lines)

    def _print_per_type_table(self, console, result: BenchmarkRunResult) -> None:
        table = Table(title="Per-Tensor-Type Results", box=box.ROUNDED)
        table.add_column("Type", style="cyan")
        table.add_column("Ratio", justify="right")
        table.add_column("Method", style="yellow")
        table.add_column("RelErr", justify="right")
        table.add_column("SNR dB", justify="right")
        table.add_column("Grade")
        table.add_column("CosSim", justify="right")
        table.add_column("Outlier%", justify="right")
        table.add_column("Time", justify="right")

        for ttype in sorted(result.per_type_results.keys()):
            for tr in sorted(result.per_type_results[ttype].keys()):
                tresults = result.per_type_results[ttype][tr]
                avg_err = float(np.mean([r.metrics.relative_error for r in tresults]))
                avg_snr = float(
                    np.mean(
                        [
                            r.metrics.snr_db
                            for r in tresults
                            if r.metrics.snr_db != float("inf")
                        ]
                    )
                )
                avg_cos = float(
                    np.mean([r.metrics.cosine_similarity for r in tresults])
                )
                avg_out = float(
                    np.mean([r.metrics.outlier_preservation_ratio for r in tresults])
                )
                avg_time = float(np.mean([r.compression_time for r in tresults]))
                methods = set(r.method for r in tresults)
                method_str = ", ".join(sorted(methods))[:20]
                grade = _grade_from_error(avg_err)
                grade_color = {
                    "S": "green",
                    "A": "cyan",
                    "B": "yellow",
                    "C": "orange3",
                    "D": "red",
                    "F": "bold red",
                }.get(grade, "white")

                table.add_row(
                    ttype,
                    f"{tr}x",
                    method_str,
                    f"{avg_err:.6f}",
                    f"{avg_snr:.1f}",
                    Text(grade, style=grade_color),
                    f"{avg_cos:.4f}",
                    f"{avg_out:.1%}",
                    f"{avg_time * 1000:.1f}ms",
                )
        console.print(table)

    def _print_method_table(self, console, result: BenchmarkRunResult) -> None:
        table = Table(title="Per-Method Results", box=box.ROUNDED)
        table.add_column("Method", style="cyan")
        table.add_column("Ratio Target", justify="right")
        table.add_column("Avg Ratio", justify="right")
        table.add_column("Avg RelErr", justify="right")
        table.add_column("Avg SNR", justify="right")
        table.add_column("Avg CosSim", justify="right")
        table.add_column("Best Grade")

        for method in sorted(result.per_method_results.keys()):
            for tr in sorted(result.per_method_results[method].keys()):
                mresults = result.per_method_results[method][tr]
                avg_err = float(np.mean([r.metrics.relative_error for r in mresults]))
                avg_ratio = float(np.mean([r.achieved_ratio for r in mresults]))
                avg_snr = float(
                    np.mean(
                        [
                            r.metrics.snr_db
                            for r in mresults
                            if r.metrics.snr_db != float("inf")
                        ]
                    )
                )
                avg_cos = float(
                    np.mean([r.metrics.cosine_similarity for r in mresults])
                )
                best_grade = _grade_from_error(avg_err)

                table.add_row(
                    method,
                    f"{tr}x",
                    f"{avg_ratio:.1f}x",
                    f"{avg_err:.6f}",
                    f"{avg_snr:.1f}",
                    f"{avg_cos:.4f}",
                    best_grade,
                )
        console.print(table)

    def _print_cascade_table(self, console, result: BenchmarkRunResult) -> None:
        table = Table(title="Cascade Results", box=box.ROUNDED)
        table.add_column("Pattern", style="cyan")
        table.add_column("Target", justify="right")
        table.add_column("Achieved", justify="right")
        table.add_column("RelErr", justify="right")
        table.add_column("SNR", justify="right")
        table.add_column("Time", justify="right")

        for pattern, cresults in result.cascade_results.items():
            for cr in cresults:
                table.add_row(
                    pattern,
                    f"{cr.target_ratio}x",
                    f"{cr.achieved_ratio:.1f}x",
                    f"{cr.metrics.relative_error:.6f}",
                    f"{cr.metrics.snr_db:.1f}",
                    f"{cr.compression_time * 1000:.1f}ms",
                )
        console.print(table)

    def _print_streaming_table(self, console, result: BenchmarkRunResult) -> None:
        table = Table(title="Streaming vs RAM Comparison", box=box.ROUNDED)
        table.add_column("Mode", style="cyan")
        table.add_column("Ratio", justify="right")
        table.add_column("RelErr", justify="right")
        table.add_column("SNR", justify="right")
        table.add_column("Peak Mem", justify="right")
        table.add_column("Time", justify="right")

        for name, sr in result.streaming_results.items():
            table.add_row(
                name,
                f"{sr.achieved_ratio:.1f}x",
                f"{sr.metrics.relative_error:.6f}",
                f"{sr.metrics.snr_db:.1f}",
                f"{sr.streaming_peak_memory_mb:.1f}MB",
                f"{sr.compression_time * 1000:.1f}ms",
            )
        console.print(table)

    def generate_json_report(self, result: BenchmarkRunResult, path: str = "") -> str:
        report = self._build_report_dict(result)
        json_str = json.dumps(report, indent=2, default=str)
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(json_str)
        return json_str

    def generate_html_report(self, result: BenchmarkRunResult, path: str = "") -> str:
        report = self._build_report_dict(result)
        html = self._build_html(report, result)
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(html)
        return html

    def _build_report_dict(self, result: BenchmarkRunResult) -> Dict[str, Any]:
        per_type_summary: Dict[str, Any] = {}
        for ttype in sorted(result.per_type_results.keys()):
            per_type_summary[ttype] = {}
            for tr in sorted(result.per_type_results[ttype].keys()):
                tresults = result.per_type_results[ttype][tr]
                if not tresults:
                    continue
                avg_err = float(np.mean([r.metrics.relative_error for r in tresults]))
                avg_ratio = float(np.mean([r.achieved_ratio for r in tresults]))
                snr_vals = [
                    r.metrics.snr_db
                    for r in tresults
                    if r.metrics.snr_db != float("inf")
                ]
                avg_snr = float(np.mean(snr_vals)) if snr_vals else 0.0
                avg_cos = float(
                    np.mean([r.metrics.cosine_similarity for r in tresults])
                )
                avg_time = float(np.mean([r.compression_time for r in tresults]))
                methods = list(set(r.method for r in tresults))
                per_type_summary[ttype][str(tr)] = {
                    "count": len(tresults),
                    "avg_ratio": round(avg_ratio, 2),
                    "avg_error": round(avg_err, 6),
                    "avg_snr_db": round(avg_snr, 2),
                    "avg_cosine_similarity": round(avg_cos, 4),
                    "avg_time_ms": round(avg_time * 1000, 2),
                    "methods": methods,
                    "grade": _grade_from_error(avg_err),
                }

        per_method_summary: Dict[str, Any] = {}
        for method in sorted(result.per_method_results.keys()):
            per_method_summary[method] = {}
            for tr in sorted(result.per_method_results[method].keys()):
                mresults = result.per_method_results[method][tr]
                avg_err = float(np.mean([r.metrics.relative_error for r in mresults]))
                avg_ratio = float(np.mean([r.achieved_ratio for r in mresults]))
                avg_snr = float(
                    np.mean(
                        [
                            r.metrics.snr_db
                            for r in mresults
                            if r.metrics.snr_db != float("inf")
                        ]
                    )
                )
                per_method_summary[method][str(tr)] = {
                    "count": len(mresults),
                    "avg_ratio": round(avg_ratio, 2),
                    "avg_error": round(avg_err, 6),
                    "avg_snr_db": round(avg_snr, 2),
                }

        pareto_frontier = self._compute_pareto_frontier(result)

        return {
            "model": result.model_name,
            "timestamp": datetime.now().isoformat(),
            "target_ratios": result.target_ratios,
            "num_tensors": result.num_tensors,
            "num_failures": result.num_failures,
            "run_time_seconds": round(result.run_time, 3),
            "per_type": per_type_summary,
            "per_method": per_method_summary,
            "pareto_frontier": pareto_frontier,
        }

    def _compute_pareto_frontier(
        self, result: BenchmarkRunResult
    ) -> List[Dict[str, Any]]:
        points: List[Tuple[float, float, str, float, str]] = []
        for method in result.per_method_results:
            for tr in result.per_method_results[method]:
                mresults = result.per_method_results[method][tr]
                if not mresults:
                    continue
                avg_ratio = float(np.mean([r.achieved_ratio for r in mresults]))
                avg_err = float(np.mean([r.metrics.relative_error for r in mresults]))
                points.append(
                    (avg_ratio, avg_err, method, tr, _grade_from_error(avg_err))
                )

        if not points:
            return []

        pareto: List[Dict[str, Any]] = []
        points.sort(key=lambda x: -x[0])
        min_err = float("inf")
        for ratio, err, method, tr, grade in points:
            if err < min_err:
                min_err = err
                pareto.append(
                    {
                        "method": method,
                        "target_ratio": tr,
                        "achieved_ratio": round(ratio, 2),
                        "error": round(err, 6),
                        "grade": grade,
                    }
                )
        return pareto

    def _build_html(self, report: Dict[str, Any], result: BenchmarkRunResult) -> str:
        ts = report["timestamp"]
        pt = report["per_type"]
        pm = report["per_method"]
        pf = report["pareto_frontier"]

        per_type_rows = ""
        for ttype in sorted(pt.keys()):
            for tr_str in sorted(pt[ttype].keys()):
                info = pt[ttype][tr_str]
                g = info["grade"]
                per_type_rows += f"""
            <tr>
              <td>{ttype}</td>
              <td>{tr_str}x</td>
              <td>{info["avg_ratio"]:.1f}x</td>
              <td>{info["avg_error"]:.6f}</td>
              <td>{info["avg_snr_db"]:.1f}</td>
              <td style="color: {"#00ff88" if g == "S" else "#00cc66" if g == "A" else "#ffd700" if g == "B" else "#ff8c00" if g == "C" else "#ff4444"}">{g}</td>
              <td>{info["avg_cosine_similarity"]:.4f}</td>
              <td>{", ".join(info["methods"][:3])}</td>
            </tr>"""

        pareto_rows = ""
        for i, p in enumerate(pf, 1):
            pareto_rows += f"""
            <tr>
              <td>{i}</td>
              <td>{p["method"]}</td>
              <td>{p["target_ratio"]}</td>
              <td>{p["achieved_ratio"]:.1f}x</td>
              <td>{p["error"]:.6f}</td>
              <td style="color: {"#00ff88" if p["grade"] == "S" else "#00cc66" if p["grade"] == "A" else "#ffd700" if p["grade"] == "B" else "#ff8c00"}">{p["grade"]}</td>
            </tr>"""

        method_chart_entries = []
        for method in sorted(pm.keys()):
            points = []
            for tr_str in sorted(pm[method].keys()):
                info = pm[method][tr_str]
                pts = (
                    "{"
                    + f"x:{tr_str}, y:{info['avg_error']:.6f}, ratio:{info['avg_ratio']:.1f}"
                    + "}"
                )
                points.append(pts)
            if points:
                data_str = ",".join(points)
                entry = "{" + f"method:'{method}', data:[{data_str}]" + "}"
                method_chart_entries.append(entry)

        method_chart_data = ",".join(method_chart_entries)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Compression Benchmark — {ts}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          max-width: 1400px; margin: 0 auto; padding: 20px; background: #0a0a0f; color: #e0e0e0; }}
  h1, h2, h3 {{ color: #ffffff; }}
  .header {{ text-align: center; padding: 40px;
             background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
             border-radius: 16px; margin-bottom: 30px; }}
  .header h1 {{ font-size: 2.5em; margin: 0; }}
  .badge-container {{ display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
  .badge {{ background: #1a1a2e; border: 1px solid #333; border-radius: 12px; padding: 20px 30px; text-align: center; min-width: 150px; }}
  .badge .value {{ font-size: 2em; font-weight: bold; color: #00ff88; }}
  .badge .label {{ font-size: 0.85em; color: #8888ff; margin-top: 5px; }}
  .section {{ background: #1a1a2e; border-radius: 12px; padding: 25px; margin: 20px 0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 10px 15px; text-align: left; border-bottom: 1px solid #333; }}
  th {{ color: #8888ff; font-weight: 600; text-transform: uppercase; font-size: 0.85em; }}
  .chart-container {{ height: 400px; margin: 20px 0; }}
  .footer {{ text-align: center; color: #666; margin-top: 40px; padding: 20px; }}
</style>
</head>
<body>
<div class="header">
  <h1>Compression Benchmark Report</h1>
  <p style="color: #888;">{ts}</p>
  <p style="color: #aaa;">Model: {report["model"]}</p>
</div>
<div class="badge-container">
  <div class="badge"><div class="value">{report["num_tensors"]}</div><div class="label">Tensors</div></div>
  <div class="badge"><div class="value">{report["run_time_seconds"]:.1f}s</div><div class="label">Run Time</div></div>
  <div class="badge"><div class="value">{report["num_failures"]}</div><div class="label">Failures</div></div>
</div>
<div class="section">
  <h2>Per-Tensor-Type Results</h2>
  <table>
    <tr><th>Type</th><th>Target</th><th>Avg Ratio</th><th>Avg Error</th><th>Avg SNR</th><th>Grade</th><th>CosSim</th><th>Methods</th></tr>
    {per_type_rows}
  </table>
</div>
<div class="section">
  <h2>Pareto Frontier</h2>
  <p style="color: #888;">Best error at each ratio level (dominant methods)</p>
  <table>
    <tr><th>Rank</th><th>Method</th><th>Target</th><th>Ratio</th><th>Error</th><th>Grade</th></tr>
    {pareto_rows}
  </table>
</div>
<div class="section">
  <h2>Rate-Distortion Curves</h2>
  <div class="chart-container">
    <canvas id="rdChart"></canvas>
  </div>
  <div class="chart-container">
    <canvas id="snrChart"></canvas>
  </div>
</div>
<div class="footer">
  <p>Generated by SpectralStream Benchmark System at {ts}</p>
</div>
<script>
const colors = ['#00ff88','#00cc66','#ffd700','#ff8c00','#ff4444','#8888ff','#ff69b4','#00ffff'];
const rdCtx = document.getElementById('rdChart').getContext('2d');
const snrCtx = document.getElementById('snrChart').getContext('2d');

const datasets = [{method_chart_data}];
const rdDatasets = [];
const snrDatasets = [];
const ratioLabels = [{",".join(f"'{r}x'" for r in sorted(report["target_ratios"]))}];
const ratioValues = [{",".join(str(r) for r in sorted(report["target_ratios"]))}];

datasets.forEach((ds, i) => {{
  const c = colors[i % colors.length];
  const sortedData = ds.data.sort((a,b) => a.x - b.x);
  rdDatasets.push({{
    label: ds.method,
    data: sortedData.map(p => ({{x: p.x, y: p.error}})),
    borderColor: c,
    backgroundColor: c + '33',
    fill: false,
    tension: 0.4,
    pointRadius: 4,
  }});
}});

new Chart(rdCtx, {{
  type: 'scatter',
  data: {{ datasets: rdDatasets }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    scales: {{
      x: {{ type: 'logarithmic', title: {{ display: true, text: 'Target Ratio', color: '#888' }},
             grid: {{ color: '#333' }}, ticks: {{ color: '#888' }} }},
      y: {{ title: {{ display: true, text: 'Error (relative)', color: '#888' }},
             type: 'logarithmic',
             grid: {{ color: '#333' }}, ticks: {{ color: '#888' }} }},
    }},
    plugins: {{
      legend: {{ labels: {{ color: '#e0e0e0' }} }},
      title: {{ display: true, text: 'Rate-Distortion: Target Ratio vs Error', color: '#fff' }},
    }},
  }},
}});
</script>
</body>
</html>"""
        return html

    def generate_all(
        self, result: BenchmarkRunResult, output_dir: str, prefix: str = "benchmark"
    ) -> Dict[str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(output_dir) / f"{prefix}_report_{ts}"
        report_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}

        json_path = str(report_dir / f"{prefix}_report.json")
        self.generate_json_report(result, json_path)
        paths["json"] = json_path

        html_path = str(report_dir / f"{prefix}_report.html")
        self.generate_html_report(result, html_path)
        paths["html"] = html_path

        txt_path = str(report_dir / f"{prefix}_report.txt")
        txt_content = self._text_report(result, per_type=True)
        with open(txt_path, "w") as f:
            f.write(txt_content)
        paths["txt"] = txt_path

        return paths
