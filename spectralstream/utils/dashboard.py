"""
Real-time Web Dashboard for SpectralStream.

Provides:
- Live inference metrics (tokens/sec, model calls, acceptance rate)
- Per-strategy breakdown (forwardless vs block vs speculative vs standard)
- KV cache statistics (size, hit rate, compression ratio)
- HDC engine statistics (targets, confidence, accuracy)
- Memory usage per component
- SSD streaming stats
- Confidence gate performance
- Online learning progress
- Real-time token stream visualization
- Configuration viewer/editor

All data comes from InferenceMonitor and engine stats.
Serves as a single-page web app with auto-refreshing JSON API.
"""

import json
import time
import http.server
import urllib.parse
import threading
import sys
import os
from typing import Optional


class RESTAPIHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for dashboard and API endpoints."""

    server_version = "SpectralStreamDashboard/0.1.0"

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body.encode("utf-8"))))
        self._send_cors()
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _send_html(self, html: str, status: int = 200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400):
        self._send_json({"error": message}, status)

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _get_dashboard(self):
        return self.server.dashboard

    def _parse_body(self) -> Optional[dict]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[Dashboard] {args[0]} {args[1]} {args[2]}\n")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = urllib.parse.parse_qs(parsed.query)

        if path == "/dashboard" or path == "/dashboard/" or path == "/":
            self._send_html(self._get_dashboard()._generate_html())
        elif path == "/api/stats":
            self._send_json(self._get_dashboard().get_stats())
        elif path == "/api/config":
            self._send_json(self._get_dashboard().get_config())
        elif path == "/api/strategies":
            self._send_json(self._get_dashboard().get_strategy_breakdown())
        elif path == "/api/memory":
            self._send_json(self._get_dashboard().get_memory_usage())
        elif path == "/api/hdc":
            self._send_json(self._get_dashboard().get_hdc_stats())
        elif path == "/api/kv":
            self._send_json(self._get_dashboard().get_kv_stats())
        elif path == "/api/learning":
            self._send_json(self._get_dashboard().get_learning_stats())
        elif path == "/api/confidence":
            self._send_json(self._get_dashboard().get_confidence_stats())
        else:
            self._send_error(f"Not found: {path}", 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/config":
            body = self._parse_body()
            if body:
                result = self._get_dashboard().update_config(body)
                self._send_json({"status": "ok", "updated": result})
            else:
                self._send_error("Invalid JSON body")
        else:
            self._send_error(f"Not found: {path}", 404)


class DashboardServer:
    """
    Lightweight web dashboard server.

    Serves:
    - GET /dashboard/ -> HTML dashboard page
    - GET /api/stats -> JSON metrics
    - GET /api/stream -> SSE event stream for real-time updates
    - POST /api/config -> Update configuration
    """

    def __init__(
        self, orchestrator, monitor, host: str = "127.0.0.1", port: int = 8080
    ):
        self.orchestrator = orchestrator
        self.monitor = monitor
        self.host = host
        self.port = port
        self._server = None
        self._thread = None
        self._running = False
        self._config_cache = {}
        self._last_stats_time = 0

    def get_stats(self) -> dict:
        stats = {}
        if self.orchestrator:
            try:
                stats = self.orchestrator.stats()
            except Exception:
                pass
        if self.monitor:
            try:
                mon = self.monitor.get_stats()
                stats["monitor"] = mon
            except Exception:
                pass
        stats["timestamp"] = time.time()
        return stats

    def get_config(self) -> dict:
        return self._config_cache

    def get_strategy_breakdown(self) -> dict:
        if self.monitor:
            return self.monitor.strategy_breakdown()
        return {}

    def get_memory_usage(self) -> dict:
        if self.monitor:
            return self.monitor.estimated_memory_bytes()
        return {}

    def get_hdc_stats(self) -> dict:
        if self.orchestrator:
            eng = getattr(self.orchestrator, "hd_engine", None)
            if eng:
                return {
                    "acceptance_rate": eng.acceptance_rate(),
                    "draft_count": getattr(eng, "draft_count", 0),
                    "accept_count": getattr(eng, "accept_count", 0),
                    "total_predictions": getattr(eng, "total_predictions", 0),
                }
        return {}

    def get_kv_stats(self) -> dict:
        if self.orchestrator:
            kv = getattr(self.orchestrator, "kv_cache", None)
            if kv:
                return {
                    "hit_rate": kv.hit_rate(),
                    "compression_ratio": kv.compression_ratio(),
                    "size": getattr(kv, "size", 0),
                    "max_size": getattr(kv, "max_size", 0),
                }
        return {}

    def get_learning_stats(self) -> dict:
        if self.orchestrator:
            learn = getattr(self.orchestrator, "learning_engine", None)
            if learn:
                try:
                    return learn.get_stats()
                except Exception:
                    pass
        return {}

    def get_confidence_stats(self) -> dict:
        if self.orchestrator:
            gate = getattr(self.orchestrator, "confidence_gate", None)
            if gate:
                return {
                    "accuracy": self.monitor.confidence_gate_accuracy()
                    if self.monitor
                    else 0,
                    "predictions": self.monitor.confidence_gate_total
                    if self.monitor
                    else 0,
                    "correct": self.monitor.confidence_gate_correct
                    if self.monitor
                    else 0,
                }
        return {}

    def update_config(self, updates: dict) -> list:
        updated = []
        for key, value in updates.items():
            self._config_cache[key] = value
            updated.append(key)
        return updated

    def _generate_html(self) -> str:
        return _DASHBOARD_HTML

    def start(self):
        """Start dashboard server in background thread."""
        if self._running:
            return

        self._server = http.server.HTTPServer((self.host, self.port), RESTAPIHandler)
        self._server.dashboard = self

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="dashboard-server",
        )
        self._thread.start()
        self._running = True
        print(f"[Dashboard] Web dashboard at http://{self.host}:{self.port}/dashboard/")

    def stop(self):
        """Stop dashboard server."""
        self._running = False
        if self._server:
            self._server.shutdown()


class ConsoleDashboard:
    """
    Terminal-based real-time dashboard using ANSI escape codes.

    Shows live stats in the terminal. Updates every second.
    Use with run.py --dashboard flag.
    """

    def __init__(self, monitor):
        self.monitor = monitor
        self._running = False
        self._thread = None

    def render(self) -> str:
        s = self.monitor.get_stats()
        lines = []
        lines.append("\033[2J\033[H")
        lines.append("\033[1;36m" + "=" * 64 + "\033[0m")
        lines.append("\033[1;33m  SpectralStream Live Dashboard\033[0m")
        lines.append("\033[1;36m" + "=" * 64 + "\033[0m")
        lines.append(f"  Uptime:        \033[1m{s['uptime_seconds']:.1f}s\033[0m")
        lines.append(f"  Tokens:        \033[1m{s['total_tokens']}\033[0m")
        lines.append(
            f"  Throughput:    \033[1;32m{s['tokens_per_second']:.1f}\033[0m tok/s"
        )
        lines.append(
            f"  Efficiency:    \033[1;32m{s['tokens_per_model_call']:.2f}\033[0m tok/call"
        )
        lines.append(f"  Avg latency:   \033[1m{s['average_latency_ms']:.1f}\033[0m ms")
        lines.append("")
        lines.append("  \033[1;34m-- HDC Draft --\033[0m")
        lines.append(
            f"  Acceptance:    \033[1;32m{s['hdc_acceptance_rate']:.1%}\033[0m"
        )
        lines.append(
            f"  Decisions:     {s['hdc_decisions']} ({s['hdc_accepted']} accepted)"
        )
        lines.append("")
        lines.append("  \033[1;34m-- Confidence Gate --\033[0m")
        lines.append(
            f"  Accuracy:      \033[1;32m{s['confidence_gate_accuracy']:.1%}\033[0m"
        )
        lines.append(
            f"  Predictions:   {s['gate_predictions']} ({s['gate_correct']} correct)"
        )
        lines.append("")
        lines.append("  \033[1;34m-- Cache --\033[0m")
        lines.append(f"  Hit rate:      \033[1;32m{s['cache_hit_rate']:.1%}\033[0m")
        if s.get("errors"):
            lines.append("")
            lines.append("  \033[1;31m-- Errors --\033[0m")
            for err, count in s["errors"].items():
                lines.append(f"  {err}: {count}")
        strategy_breakdown = s.get("strategy_breakdown", {})
        if strategy_breakdown:
            lines.append("")
            lines.append("  \033[1;35m-- Strategy Breakdown --\033[0m")
            for strat, info in strategy_breakdown.items():
                lines.append(
                    f"  {strat:20s}  {info['tokens']:4d} tok  "
                    f"{info['avg_latency_ms']:6.1f} ms  {info['calls']:3d} calls"
                )
        lines.append("\033[1;36m" + "=" * 64 + "\033[0m")
        return "\n".join(lines)

    def _loop(self):
        import time as _time

        try:
            while self._running:
                print(self.render(), end="", flush=True)
                _time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            print("\033[2J\033[HConsole dashboard stopped.")

    def start(self):
        """Start live dashboard in terminal."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False


_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SpectralStream Dashboard</title>
<style>
  :root { --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #c9d1d9; --green: #3fb950; --yellow: #d29922; --red: #f85149; --blue: #58a6ff; --purple: #bc8cff; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace; background: var(--bg); color: var(--text); padding: 20px; }
  h1 { font-size: 1.4em; margin-bottom: 16px; color: var(--blue); }
  h2 { font-size: 1em; color: var(--purple); margin-bottom: 8px; border-bottom: 1px solid var(--border); padding-bottom: 4px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-bottom: 16px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
  .card .label { font-size: 0.75em; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
  .card .value { font-size: 1.6em; font-weight: bold; margin-top: 2px; }
  .card .value.green { color: var(--green); }
  .card .value.yellow { color: var(--yellow); }
  .card .value.red { color: var(--red); }
  .card .sub { font-size: 0.75em; color: #8b949e; margin-top: 2px; }
  .bar-bg { background: #21262d; border-radius: 4px; height: 8px; margin-top: 6px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
  .bar-fill.green { background: var(--green); }
  .bar-fill.yellow { background: var(--yellow); }
  .bar-fill.red { background: var(--red); }
  .bar-fill.blue { background: var(--blue); }
  .strategy-row { display: flex; justify-content: space-between; padding: 3px 0; font-size: 0.85em; }
  .strategy-row .name { color: var(--blue); }
  .strategy-row .val { color: var(--text); }
  table { width: 100%; border-collapse: collapse; font-size: 0.8em; }
  th { text-align: left; color: #8b949e; padding: 4px 8px; border-bottom: 1px solid var(--border); }
  td { padding: 4px 8px; border-bottom: 1px solid var(--border); }
  .timestamp { position: fixed; top: 12px; right: 20px; font-size: 0.7em; color: #8b949e; }
  .log-viewer { background: #000; padding: 8px; border-radius: 4px; font-size: 0.75em; max-height: 200px; overflow-y: auto; }
  .log-viewer .line { padding: 1px 0; }
  .log-viewer .info { color: var(--green); }
  .log-viewer .warn { color: var(--yellow); }
  .log-viewer .error { color: var(--red); }
</style>
</head>
<body>
<span class="timestamp" id="timestamp"></span>
<h1>SpectralStream Inference Engine</h1>
<div class="grid" id="top-metrics">
  <div class="card"><div class="label">Throughput</div><div class="value green" id="tps">--</div><div class="sub">tokens/sec</div></div>
  <div class="card"><div class="label">Efficiency</div><div class="value green" id="tpms">--</div><div class="sub">tok/model-call</div></div>
  <div class="card"><div class="label">Total Tokens</div><div class="value" id="total-tokens">--</div><div class="sub">generated</div></div>
  <div class="card"><div class="label">HDC Acceptance</div><div class="value green" id="hdc-accept">--</div><div class="sub">acceptance rate</div></div>
  <div class="card"><div class="label">Gate Accuracy</div><div class="value yellow" id="gate-acc">--</div><div class="sub">confidence gate</div></div>
  <div class="card"><div class="label">Cache Hit Rate</div><div class="value blue" id="cache-hit">--</div><div class="sub">KV cache</div></div>
</div>

<div class="grid">
  <div class="card">
    <h2>Strategy Breakdown</h2>
    <div id="strategy-breakdown"></div>
  </div>
  <div class="card">
    <h2>KV Cache</h2>
    <div id="kv-stats"></div>
  </div>
  <div class="card">
    <h2>HDC Engine</h2>
    <div id="hdc-stats"></div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h2>Memory Usage</h2>
    <div id="memory-stats"></div>
  </div>
  <div class="card">
    <h2>Online Learning</h2>
    <div id="learning-stats"></div>
  </div>
  <div class="card">
    <h2>System</h2>
    <div id="system-stats"></div>
  </div>
</div>

<script>
function fmt(v) { if (v === undefined || v === null) return '--'; if (typeof v === 'number') return v.toFixed(2); return String(v); }
function pct(v) { return (v * 100).toFixed(1) + '%'; }
function bar(v, color) { return '<div class="bar-bg"><div class="bar-fill ' + color + '" style="width:' + Math.min(v * 100, 100) + '%"></div></div>'; }

async function refresh() {
  try {
    let r = await fetch('/api/stats');
    let s = await r.json();
    let d = s.monitor || s;
    document.getElementById('tps').textContent = fmt(d.tokens_per_second);
    document.getElementById('tpms').textContent = fmt(d.tokens_per_model_call);
    document.getElementById('total-tokens').textContent = d.total_tokens || '--';
    document.getElementById('hdc-accept').textContent = pct(d.hdc_acceptance_rate);
    document.getElementById('gate-acc').textContent = pct(d.confidence_gate_accuracy);
    document.getElementById('cache-hit').textContent = pct(d.cache_hit_rate);
    document.getElementById('timestamp').textContent = new Date().toLocaleTimeString();

    let stratDiv = document.getElementById('strategy-breakdown');
    let sb = d.strategy_breakdown || {};
    let stratHtml = '';
    for (let [k, v] of Object.entries(sb)) {
      stratHtml += '<div class="strategy-row"><span class="name">' + k + '</span><span class="val">' + v.tokens + ' tok, ' + v.avg_latency_ms + ' ms</span></div>';
    }
    stratHtml += '<div class="strategy-row" style="margin-top:6px;border-top:1px solid var(--border);padding-top:4px"><span class="name">Total calls</span><span class="val">' + (d.total_model_calls || '--') + '</span></div>';
    stratDiv.innerHTML = stratHtml;

    document.getElementById('kv-stats').innerHTML =
      '<div class="strategy-row"><span class="name">Compression</span><span class="val">' + fmt(s.kv_compression_ratio) + 'x</span></div>' +
      '<div class="strategy-row"><span class="name">Hit rate</span><span class="val">' + pct(d.cache_hit_rate || d.kv_cache_hit_rate) + '</span></div>';

    document.getElementById('hdc-stats').innerHTML =
      '<div class="strategy-row"><span class="name">Acceptance</span><span class="val">' + pct(d.hdc_acceptance_rate) + '</span></div>' +
      '<div class="strategy-row"><span class="name">Decisions</span><span class="val">' + (d.hdc_decisions || '--') + '</span></div>' +
      '<div class="strategy-row"><span class="name">Accepted</span><span class="val">' + (d.hdc_accepted || '--') + '</span></div>';

    let memHtml = '';
    let mem = d.monitor ? d.monitor.estimated_memory_bytes : null;
    if (!mem) { mem = s.memory_usage || {}; }
    if (Object.keys(mem).length) {
      for (let [k, v] of Object.entries(mem)) {
        let mb = (v / 1024 / 1024).toFixed(1);
        memHtml += '<div class="strategy-row"><span class="name">' + k + '</span><span class="val">' + mb + ' MB</span></div>';
      }
    } else {
      memHtml = '<div class="strategy-row"><span class="name">Data</span><span class="val">--</span></div>';
    }
    document.getElementById('memory-stats').innerHTML = memHtml;

    let learn = s.learning || {};
    document.getElementById('learning-stats').innerHTML =
      '<div class="strategy-row"><span class="name">Buffer</span><span class="val">' + (learn.buffer_size || '--') + '</span></div>' +
      '<div class="strategy-row"><span class="name">Corrections</span><span class="val">' + (learn.total_corrections || '--') + '</span></div>';

    let sysHtml =
      '<div class="strategy-row"><span class="name">Uptime</span><span class="val">' + fmt(d.uptime_seconds) + 's</span></div>' +
      '<div class="strategy-row"><span class="name">Model loaded</span><span class="val">' + (s.model_loaded ? 'yes' : 'no') + '</span></div>' +
      '<div class="strategy-row"><span class="name">Hidden dim</span><span class="val">' + (s.hidden_dim || '--') + '</span></div>' +
      '<div class="strategy-row"><span class="name">Vocab size</span><span class="val">' + (s.vocab_size || '--') + '</span></div>';
    document.getElementById('system-stats').innerHTML = sysHtml;

  } catch(e) {
    console.error('Refresh failed', e);
  }
}
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>"""


def create_dashboard(orchestrator=None, monitor=None, host="127.0.0.1", port=8080):
    """Create and start a dashboard server. Convenience factory."""
    dash = DashboardServer(orchestrator, monitor, host, port)
    dash.start()
    return dash


def create_console(monitor):
    """Create and start a console dashboard. Convenience factory."""
    con = ConsoleDashboard(monitor)
    con.start()
    return con
