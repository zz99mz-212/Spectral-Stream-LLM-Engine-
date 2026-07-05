/**
 * SpectralStream Web UI — Main Application Script
 * Handles theme toggling, sidebar, toast notifications, and global interactions.
 */

'use strict';

// ═══════════════════════════════════════════════════════════════════════
// DOM Ready
// ═══════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initSidebar();
  initToasts();
  initGlobalKeyboard();
});

// ═══════════════════════════════════════════════════════════════════════
// Theme Management
// ═══════════════════════════════════════════════════════════════════════
function initTheme() {
  const toggle = document.getElementById('themeToggle');
  const label = document.getElementById('themeLabel');
  if (!toggle) return;

  // Load saved theme
  const saved = localStorage.getItem('spectralstream-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  toggle.querySelector('input').checked = saved === 'light';
  if (label) label.textContent = saved === 'light' ? 'Light' : 'Dark';

  toggle.addEventListener('change', () => {
    const isLight = toggle.querySelector('input').checked;
    const theme = isLight ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('spectralstream-theme', theme);
    if (label) label.textContent = isLight ? 'Light' : 'Dark';
  });
}

// ═══════════════════════════════════════════════════════════════════════
// Sidebar
// ═══════════════════════════════════════════════════════════════════════
function initSidebar() {
  const toggleBtn = document.getElementById('sidebarToggle');
  const sidebar = document.getElementById('sidebar');
  if (!toggleBtn || !sidebar) return;

  toggleBtn.addEventListener('click', () => {
    sidebar.classList.toggle('open');
  });

  // Close sidebar on outside click (mobile)
  document.addEventListener('click', (e) => {
    if (window.innerWidth <= 768 &&
        sidebar.classList.contains('open') &&
        !sidebar.contains(e.target) &&
        !toggleBtn.contains(e.target)) {
      sidebar.classList.remove('open');
    }
  });

  // Close sidebar on escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && sidebar.classList.contains('open')) {
      sidebar.classList.remove('open');
    }
  });
}

// ═══════════════════════════════════════════════════════════════════════
// Toast Notifications
// ═══════════════════════════════════════════════════════════════════════
function initToasts() {
  window.showToast = showToast;
}

/**
 * Display a toast notification.
 * @param {string} message - The message to display.
 * @param {'success'|'error'|'warning'|'info'} type - Toast type.
 * @param {number} duration - Display duration in ms (default 3000).
 */
function showToast(message, type = 'info', duration = 3000) {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const icons = {
    success: '✓',
    error: '✕',
    warning: '⚠',
    info: 'ℹ',
  };

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span>${icons[type] || 'ℹ'}</span>
    <span>${escapeHtml(message)}</span>
  `;

  container.appendChild(toast);

  // Auto-remove
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    toast.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, duration);

  // Allow click to dismiss
  toast.addEventListener('click', () => {
    toast.remove();
  });

  return toast;
}

// ═══════════════════════════════════════════════════════════════════════
// Global Keyboard Shortcuts
// ═══════════════════════════════════════════════════════════════════════
function initGlobalKeyboard() {
  document.addEventListener('keydown', (e) => {
    // Don't trigger if user is typing in an input
    const tag = e.target.tagName;
    const isInput = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';

    // Ctrl+K to focus search/command bar (future)
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      // Future: open command palette
    }

    // Escape to close modals
    if (e.key === 'Escape') {
      document.querySelectorAll('.modal-overlay[style*="flex"]').forEach(m => {
        m.style.display = 'none';
      });
    }
  });
}

// ═══════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════

/**
 * Escape HTML to prevent XSS.
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

/**
 * Format a timestamp to a readable date string.
 * @param {number|string} ts - Unix timestamp or ISO string.
 * @returns {string}
 */
function formatTimestamp(ts) {
  if (!ts) return '—';
  const d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
  if (isNaN(d.getTime())) return String(ts);
  return d.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Format bytes into a human-readable string.
 * @param {number} bytes
 * @param {number} decimals
 * @returns {string}
 */
function formatBytes(bytes, decimals = 2) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(Math.abs(bytes)) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + ' ' + sizes[i];
}

/**
 * Debounce a function call.
 * @param {Function} fn
 * @param {number} ms
 * @returns {Function}
 */
function debounce(fn, ms = 300) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

/**
 * Throttle a function call.
 * @param {Function} fn
 * @param {number} ms
 * @returns {Function}
 */
function throttle(fn, ms = 300) {
  let last = 0;
  return (...args) => {
    const now = Date.now();
    if (now - last >= ms) {
      last = now;
      fn(...args);
    }
  };
}

// ═══════════════════════════════════════════════════════════════════════
// Active Navigation Highlight
// ═══════════════════════════════════════════════════════════════════════
// The active page is set via the Jinja2 template variable `active_page`
// and rendered in the sidebar nav items. This script updates the header
// title if needed.

// ═══════════════════════════════════════════════════════════════════════
// Global re-initialization for HTMX-style page transitions
// ═══════════════════════════════════════════════════════════════════════
// If the page content is replaced dynamically, call this to re-attach handlers.
function reinitUI() {
  initTheme();
  initSidebar();
}

// ═══════════════════════════════════════════════════════════════════════
// Compression UI
// ═══════════════════════════════════════════════════════════════════════

// These functions are also defined inline in compress.html for self-containment.
// Shared globally for cross-page use.

/**
 * Update a slider's displayed value.
 * @param {HTMLInputElement} slider
 * @param {string} outputId
 * @param {number} decimals
 */
function updateSliderValue(slider, outputId, decimals) {
  const val = parseFloat(slider.value);
  const el = document.getElementById(outputId);
  if (el) el.textContent = val.toFixed(decimals || 0);
  if (outputId === 'error-value') {
    const pct = document.getElementById('error-pct');
    if (pct) pct.textContent = (val * 100).toFixed(2);
  }
}

/**
 * Scan for available .safetensors models.
 */
async function scanModels() {
  const select = document.getElementById('model-select');
  if (!select) return;
  const hint = document.getElementById('scan-hint');
  if (hint) hint.textContent = 'Scanning...';

  try {
    const res = await fetch('/api/models/scan');
    const data = await res.json();
    select.innerHTML = '<option value="">Select a model...</option>';
    for (const model of data.models) {
      const opt = document.createElement('option');
      opt.value = model.path;
      opt.textContent = `${model.name} (${model.size_gb} GB)`;
      select.appendChild(opt);
    }
    if (hint) {
      hint.textContent = `Found ${data.models.length} model(s)`;
      setTimeout(() => { hint.textContent = 'Click refresh to scan for .safetensors models'; }, 3000);
    }
  } catch (e) {
    if (hint) {
      hint.textContent = 'Scan failed: ' + e.message;
      hint.style.color = 'var(--color-error)';
    }
  }
}

/**
 * Start a compression job from the form.
 */
async function startCompression() {
  const form = document.getElementById('compress-form');
  if (!form) return;
  const formData = new FormData(form);

  if (!formData.get('model_path')) {
    showToast('Please select a model first', 'warning');
    return;
  }

  try {
    const res = await fetch('/api/compress/start', { method: 'POST', body: formData });
    const data = await res.json();
    window.currentJobId = data.job_id;

    const configPanel = document.getElementById('config-panel');
    const vizPlaceholder = document.getElementById('viz-placeholder');
    const tensorGrid = document.getElementById('tensor-grid-container');
    const progressPanel = document.getElementById('progress-panel');

    if (configPanel) configPanel.style.display = 'none';
    if (vizPlaceholder) vizPlaceholder.style.display = 'none';
    if (tensorGrid) tensorGrid.style.display = 'block';
    if (progressPanel) progressPanel.style.display = 'block';

    const statusEl = document.getElementById('progress-status');
    if (statusEl) {
      statusEl.textContent = 'Starting';
      statusEl.className = 'badge badge-primary';
    }

    const startBtn = document.getElementById('start-btn');
    if (startBtn) startBtn.disabled = true;

    pollCompressionStatus(data.job_id);
  } catch (e) {
    showToast('Failed to start: ' + e.message, 'error');
  }
}

/**
 * Poll compression job status.
 * @param {string} jobId
 */
async function pollCompressionStatus(jobId) {
  try {
    const res = await fetch(`/api/compress/status/${jobId}`);
    const status = await res.json();

    const runningStates = ['running', 'profiling', 'compressing', 'initializing', 'finalizing', 'starting'];

    updateCompressionUI(status);

    if (runningStates.includes(status.status)) {
      setTimeout(() => pollCompressionStatus(jobId), 400);
    } else if (status.status === 'completed') {
      await showResults(jobId);
    } else if (status.status === 'failed') {
      showCompressionError(status.error || 'Unknown error');
    }
  } catch (e) {
    setTimeout(() => pollCompressionStatus(jobId), 1000);
  }
}

/**
 * Update UI with latest compression status.
 * @param {Object} status
 */
function updateCompressionUI(status) {
  const pct = Math.round(status.progress || 0);

  const fill = document.getElementById('progress-fill');
  const text = document.getElementById('progress-text');
  const statusEl = document.getElementById('progress-status');
  const tensorEl = document.getElementById('current-tensor');
  const methodEl = document.getElementById('current-method');
  const tensorsEl = document.getElementById('tensors-done');
  const ratioEl = document.getElementById('ratio-so-far');
  const errorEl = document.getElementById('error-so-far');
  const timeEl = document.getElementById('time-elapsed');

  if (fill) fill.style.width = pct + '%';
  if (text) text.textContent = pct + '%';
  if (statusEl) {
    statusEl.textContent = status.status.charAt(0).toUpperCase() + status.status.slice(1);
  }
  if (tensorEl && status.current_tensor) tensorEl.textContent = status.current_tensor;
  if (methodEl && status.current_method) methodEl.textContent = status.current_method;
  if (tensorsEl) tensorsEl.textContent = `${status.tensors_done || 0} / ${status.total_tensors || 0}`;
  if (ratioEl) ratioEl.textContent = status.ratio_so_far ? `${status.ratio_so_far}x` : '—';
  if (errorEl) errorEl.textContent = status.error_so_far ? `${(status.error_so_far * 100).toFixed(4)}%` : '—';
  if (timeEl) timeEl.textContent = `${(status.elapsed || 0).toFixed(1)}s`;

  if (status.tensors && status.tensors.length > 0) {
    updateTensorGrid(status.tensors);
  }
}

/**
 * Render the tensor visualization grid.
 * @param {Array} tensors
 */
function updateTensorGrid(tensors) {
  const grid = document.getElementById('tensor-grid');
  if (!grid) return;
  grid.innerHTML = '';
  const tierColors = { 1: '#00FF88', 2: '#00CEC9', 3: '#6C5CE7', 4: '#FFD700', 5: '#FF4444' };

  for (const tensor of tensors) {
    const el = document.createElement('div');
    el.className = 'tensor-block';
    const color = tierColors[tensor.tier] || '#555';
    el.style.background = color;
    el.style.boxShadow = `0 0 8px ${color}44`;

    const sizeNorm = Math.min((tensor.size_mb || 0.1) / 50, 1);
    const dim = Math.max(8, Math.min(48, 8 + sizeNorm * 40));
    el.style.width = dim + 'px';
    el.style.height = dim + 'px';

    el.title = `${tensor.name}\nMethod: ${tensor.method}\nTier: ${tensor.tier}\nRatio: ${tensor.ratio}x\nError: ${(tensor.error * 100).toFixed(4)}%\nGrade: ${tensor.grade}`;
    grid.appendChild(el);
  }
}

/**
 * Show compression results after job completion.
 * @param {string} jobId
 */
async function showResults(jobId) {
  try {
    const res = await fetch(`/api/compress/result/${jobId}`);
    const result = await res.json();

    const progressPanel = document.getElementById('progress-panel');
    const resultsPanel = document.getElementById('results-panel');
    const tableCard = document.getElementById('tensor-table-card');

    if (progressPanel) progressPanel.style.display = 'none';
    if (resultsPanel) resultsPanel.style.display = 'block';

    const cert = result.certificate_json || {};
    const comp = cert.compression || {};
    const qual = cert.quality || {};
    const model = cert.model || {};
    const marketing = cert.marketing || {};

    const statsEl = document.getElementById('result-stats');
    if (statsEl) {
      statsEl.innerHTML = `
        <div class="stat-card"><div class="stat-label">Compression Ratio</div><div class="stat-value" style="color:var(--color-success);">${comp.ratio || 0}x</div></div>
        <div class="stat-card"><div class="stat-label">Original Size</div><div class="stat-value">${model.original_size_gb || 0} <span class="unit">GB</span></div></div>
        <div class="stat-card"><div class="stat-label">Compressed Size</div><div class="stat-value">${model.compressed_size_gb || 0} <span class="unit">GB</span></div></div>
        <div class="stat-card"><div class="stat-label">Space Saved</div><div class="stat-value" style="color:var(--color-secondary);">${marketing.space_saved_gb || 0} <span class="unit">GB</span></div></div>
        <div class="stat-card"><div class="stat-label">Avg Error</div><div class="stat-value" style="color:var(--color-warning);">${qual.avg_error_percent || 0}%</div></div>
        <div class="stat-card"><div class="stat-label">Avg SNR</div><div class="stat-value">${qual.avg_snr_db || 0} <span class="unit">dB</span></div></div>
        <div class="stat-card"><div class="stat-label">Compression Time</div><div class="stat-value">${comp.time_seconds || 0} <span class="unit">s</span></div></div>
        <div class="stat-card"><div class="stat-label">Methods Used</div><div class="stat-value">${comp.methods_used || 0}</div></div>
      `;
    }

    renderMethodChart(cert.method_distribution || {});
    renderGradeChart(qual.grade_distribution || {});

    // Fill certificate views
    const certHtml = document.getElementById('certificate-content');
    const certMd = document.getElementById('certificate-md-content');
    const certJson = document.getElementById('certificate-json-content');
    const certText = document.getElementById('certificate-text-content');

    if (certHtml) certHtml.innerHTML = result.certificate_html || '';
    if (certMd) certMd.textContent = result.certificate_md || '';
    if (certJson) certJson.textContent = JSON.stringify(result.certificate_json, null, 2);
    if (certText) certText.textContent = result.certificate_txt || '';

    // Build tensor table
    const tensors = cert.tensors || [];
    if (tableCard) tableCard.style.display = 'block';

    const countBadge = document.getElementById('tensor-count-badge');
    if (countBadge) countBadge.textContent = tensors.length + ' tensors';

    const tbody = document.getElementById('tensor-table-body');
    if (tbody) {
      tbody.innerHTML = '';
      for (const t of tensors) {
        const tr = document.createElement('tr');
        const shapeStr = (t.shape || []).slice(0, 3).join('\u00D7') + ((t.shape || []).length > 3 ? '\u2026' : '');
        const errPct = ((t.relative_error || 0) * 100).toFixed(4);
        tr.innerHTML = `
          <td style="font-size:0.85em;max-width:200px;overflow:hidden;text-overflow:ellipsis;" title="${escapeHtml(t.name || '')}">${escapeHtml((t.name || '').slice(0, 50))}</td>
          <td>${shapeStr}</td>
          <td>${escapeHtml(t.method || '')}</td>
          <td><span class="tier-badge tier-${t.quality_grade === 'S' || t.quality_grade === 'A' ? 1 : t.quality_grade === 'B' ? 2 : t.quality_grade === 'C' ? 3 : t.quality_grade === 'D' ? 4 : 5}">Tier ${t.quality_grade === 'S' || t.quality_grade === 'A' ? 1 : t.quality_grade === 'B' ? 2 : t.quality_grade === 'C' ? 3 : t.quality_grade === 'D' ? 4 : 5}</span></td>
          <td>${(t.compression_ratio || 0).toFixed(2)}x</td>
          <td style="color:${errPct < 0.1 ? 'var(--color-success)' : errPct < 1 ? 'var(--color-warning)' : 'var(--color-error)'}">${errPct}%</td>
          <td>${(t.snr_db || 0).toFixed(1)} dB</td>
          <td><span class="badge badge-grade ${t.quality_grade || 'F'}">${t.quality_grade || 'F'}</span></td>
        `;
        tbody.appendChild(tr);
      }
    }

    setupDownloadButtons(result);
    showToast('Compression completed successfully!', 'success');
  } catch (e) {
    showToast('Failed to load results: ' + e.message, 'error');
  }
}

/**
 * Render method distribution as a CSS pie chart.
 * @param {Object} dist
 */
function renderMethodChart(dist) {
  const container = document.getElementById('method-chart');
  if (!container) return;
  const entries = Object.entries(dist);
  const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
  const colors = ['#6C5CE7', '#00CEC9', '#00FF88', '#FFD700', '#FD79A8', '#FF4444', '#8888FF', '#33DAD6'];

  const conicParts = entries.map(([_, count], i) => {
    const startPct = entries.slice(0, i).reduce((s, [__, c]) => s + c / total, 0) * 100;
    const pct = count / total * 100;
    return `${colors[i % colors.length]} ${startPct}% ${startPct + pct}%`;
  }).join(', ');

  container.innerHTML = `
    <div class="css-pie-chart" style="background: conic-gradient(${conicParts});"></div>
    <div class="flex flex-wrap gap-2" style="justify-content:center;margin-top:16px;">
      ${entries.map(([method, count], i) =>
        `<span class="method-tag" style="background:${colors[i % colors.length]}22;border:1px solid ${colors[i % colors.length]}44;">
          <span style="color:${colors[i % colors.length]}">●</span> ${method} (${count})
        </span>`
      ).join('')}
    </div>`;
}

/**
 * Render grade distribution as CSS bar chart.
 * @param {Object} grades
 */
function renderGradeChart(grades) {
  const container = document.getElementById('grade-chart');
  if (!container) return;
  const gradeOrder = ['S', 'A', 'B', 'C', 'D', 'F'];
  const gradeColors = { S: '#00FF88', A: '#00CEC9', B: '#6C5CE7', C: '#8888FF', D: '#FFD700', F: '#FF4444' };
  const total = Object.values(grades).reduce((a, b) => a + b, 0) || 1;

  container.innerHTML = `<div class="css-bar-chart">
    ${gradeOrder.map(g => {
      const count = grades[g] || 0;
      const pct = count / total * 100;
      return `<div class="css-bar" style="height:${Math.max(pct, 2)}%;background:${gradeColors[g]};">
        <div class="bar-label">${g}<br><strong>${count}</strong></div>
      </div>`;
    }).join('')}
  </div>`;
}

/**
 * Switch certificate tab.
 * @param {string} tab
 */
function switchCertTab(tab) {
  document.querySelectorAll('.cert-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.cert-content').forEach(c => c.classList.remove('active'));
  const tabEl = document.querySelector(`.cert-tab[data-target="${tab}"]`);
  const contentEl = document.getElementById(`cert-${tab}`);
  if (tabEl) tabEl.classList.add('active');
  if (contentEl) contentEl.classList.add('active');
}

/**
 * Set up download buttons for certificate formats.
 * @param {Object} result
 */
function setupDownloadButtons(result) {
  const container = document.getElementById('download-buttons');
  if (!container) return;
  container.innerHTML = '';
  const formats = [
    { label: 'JSON', data: JSON.stringify(result.certificate_json, null, 2), mime: 'application/json', ext: 'json' },
    { label: 'HTML', data: result.certificate_html, mime: 'text/html', ext: 'html' },
    { label: 'Markdown', data: result.certificate_md, mime: 'text/markdown', ext: 'md' },
    { label: 'Text', data: result.certificate_txt, mime: 'text/plain', ext: 'txt' },
  ];
  formats.forEach(f => {
    const btn = document.createElement('button');
    btn.className = 'btn btn-ghost btn-sm';
    btn.textContent = '\u2B07 ' + f.label;
    btn.onclick = () => {
      const blob = new Blob([f.data], { type: f.mime });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `certificate.${f.ext}`;
      a.click();
      URL.revokeObjectURL(url);
    };
    container.appendChild(btn);
  });
}

/**
 * Show compression error state.
 * @param {string} msg
 */
function showCompressionError(msg) {
  const statusEl = document.getElementById('progress-status');
  if (statusEl) {
    statusEl.textContent = 'Failed';
    statusEl.className = 'badge badge-error';
  }
  showToast('Compression failed: ' + msg, 'error');
  const startBtn = document.getElementById('start-btn');
  if (startBtn) startBtn.disabled = false;
}
