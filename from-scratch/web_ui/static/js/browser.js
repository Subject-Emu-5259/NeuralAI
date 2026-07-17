// NeuralBrowser — Embedded web browser controller (Mirror + Standalone modes)
// Talks to: /api/browser/navigate, /api/browser/action, /api/browser/run (SSE), /api/browser/close
// Depends on globals from main_v2.js: authToken, switchTab, showToast

let _browserMode = 'standalone';
let _browserBusy = false;

function _browserHeaders() {
  const h = { 'Content-Type': 'application/json' };
  if (typeof authToken !== 'undefined' && authToken) h['Authorization'] = 'Bearer ' + authToken;
  return h;
}

function setBrowserMode(mode) {
  _browserMode = mode;
  const ms = document.getElementById('modeStandalone');
  const mm = document.getElementById('modeMirror');
  const aiInputWrap = document.getElementById('browserAiInputWrap');
  const takeover = document.getElementById('browserTakeover');
  const stop = document.getElementById('browserStop');
  if (ms) ms.classList.toggle('active', mode === 'standalone');
  if (mm) mm.classList.toggle('active', mode === 'mirror');
  if (aiInputWrap) aiInputWrap.style.display = (mode === 'mirror') ? 'flex' : 'none';
  if (takeover) takeover.style.display = 'none';
  if (stop) stop.style.display = (mode === 'mirror' && _browserBusy) ? 'inline-flex' : 'none';
  const titleEl = document.getElementById('browserSideTitle');
  if (titleEl) titleEl.textContent = (mode === 'mirror') ? 'AI Driver Log' : 'Session';
  const log = document.getElementById('browserAiLog');
  if (log && mode === 'mirror') log.innerHTML = '<div class="browser-log-empty">Describe a task below (e.g. "search for AI news") and the AI will drive the page live.</div>';
}

function _browserShow(pngDataUrl) {
  const img = document.getElementById('browserScreenshot');
  const ph = document.getElementById('browserPlaceholder');
  if (img && pngDataUrl) {
    img.src = pngDataUrl;
    img.style.display = 'block';
    if (ph) ph.style.display = 'none';
  }
}

function _browserSetStatus(msg) {
  const el = document.getElementById('browserStatus');
  if (el) el.textContent = msg || '';
}

function _browserLog(kind, text) {
  const log = document.getElementById('browserAiLog');
  if (!log) return;
  const empty = log.querySelector('.browser-log-empty');
  if (empty) empty.remove();
  const row = document.createElement('div');
  row.className = 'browser-log-row ' + (kind || 'step');
  const icon = kind === 'done' ? '✅' : kind === 'error' ? '❌' : '▶';
  row.innerHTML = '<span class="browser-log-icon">' + icon + '</span><span class="browser-log-text">' + (text || '').replace(/</g, '&lt;') + '</span>';
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
}

async function browserGo() {
  if (_browserBusy) return;
  const urlInput = document.getElementById('browserUrl');
  let url = (urlInput && urlInput.value || '').trim();
  if (!url) return;
  _browserSetStatus('Loading ' + url + ' …');
  try {
    const res = await fetch('/api/browser/navigate', { method: 'POST', headers: _browserHeaders(), body: JSON.stringify({ url }) });
    const data = await res.json();
    if (data && data.success) {
      _browserShow(data.screenshot);
      if (urlInput) urlInput.value = data.url || url;
      _browserSetStatus('Loaded ' + (data.url || url));
    } else {
      _browserSetStatus('Failed: ' + (data && data.error ? data.error : 'unknown'));
      showToast && showToast('Browser failed to load', 'error');
    }
  } catch (e) {
    _browserSetStatus('Error: ' + e.message);
  }
}

function quickBrowse(url) {
  const urlInput = document.getElementById('browserUrl');
  if (urlInput && url) urlInput.value = url;
  browserGo();
}

async function browserAction(act, extra) {
  if (_browserBusy) return;
  const payload = Object.assign({ action: act }, extra || {});
  try {
    const res = await fetch('/api/browser/action', { method: 'POST', headers: _browserHeaders(), body: JSON.stringify(payload) });
    const data = await res.json();
    if (data && data.screenshot) _browserShow(data.screenshot);
    if (data && data.url) { const u = document.getElementById('browserUrl'); if (u) u.value = data.url; }
    if (!data || !data.success) _browserSetStatus('Action failed: ' + (data && data.error ? data.error : ''));
  } catch (e) {
    _browserSetStatus('Action error: ' + e.message);
  }
}

function browserTakeover() {
  // User grabs control: stop AI streaming, reveal standalone controls.
  _browserBusy = false;
  browserStopAI();
  setBrowserMode('standalone');
  _browserSetStatus('You have control of the browser.');
  showToast && showToast('You took over the browser', 'success');
}

let _browserAbort = false;
function browserStopAI() { _browserAbort = true; _browserBusy = false; _browserSetStatus('AI stopped.'); }

async function browserRunAI() {
  if (_browserBusy) return;
  const taskEl = document.getElementById('browserAiTask');
  const urlEl = document.getElementById('browserUrl');
  const task = (taskEl && taskEl.value || '').trim();
  const url = (urlEl && urlEl.value || '').trim();
  if (!task && !url) { showToast && showToast('Enter a task or URL first', 'error'); return; }

  _browserBusy = true;
  _browserAbort = false;
  const takeover = document.getElementById('browserTakeover');
  const stop = document.getElementById('browserStop');
  if (takeover) takeover.style.display = 'inline-flex';
  if (stop) stop.style.display = 'inline-flex';
  _browserLog('step', '🤖 Starting: ' + (task || url));

  try {
    const res = await fetch('/api/browser/run', {
      method: 'POST', headers: _browserHeaders(),
      body: JSON.stringify({ task, url })
    });
    if (!res.ok) { _browserLog('error', 'Server error ' + res.status); _browserBusy = false; return; }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      if (_browserAbort) { await reader.cancel(); break; }
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        const t = line.trim();
        if (!t.startsWith('data: ')) continue;
        const raw = t.slice(6).trim();
        if (!raw) continue;
        try {
          const ev = JSON.parse(raw);
          if (ev.screenshot) _browserShow(ev.screenshot);
          if (ev.kind === 'done' || ev.kind === 'error') {
            _browserLog(ev.kind, ev.text);
          } else {
            _browserLog('step', ev.text);
          }
        } catch (_) { /* ignore malformed */ }
      }
    }
  } catch (e) {
    _browserLog('error', 'Stream error: ' + e.message);
  } finally {
    _browserBusy = false;
    const takeover2 = document.getElementById('browserTakeover');
    const stop2 = document.getElementById('browserStop');
    if (takeover2) takeover2.style.display = 'none';
    if (stop2) stop2.style.display = 'none';
  }
}

async function browserClose() {
  try {
    await fetch('/api/browser/close', { method: 'POST', headers: _browserHeaders() });
  } catch (_) {}
  const img = document.getElementById('browserScreenshot');
  const ph = document.getElementById('browserPlaceholder');
  if (img) { img.style.display = 'none'; img.src = ''; }
  if (ph) ph.style.display = 'flex';
  const log = document.getElementById('browserAiLog');
  if (log) log.innerHTML = '';
  _browserSetStatus('Session closed.');
}

// Wire navigation buttons when the browser tab is shown.
document.addEventListener('DOMContentLoaded', () => {
  const back = document.getElementById('browserBack');
  const fwd = document.getElementById('browserFwd');
  const reload = document.getElementById('browserReload');
  if (back) back.addEventListener('click', () => browserAction('back'));
  if (fwd) fwd.addEventListener('click', () => browserAction('forward'));
  if (reload) reload.addEventListener('click', () => browserAction('reload'));
  setBrowserMode('standalone');
});
