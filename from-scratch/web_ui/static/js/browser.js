// NeuralBrowser — Embedded web browser controller (Mirror + Standalone modes)
// Talks to: /api/browser/navigate, /api/browser/action, /api/browser/run (SSE), /api/browser/close
// Depends on globals: authToken, switchTab, showToast

let _browserMode = 'standalone';
let _browserBusy = false;
let _zoom = 100;
let _tabs = [];           // {id, title, url, screenshot}
let _activeTab = null;
let _bmKey = 'neuralBrowserBookmarks';

function _browserHeaders() {
  const h = { 'Content-Type': 'application/json' };
  if (typeof authToken !== 'undefined' && authToken) h['Authorization'] = 'Bearer ' + authToken;
  return h;
}

function _resolveUrl(input) {
  input = (input || '').trim();
  if (!input) return '';
  if (/^https?:\/\//i.test(input)) return input;
  if (/^[\w-]+(\.[\w-]+)+(\/.*)?$/.test(input) && !input.includes(' ')) {
    return 'https://' + input;
  }
  // treat as a search query
  return 'https://www.google.com/search?q=' + encodeURIComponent(input);
}

function _loadBookmarks() {
  try { return JSON.parse(localStorage.getItem(_bmKey) || '[]'); }
  catch (_) { return []; }
}
function _saveBookmarks(list) { localStorage.setItem(_bmKey, JSON.stringify(list)); }

function _renderBookmarks() {
  const wrap = document.getElementById('browserBmList');
  if (!wrap) return;
  const list = _loadBookmarks();
  wrap.innerHTML = '';
  list.forEach((bm, i) => {
    const el = document.createElement('div');
    el.className = 'browser-bm';
    el.innerHTML = '<span class="browser-bm-fav"></span><span>' + (bm.title || bm.url) + '</span><span class="browser-bm-del" title="Remove">✕</span>';
    el.querySelector('.browser-bm-fav').onclick = () => quickBrowse(bm.url);
    el.querySelector('span:nth-child(2)').onclick = () => quickBrowse(bm.url);
    el.querySelector('.browser-bm-del').onclick = (e) => {
      e.stopPropagation();
      list.splice(i, 1); _saveBookmarks(list); _renderBookmarks();
    };
    wrap.appendChild(el);
  });
}

function _addBookmark() {
  const tab = _activeTab;
  if (!tab || !tab.url) { showToast && showToast('Open a page first', 'error'); return; }
  const list = _loadBookmarks();
  if (list.some(b => b.url === tab.url)) { showToast && showToast('Already bookmarked', 'error'); return; }
  list.push({ title: tab.title || tab.url, url: tab.url });
  _saveBookmarks(list); _renderBookmarks();
  showToast && showToast('Bookmarked ' + (tab.title || tab.url), 'success');
}

function _newTab(url) {
  const tab = { id: 't' + Date.now() + Math.floor(Math.random()*1000), title: 'New Tab', url: '', screenshot: '' };
  _tabs.push(tab);
  _activeTab = tab;
  _renderTabs();
  if (url) { const u = document.getElementById('browserUrl'); if (u) u.value = url; browserGo(); }
}

function _renderTabs() {
  const strip = document.getElementById('browserTabs');
  if (!strip) return;
  strip.innerHTML = '';
  _tabs.forEach(tab => {
    const el = document.createElement('div');
    el.className = 'browser-tab' + (tab === _activeTab ? ' active' : '');
    el.innerHTML = '<span class="browser-tab-fav">N</span><span class="browser-tab-title">' + (tab.title || 'New Tab') + '</span><span class="browser-tab-close">✕</span>';
    el.querySelector('.browser-tab-fav').onclick = () => _selectTab(tab);
    el.querySelector('.browser-tab-title').onclick = () => _selectTab(tab);
    el.querySelector('.browser-tab-close').onclick = (e) => { e.stopPropagation(); _closeTab(tab); };
    strip.appendChild(el);
  });
  const img = document.getElementById('browserScreenshot');
  const ph = document.getElementById('browserPlaceholder');
  if (_activeTab && _activeTab.screenshot && img) {
    img.src = _activeTab.screenshot; img.style.display = 'block';
    if (ph) ph.style.display = 'none';
  } else if (img) { img.style.display = 'none'; if (ph) ph.style.display = 'flex'; }
  const urlBox = document.getElementById('browserUrl');
  if (urlBox) urlBox.value = _activeTab ? _activeTab.url : '';
}

function _selectTab(tab) { _activeTab = tab; _renderTabs(); }
function _closeTab(tab) {
  const i = _tabs.indexOf(tab);
  if (i >= 0) _tabs.splice(i, 1);
  if (_tabs.length === 0) { _newTab(); return; }
  if (_activeTab === tab) _activeTab = _tabs[Math.max(0, i - 1)];
  _renderTabs();
}

function _setZoom(pct) {
  _zoom = Math.max(50, Math.min(200, pct));
  const img = document.getElementById('browserScreenshot');
  if (img) img.style.width = _zoom + '%';
  const lbl = document.getElementById('browserZoomLvl');
  if (lbl) lbl.textContent = _zoom + '%';
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

function _browserShow(pngDataUrl, url, title) {
  const img = document.getElementById('browserScreenshot');
  const ph = document.getElementById('browserPlaceholder');
  if (img && pngDataUrl) {
    img.src = pngDataUrl;
    img.style.display = 'block';
    if (ph) ph.style.display = 'none';
    if (_activeTab) { _activeTab.screenshot = pngDataUrl; if (url) _activeTab.url = url; if (title) _activeTab.title = title; _renderTabs(); }
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
  url = _resolveUrl(url);
  if (!url) return;
  if (urlInput) urlInput.value = url;
  if (!_activeTab) _newTab();
  _browserSetStatus('Loading ' + url + ' …');
  try {
    const res = await fetch('/api/browser/navigate', { method: 'POST', headers: _browserHeaders(), body: JSON.stringify({ url }) });
    const data = await res.json();
    if (data && data.success) {
      _browserShow(data.screenshot, data.url || url, (data.title || url));
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
  if (!_activeTab) _newTab();
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
    if (data && data.screenshot) _browserShow(data.screenshot, data.url, data.title);
    if (data && data.url) { const u = document.getElementById('browserUrl'); if (u) u.value = data.url; }
    if (!data || !data.success) _browserSetStatus('Action failed: ' + (data && data.error ? data.error : ''));
  } catch (e) {
    _browserSetStatus('Action error: ' + e.message);
  }
}

function browserTakeover() {
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
  const url = _resolveUrl((urlEl && urlEl.value || '').trim());
  if (!task && !url) { showToast && showToast('Enter a task or URL first', 'error'); return; }
  if (urlEl && url) urlEl.value = url;

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
          if (ev.screenshot) _browserShow(ev.screenshot, ev.url, ev.title);
          if (ev.kind === 'done' || ev.kind === 'error') _browserLog(ev.kind, ev.text);
          else _browserLog('step', ev.text);
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
  try { await fetch('/api/browser/close', { method: 'POST', headers: _browserHeaders() }); }
  catch (_) {}
  const img = document.getElementById('browserScreenshot');
  const ph = document.getElementById('browserPlaceholder');
  if (img) { img.style.display = 'none'; img.src = ''; }
  if (ph) ph.style.display = 'flex';
  const log = document.getElementById('browserAiLog');
  if (log) log.innerHTML = '';
  _browserSetStatus('Session closed.');
  _tabs = []; _activeTab = null; _newTab();
}

document.addEventListener('DOMContentLoaded', () => {
  const back = document.getElementById('browserBack');
  const fwd = document.getElementById('browserFwd');
  const reload = document.getElementById('browserReload');
  if (back) back.addEventListener('click', () => browserAction('back'));
  if (fwd) fwd.addEventListener('click', () => browserAction('forward'));
  if (reload) reload.addEventListener('click', () => browserAction('reload'));
  const newBtn = document.getElementById('browserTabNew');
  if (newBtn) newBtn.addEventListener('click', () => _newTab());
  const bmAdd = document.getElementById('browserBmAdd');
  if (bmAdd) bmAdd.addEventListener('click', _addBookmark);
  const zoomIn = document.getElementById('browserZoomIn');
  const zoomOut = document.getElementById('browserZoomOut');
  if (zoomIn) zoomIn.addEventListener('click', () => _setZoom(_zoom + 10));
  if (zoomOut) zoomOut.addEventListener('click', () => _setZoom(_zoom - 10));
  setBrowserMode('standalone');
  _renderBookmarks();
  _newTab();
});

/* ==================== TABS + OMNI + BOOKMARKS + ZOOM ==================== */
