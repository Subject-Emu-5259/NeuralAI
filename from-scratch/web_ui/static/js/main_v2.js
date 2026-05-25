/**
 * NeuralAI Core UI Engine v6.3
 * Unified script handling Auth, Chat, Memory, Rules, Settings, Terminal and NeuralDrive.
 */

// ========================================
// GLOBAL STATE
// ========================================
let authToken = localStorage.getItem("neural_token");
let currentUser = null;
let currentConversationId = null;
let isStreaming = false;
let conversation = [];
let attachedFiles = {};
let uplinkEnabled = true;
let termSid = null;
let termPoll = null;
let authMode = "login";
let currentShell = "bash";
let searchMode = "all";
let userSettings = { theme: 'dark', neural_voice: 'Andrew', user_bio: '' };
let voiceWS = null;
let audioCtx = null;
let processor = null;
let micStream = null;
let audioQueue = [];
let isPlaying = false;

// ========================================
// AUTHENTICATION
// ========================================
async function initAuth() {
  if (!authToken) { showAuth(); return; }
  try {
    const r = await fetch("/api/user/me", {
      headers: { "Authorization": `Bearer ${authToken}` }
    });
    if (!r.ok) throw new Error();
    const data = await r.json();
    currentUser = data.user;
    const greeting = document.getElementById("userGreeting");
    if (greeting) greeting.textContent = `Hi, ${currentUser.username}`;
    document.getElementById("authOverlay").classList.add("hidden");
    loadConversations();
    loadUserProfile();
  } catch { logout(); }
}

function showAuth() {
  document.getElementById("authOverlay").classList.remove("hidden");
}

function logout() {
  localStorage.removeItem("neural_token");
  location.reload();
}

function toggleAuthMode() {
  authMode = authMode === "login" ? "signup" : "login";
  const title = document.getElementById("authTitle");
  const subtitle = document.getElementById("authSubtitle");
  const submit = document.getElementById("authSubmit");
  const toggle = document.getElementById("authToggle");
  const emailGroup = document.getElementById("emailGroup");
  const usernameGroup = document.getElementById("usernameGroup");
  const confirmGroup = document.getElementById("confirmPasswordGroup");

  if (authMode === "signup") {
    if (title) title.textContent = "Create Account";
    if (subtitle) subtitle.textContent = "Join the NeuralAI network and start building.";
    if (submit) submit.textContent = "Sign Up";
    if (toggle) toggle.innerHTML = "Already have an account? <span>Login</span>";
    if (emailGroup) emailGroup.classList.remove("hidden");
    if (usernameGroup) usernameGroup.classList.remove("hidden");
    if (confirmGroup) confirmGroup.classList.remove("hidden");
  } else {
    if (title) title.textContent = "Welcome Back";
    if (subtitle) subtitle.textContent = "Enter your credentials to access your cloud brain.";
    if (submit) submit.textContent = "Login";
    if (toggle) toggle.innerHTML = "Don't have an account? <span>Sign Up</span>";
    if (emailGroup) emailGroup.classList.remove("hidden");
    if (usernameGroup) usernameGroup.classList.add("hidden");
    if (confirmGroup) confirmGroup.classList.add("hidden");
  }
}

async function handleAuth() {
  const email = document.getElementById("authEmail")?.value.trim();
  const username = document.getElementById("authUsername")?.value.trim();
  const password = document.getElementById("authPassword")?.value.trim();
  const confirm = document.getElementById("authConfirmPassword")?.value.trim();

  if (!email || !password) return showToast("Email and password required", "error");
  if (authMode === "signup" && !username) return showToast("Username required for signup", "error");
  if (authMode === "signup" && password !== confirm) return showToast("Passwords do not match", "error");

  const url = authMode === "signup" ? "/api/auth/signup" : "/api/auth/login";
  const body = authMode === "signup" ? { username, password, email } : { email, password };

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Authentication failed");

    if (authMode === "signup") {
      showToast("Account created! Please login.", "success");
      toggleAuthMode();
    } else {
      localStorage.setItem("neural_token", data.token);
      authToken = data.token;
      showToast("Access granted", "success");
      document.getElementById("authOverlay").classList.add("hidden");
      if (data.user) {
        currentUser = data.user;
        const greeting = document.getElementById("userGreeting");
        if (greeting) greeting.textContent = `Hi, ${currentUser.username}`;
      }
      loadConversations();
    }
  } catch (err) {
    showToast(err.message, "error");
  }
}

// ========================================
// CHAT & CONVERSATIONS
// ========================================
async function loadConversations() {
  try {
    const res = await fetch('/api/conversations', {
      headers: { 'Authorization': `Bearer ${authToken}` }
    });
    const data = await res.json();
    renderConversationList(data.conversations || []);
  } catch (e) { console.error('Failed to load conversations', e); }
}

async function createNewConversation() {
  try {
    const res = await fetch('/api/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
      body: JSON.stringify({ title: 'New Conversation' })
    });
    if (!res.ok) throw new Error('Server error');
    const data = await res.json();
    if (!data.id) throw new Error('No ID returned');
    
    currentConversationId = data.id;
    conversation = [];
    document.getElementById('messages').innerHTML = '';
    const welcome = document.getElementById('welcomeScreen');
    if (welcome) welcome.style.display = 'flex';
    await loadConversations();
    return data.id;
  } catch (e) { 
    showToast('Failed to create chat: ' + e.message, 'error'); 
    throw e;
  }
}

async function loadConversation(id) {
  currentConversationId = id;
  try {
    const res = await fetch(`/api/conversations/${id}`, {
      headers: { 'Authorization': `Bearer ${authToken}` }
    });
    if (!res.ok) throw new Error('Failed to fetch chat');
    const data = await res.json();
    conversation = data.messages || [];
    const container = document.getElementById('messages');
    container.innerHTML = '';
    document.getElementById('welcomeScreen').style.display = 'none';
    conversation.forEach(m => addMsg(m.role, m.content));
    loadConversations();
  } catch (e) { showToast('Failed to load history', 'error'); }
}

async function deleteConversation(id) {
  if (!confirm('Purge this intelligence log permanently?')) return;
  try {
    await fetch(`/api/conversations/${id}`, {
      method: 'DELETE',
      headers: { 'Authorization': `Bearer ${authToken}` }
    });
    if (currentConversationId === id) {
      currentConversationId = null;
      conversation = [];
      document.getElementById('messages').innerHTML = '';
      document.getElementById('welcomeScreen').style.display = 'flex';
    }
    loadConversations();
  } catch (e) { showToast('Failed to delete', 'error'); }
}

async function renameConversation(id) {
  const newTitle = prompt('Enter new name for this chat:');
  if (!newTitle || newTitle.trim() === '') return;
  try {
    const res = await fetch(`/api/conversations/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
      body: JSON.stringify({ title: newTitle.trim() })
    });
    if (!res.ok) throw new Error('Rename failed');
    showToast('Chat renamed', 'success');
    loadConversations();
  } catch (e) { showToast('Failed to rename', 'error'); }
}

function renderConversationList(convs) {
  const list = document.getElementById('sidebarHistoryList');
  if (!list) return;
  list.innerHTML = convs.map(c => `
    <div class="history-item ${currentConversationId === c.id ? 'active' : ''}" onclick="loadConversation('${c.id}')">
      <span class="history-item-text">${escHtml(c.title)}</span>
      <div class="history-item-actions">
        <button class="history-item-rename" onclick="event.stopPropagation(); renameConversation('${c.id}')">✏️</button>
        <button class="history-item-delete" onclick="event.stopPropagation(); deleteConversation('${c.id}')">&times;</button>
      </div>
    </div>
  `).join('') || '<p style="font-size:12px;color:#555;padding:10px;">No logs found.</p>';
  
  // Update dropdown too
  const dropList = document.getElementById('historyDropdownList');
  if (dropList) {
    dropList.innerHTML = convs.slice(0, 5).map(c => `
      <div class="history-item" onclick="loadConversation('${c.id}'); hideHistoryDropdown();" style="border-radius:0;border-bottom:1px solid rgba(255,255,255,0.05);">
        <span class="history-item-text">${escHtml(c.title)}</span>
      </div>
    `).join('');
    const emptyMsg = document.getElementById('historyDropdownEmpty');
    if (emptyMsg) emptyMsg.style.display = convs.length === 0 ? 'block' : 'none';
  }
}

async function sendMessage(textOverride = null) {
  const input = document.getElementById('chatInput');
  const text = textOverride || input.value.trim();
  if (!text || isStreaming) return;

  try {
    if (!currentConversationId) {
      await createNewConversation();
    }
  } catch (e) {
    return; // Stop if conversation creation failed
  }

  if (!textOverride) {
    input.value = '';
    input.style.height = 'auto';
  }
  document.getElementById('welcomeScreen').style.display = 'none';

  addMsg('user', text);
  conversation.push({ role: 'user', content: text });
  isStreaming = true;
  const assistantMsg = addMsg('assistant', '');
  const bubble = assistantMsg.querySelector('.msg-bubble');
  bubble.innerHTML = '<div class="thinking-dots"><span></span><span></span><span></span></div>';

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
      body: JSON.stringify({ prompt: text, conversation_id: currentConversationId, messages: conversation })
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let full = '';
    bubble.innerHTML = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') break;
          try {
            const data = JSON.parse(raw);
            if (data.content) {
              full += data.content;
              bubble.innerHTML = fmt(full);
              const msgs = document.getElementById('messages');
              msgs.scrollTop = msgs.scrollHeight;
            }
          } catch {}
        }
      }
    }
    conversation.push({ role: 'assistant', content: full });
    
    // Send to Voice TTS if Live Voice is active
    if (typeof voiceWS !== 'undefined' && voiceWS && voiceWS.readyState === WebSocket.OPEN) {
      voiceWS.send(JSON.stringify({ type: 'text', data: full }));
    }
  } catch (e) {
    bubble.innerHTML = '<span style="color:#ff6b6b">Generation failed.</span>';
  } finally {
    isStreaming = false;
  }
}

function addMsg(role, content) {
  const container = document.getElementById('messages');
  if (!container) return document.createElement('div');
  const div = document.createElement('div');
  div.className = `msg ${role === 'assistant' ? 'ai' : 'user'}`;
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  div.innerHTML = `
    <div class="msg-meta"><span>${role === 'assistant' ? 'NeuralAI' : 'You'}</span><span class="msg-timestamp">${time}</span></div>
    <div class="msg-bubble">${content ? fmt(content) : ''}</div>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

// ========================================
// LIVE VOICE ENGINE (v2.0 - Multimodal Live)
// ========================================
async function toggleLiveVoice(show) {
  const overlay = document.getElementById('liveVoiceOverlay');
  const orb = overlay?.querySelector('.live-orb');
  
  if (show) {
    overlay?.classList.remove('hidden');
    try {
      await initLiveSession();
      orb?.classList.add('listening');
    } catch (err) {
      showToast('Live initialization failed: ' + err.message, 'error');
      toggleLiveVoice(false);
    }
  } else {
    overlay?.classList.add('hidden');
    orb?.classList.remove('listening', 'speaking');
    stopLiveSession();
  }
}

async function initLiveSession() {
  return new Promise(async (resolve, reject) => {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    if (audioCtx.state === 'suspended') await audioCtx.resume();

    // WebSocket to NeuralVoice (FastAPI)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname;
    
    // Resolve Voice Service Host
    let voiceHost = host.replace('neuralai-', 'neural-voice-');
    if (host === 'localhost' || host === '127.0.0.1') voiceHost = `${host}:5001`;

    console.log(`Connecting to Voice Service: ${voiceHost}`);
    voiceWS = new WebSocket(`${protocol}//${voiceHost}/ws`);

    voiceWS.onopen = () => {
      console.log('NeuralVoice Live Connected');
      
      // Send initial voice configuration
      const selectedVoice = document.getElementById('voiceSelection')?.value || userSettings.neural_voice || 'Andrew';
      voiceWS.send(JSON.stringify({ 
        type: 'config', 
        voice: selectedVoice 
      }));
      
      startMicCapture().then(resolve).catch(reject);
    };

    voiceWS.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'audio') {
        const orb = document.querySelector('.live-orb');
        orb?.classList.remove('listening');
        orb?.classList.add('speaking');
        updateLiveStatus('NeuralAI Speaking...');
        
        const audioData = base64ToUint8Array(data.data);
        queueAudioChunk(audioData);
      } else if (data.type === 'turn_complete') {
        console.log('Assistant turn complete');
      } else if (data.type === 'error') {
        showToast('Live Error: ' + data.message, 'error');
        toggleLiveVoice(false);
      }
    };

    voiceWS.onerror = (err) => {
      console.error('WS Error:', err);
      reject(new Error('Voice connection lost'));
    };

    voiceWS.onclose = () => {
      console.log('NeuralVoice Live Disconnected');
      stopLiveSession();
    };
  });
}

function base64ToUint8Array(base64) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function startMicCapture() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    showToast('Speech Recognition not supported in this browser.', 'error');
    toggleLiveVoice(false);
    return;
  }

  if (!window.recognition) {
    window.recognition = new SpeechRecognition();
    window.recognition.continuous = true;
    window.recognition.interimResults = true; // Enable interim for better "hearing" feedback
    
    window.recognition.onresult = (event) => {
      let finalTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          // Provide real-time feedback for "hearing"
          const interim = event.results[i][0].transcript;
          updateLiveStatus('Hearing: ' + interim);
        }
      }

      if (finalTranscript.trim()) {
        const transcript = finalTranscript.trim();
        updateLiveStatus('You: ' + transcript);
        
        // If voiceWS is active, we might want to send directly to it
        // but sendMessage handles the chat UI and model response too.
        sendMessage(transcript);
        
        // Visual feedback
        const orb = document.querySelector('.live-orb');
        orb?.classList.remove('listening');
        orb?.classList.add('processing');
      }
    };
    
    window.recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
      if (event.error === 'not-allowed') {
        showToast('Microphone access denied.', 'error');
        toggleLiveVoice(false);
      }
    };
    
    window.recognition.onend = () => {
      const overlay = document.getElementById('liveVoiceOverlay');
      if (overlay && !overlay.classList.contains('hidden')) {
        console.log('Restarting recognition...');
        setTimeout(() => {
          try { window.recognition.start(); } catch(e) { console.warn('Recognition restart failed:', e); }
        }, 300);
      }
    };
  }
  
  try { 
    window.recognition.stop(); // Stop if already running to avoid error
    setTimeout(() => {
      window.recognition.start();
      updateLiveStatus('NeuralAI Listening...');
    }, 200);
  } catch(e) {
    console.warn('Initial recognition start failed:', e);
  }
}

function float32ToInt16(buffer) {
  let l = buffer.length;
  let buf = new Int16Array(l);
  while (l--) {
    let s = Math.max(-1, Math.min(1, buffer[l]));
    buf[l] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return buf;
}

function queueAudioChunk(data) {
  // Simple PCM playback queue
  const int16Data = new Int16Array(data.buffer);
  const float32Data = new Float32Array(int16Data.length);
  for (let i = 0; i < int16Data.length; i++) {
    float32Data[i] = int16Data[i] / 32768.0;
  }
  
  const audioBuffer = audioCtx.createBuffer(1, float32Data.length, 16000);
  audioBuffer.getChannelData(0).set(float32Data);
  
  const source = audioCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioCtx.destination);
  source.start(nextPlayTime);
  
  nextPlayTime = Math.max(audioCtx.currentTime, nextPlayTime) + audioBuffer.duration;
}

function stopLiveSession() {
  if (window.recognition) {
    window.recognition.stop();
  }
  if (voiceWS) {
    voiceWS.close();
    voiceWS = null;
  }
  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }
  if (processor) {
    processor.disconnect();
    processor = null;
  }
}

function updateLiveStatus(text) {
  const statusEl = document.getElementById('liveVoiceStatus');
  if (statusEl) statusEl.innerText = text;
}

// ========================================
// SEARCH BAR & TABS
// ========================================
function handleSearch() {
  const queryInput = document.getElementById("queryInput");
  const text = queryInput?.value.trim();
  if (!text) return;
  
  queryInput.value = "";
  hideHistoryDropdown();

  if (searchMode === "files") {
    switchTab('files');
    const fileSearch = document.getElementById("fileSearch");
    if (fileSearch) {
      fileSearch.value = text;
      filterFiles();
    }
  } else if (searchMode === "system") {
    switchTab('terminal');
    const termInput = document.getElementById("terminalInput");
    if (termInput) {
      termInput.value = text;
      executeTerminalCmd();
    }
  } else {
    switchTab('chat');
    sendMessage(text);
  }
}

function showHistoryDropdown() {
  const dropdown = document.getElementById('historyDropdown');
  if (dropdown) dropdown.style.display = 'block';
}

function hideHistoryDropdown() {
  const dropdown = document.getElementById('historyDropdown');
  if (dropdown) dropdown.style.display = 'none';
}

function hideHistoryDropdownDelayed() {
  setTimeout(hideHistoryDropdown, 200);
}

// ========================================
// TERMINAL ENGINE
// ========================================
async function initTerm() {
  if (termSid) return;
  try {
    const res = await fetch("/api/terminal/create", { method: "POST", headers: { "Authorization": `Bearer ${authToken}` } });
    if (!res.ok) throw new Error(`Server responded ${res.status}`);
    const data = await res.json();
    if (!data.session_id) throw new Error('No session ID returned');
    termSid = data.session_id;
    termOut(`Connected to NeuralAI High-Velocity Console [SID: ${termSid}]`, 'info');
  } catch (err) { termOut(`Failed to establish Neural Uplink: ${err.message}`, 'error'); }
}

async function executeTerminalCmd() {
  const input = document.getElementById("terminalInput");
  const cmd = input?.value.trim();
  if (!cmd || !termSid) return;
  
  termOut(`<span class="terminal-prompt">${currentShell === 'bash' ? 'root@Neural:~#' : (currentShell === 'python' ? '>>>' : 'node>')}</span> ${cmd}`, 'input');
  input.value = "";

  if (cmd.toLowerCase() === 'help') {
    termOut("NeuralAI Terminal Help:\n- bash: Default shell\n- python: Python runtime\n- node: JavaScript runtime\n- clear: Clear screen\n- restart: Reset session", 'info');
    return;
  }
  
  try {
    const res = await fetch(`/api/terminal/${termSid}/send`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Authorization": `Bearer ${authToken}` },
      body: JSON.stringify({ command: cmd, shell: currentShell })
    });
    if (!res.ok) throw new Error(`Server responded ${res.status}`);
    const data = await res.json();
    termOut(data.output || data.error || 'No output');
  } catch (err) { termOut(`Neural Uplink Error: ${err.message}`, 'error'); }
}

function termOut(txt, cls = '') {
  const el = document.getElementById("terminalOutput");
  if (!el) return;
  const div = document.createElement('div');
  div.className = `term-line ${cls}`;
  div.innerHTML = txt;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function clearTerm() {
  const el = document.getElementById("terminalOutput");
  if (el) el.innerHTML = '<div class="term-line info">Console cleared.</div>';
}

function restartTerm() {
  termSid = null;
  clearTerm();
  initTerm();
}

function toggleHistory() {
  const panel = document.getElementById("historyPanel");
  panel?.classList.toggle("hidden");
}

function showSnippets() {
  document.getElementById("snippetsPanel")?.classList.remove("hidden");
}

function hideSnippets() {
  document.getElementById("snippetsPanel")?.classList.add("hidden");
}

function insertCmd(cmd) {
  const input = document.getElementById("terminalInput");
  if (input) {
    input.value = cmd;
    input.focus();
  }
}

// ====================
// NEURALDRIVE FILES
// ====================
async function loadFiles() {
  const container = document.getElementById("filesGrid");
  if (!container) return;
  container.innerHTML = '<div class="loading-shimmer">Scanning NeuralDrive...</div>';
  try {
    const res = await fetch("/api/files", { headers: { "Authorization": `Bearer ${authToken}` } });
    const data = await res.json();
    window.neuralFiles = data.files || [];
    renderFiles(window.neuralFiles);
  } catch { container.innerHTML = "Error accessing NeuralDrive."; }
}

function renderFiles(files) {
  const container = document.getElementById("filesGrid");
  if (!container) return;
  if (!files || files.length === 0) {
    container.innerHTML = '<div class="empty-state">NeuralDrive is empty.</div>';
    return;
  }
  container.innerHTML = files.map(f => `
    <div class="file-card" onclick="previewFile('${f.type}', '${f.name}')">
      <div class="file-info">
        <div class="file-icon">${f.name.endsWith('.png') || f.name.endsWith('.jpg') ? '🖼️' : '📄'}</div>
        <div class="file-name">${f.name}</div>
      </div>
      <div class="file-actions">
        <button class="file-action-btn" onclick="event.stopPropagation(); window.open('/api/files/${f.type}/${f.name}', '_blank')">⬇️</button>
      </div>
    </div>
  `).join('');
}

function filterFiles() {
  const query = document.getElementById("fileSearch")?.value.toLowerCase() || "";
  const filtered = (window.neuralFiles || []).filter(f => f.name.toLowerCase().includes(query));
  renderFiles(filtered);
}

function previewFile(folder, filename) {
  const url = `/api/files/${folder}/${filename}`;
  window.open(url, '_blank');
}

// ====================
// SETTINGS & PERSISTENCE
// ====================
async function loadUserProfile() {
  try {
    const res = await fetch('/api/user/me', { headers: { 'Authorization': `Bearer ${authToken}` } });
    const data = await res.json();
    if (data.user) {
      currentUser = data.user;
      if (document.getElementById('profileFirstName')) document.getElementById('profileFirstName').value = data.user.first_name || '';
      if (document.getElementById('profileLastName')) document.getElementById('profileLastName').value = data.user.last_name || '';
      if (document.getElementById('profileEmail')) document.getElementById('profileEmail').value = data.user.email || '';
      if (document.getElementById('profileUsername')) document.getElementById('profileUsername').value = data.user.username || '';
      if (document.getElementById('userBioInput')) document.getElementById('userBioInput').value = data.user.bio || '';
      const initial = document.getElementById('profileInitial');
      if (initial) initial.textContent = (data.user.username || 'U')[0].toUpperCase();
    }
    loadBio();
    loadMemoryList();
    loadRulesList();
    loadSettings();
  } catch {}
}

async function saveFullProfile() {
  const data = {
    first_name: document.getElementById('profileFirstName')?.value,
    last_name: document.getElementById('profileLastName')?.value,
    email: document.getElementById('profileEmail')?.value,
    username: document.getElementById('profileUsername')?.value,
    bio: document.getElementById('userBioInput')?.value
  };
  
  try {
    const res = await fetch('/api/user/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
      body: JSON.stringify(data)
    });
    if (res.ok) showToast('Identity Vault Updated', 'success');
    else throw new Error();
  } catch { showToast('Update failed', 'error'); }
}

async function loadSettings() {
  try {
    const res = await fetch('/api/settings', { headers: { 'Authorization': `Bearer ${authToken}` } });
    const data = await res.json();
    userSettings = data.settings || userSettings;
    if (userSettings.neural_voice) {
      const sel = document.getElementById('voiceSelection');
      if (sel) sel.value = userSettings.neural_voice;
    }
  } catch {}
}

async function saveVoicePreference() {
  const voice = document.getElementById('voiceSelection')?.value;
  if (!voice) return;
  try {
    await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
      body: JSON.stringify({ neural_voice: voice })
    });
    showToast('Voice preference saved', 'success');
  } catch {}
}

async function loadBio() {
  try {
    const res = await fetch('/api/settings', { headers: { 'Authorization': `Bearer ${authToken}` } });
    const data = await res.json();
    const input = document.getElementById('userBioInput');
    if (input) input.value = data.settings?.user_bio || '';
  } catch {}
}

async function addMemoryFromTab() {
  const input = document.getElementById('memoryInput');
  const fact = input?.value.trim();
  if (!fact) return;
  await fetch('/api/memory', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
    body: JSON.stringify({ fact })
  });
  input.value = '';
  loadMemoryList();
  showToast('Memory stored', 'success');
}

async function loadMemoryList() {
  try {
    const res = await fetch('/api/memory', { headers: { 'Authorization': `Bearer ${authToken}` } });
    const data = await res.json();
    const list = document.getElementById('memoryList');
    if (!list) return;
    list.innerHTML = (data.facts || []).map(f => `
      <div class="memory-item"><span>${escHtml(f.fact)}</span><button onclick="deleteMemoryItem(${f.id})">&times;</button></div>
    `).join('') || '<p style="font-size:12px;color:#888;padding:10px;">No dynamic memories stored.</p>';
  } catch {}
}

async function deleteMemoryItem(id) {
  await fetch(`/api/memory/${id}`, { method: 'DELETE', headers: { 'Authorization': `Bearer ${authToken}` } });
  loadMemoryList();
}

async function addRuleFromTab() {
  const input = document.getElementById('ruleInput');
  const rule = input?.value.trim();
  if (!rule) return;
  await fetch('/api/rules', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
    body: JSON.stringify({ rule })
  });
  input.value = '';
  loadRulesList();
}

async function loadRulesList() {
  try {
    const res = await fetch('/api/rules', { headers: { 'Authorization': `Bearer ${authToken}` } });
    const data = await res.json();
    const list = document.getElementById('rulesList');
    if (!list) return;
    list.innerHTML = (data.rules || []).map(r => `
      <div class="rule-item">
        <span>${escHtml(r.rule)}</span>
        <button class="rule-toggle ${r.is_active ? 'on' : 'off'}" onclick="toggleRuleItem(${r.id})">${r.is_active ? 'ON' : 'OFF'}</button>\n        <button onclick="deleteRuleItem(${r.id})">&times;</button>
      </div>
    `).join('') || '<p style="font-size:12px;color:#888;padding:10px;">No behavioral rules defined.</p>';
  } catch {}
}

async function toggleRuleItem(id) {
  await fetch(`/api/rules/${id}/toggle`, { method: 'POST', headers: { 'Authorization': `Bearer ${authToken}` } });
  loadRulesList();
}

async function deleteRuleItem(id) {
  await fetch(`/api/rules/${id}`, { method: 'DELETE', headers: { 'Authorization': `Bearer ${authToken}` } });
  loadRulesList();
}

// ========================================
// UX & UTILS
// ========================================
function switchTab(tabName) {
  document.querySelectorAll('.nav-item').forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tabName));
  document.querySelectorAll('[id^="tab-"]').forEach(tab => tab.classList.add('hidden'));
  const activeTab = document.getElementById('tab-' + tabName);
  if (activeTab) activeTab.classList.remove('hidden');
  
  if (tabName === 'terminal') initTerm();
  if (tabName === 'files') loadFiles();
  if (tabName === 'settings') loadUserProfile();
}

function toggleDarkMode() {
  document.body.classList.toggle('dark-mode');
  const isDark = document.body.classList.contains('dark-mode');
  localStorage.setItem('neural_theme', isDark ? 'dark' : 'light');
}

function showToast(msg, type = 'info') {
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

function escHtml(text) {
  if (!text) return '';
  const p = document.createElement('p');
  p.textContent = text;
  return p.innerHTML;
}

function fmt(text) {
  let out = escHtml(text);
  out = out.replace(/\n/g, '<br>');
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
  return out;
}

function closeOnboarding() {
  const overlay = document.getElementById("onboardingOverlay");
  if (overlay) {
    overlay.classList.remove("visible");
    localStorage.setItem("neural_onboarded", "true");
  }
}

// ========================================
// INITIALIZATION
// ========================================
document.addEventListener('DOMContentLoaded', () => {
  initAuth();
  
  // Onboarding Check
  if (!localStorage.getItem("neural_onboarded")) {
    document.getElementById("onboardingOverlay")?.classList.add("visible");
  }

  // Theme
  if (localStorage.getItem('neural_theme') === 'dark' || !localStorage.getItem('neural_theme')) {
    document.body.classList.add('dark-mode');
    localStorage.setItem('neural_theme', 'dark');
  }

  // Listeners
  document.getElementById('sendBtn')?.addEventListener('click', () => sendMessage());
  document.getElementById('chatInput')?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  
  // Voice Activation
  document.getElementById('voiceBtn')?.addEventListener('click', () => toggleLiveVoice(true));
  document.getElementById('closeLiveBtn')?.addEventListener('click', () => toggleLiveVoice(false));
  
  document.getElementById('searchBtn')?.addEventListener('click', handleSearch);
  document.getElementById('queryInput')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') handleSearch();
  });

  document.querySelectorAll('.search-tabs .tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.search-tabs .tab').forEach(t => t.classList.remove('active'));
      btn.classList.add('active');
      searchMode = btn.dataset.mode;
    });
  });

  document.getElementById('authSubmit')?.addEventListener('click', handleAuth);
  document.getElementById('authToggle')?.addEventListener('click', toggleAuthMode);
  
  document.getElementById('terminalInput')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') executeTerminalCmd();
  });

  document.querySelectorAll('.terminal-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.terminal-tab').forEach(t => t.classList.remove('active'));
      btn.classList.add('active');
      currentShell = btn.dataset.shell;
      const prompt = document.getElementById("terminalPrompt");
      if (prompt) prompt.textContent = currentShell === 'bash' ? 'root@Neural:~#' : (currentShell === 'python' ? '>>>' : 'node>');
    });
  });

  document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Prompt Cards
  document.querySelectorAll('.prompt-card').forEach(card => {
    card.addEventListener('click', () => {
      sendMessage(card.dataset.prompt);
    });
  });
});

// Global Exports
window.switchTab = switchTab;
window.loadConversation = loadConversation;
window.deleteConversation = deleteConversation;
window.renameConversation = renameConversation;
window.logout = logout;
window.toggleDarkMode = toggleDarkMode;
window.saveFullProfile = saveFullProfile;
window.saveVoicePreference = saveVoicePreference;
window.addMemoryFromTab = addMemoryFromTab;
window.deleteMemoryItem = deleteMemoryItem;
window.addRuleFromTab = addRuleFromTab;
window.deleteRuleItem = deleteRuleItem;
window.toggleRuleItem = toggleRuleItem;
window.previewFile = previewFile;
window.clearTerm = clearTerm;
window.restartTerm = restartTerm;
window.toggleHistory = toggleHistory;
window.showSnippets = showSnippets;
window.hideSnippets = hideSnippets;
window.insertCmd = insertCmd;
window.showHistoryDropdown = showHistoryDropdown;
window.hideHistoryDropdownDelayed = hideHistoryDropdownDelayed;
window.filterFiles = filterFiles;
window.handleSearch = handleSearch;
window.toggleLiveVoice = toggleLiveVoice;
window.closeOnboarding = closeOnboarding;
