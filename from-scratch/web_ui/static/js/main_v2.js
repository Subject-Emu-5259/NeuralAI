/**
 * NeuralAI Core UI Engine v6.3
 * Unified script handling Auth, Chat, Memory, Rules, Settings, Terminal and NeuralDrive.
 */

// ========================================
// GLOBAL STATE
// ========================================
let authToken = window.localStorage.getItem("neural_token");
let currentUser = null;
let currentConversationId = null;
let isStreaming = false;
let abortStream = false;
let lastSeen = {};
let conversation = [];
let attachedFiles = {};
let uplinkEnabled = false;
let termSid = null;
let termPoll = null;
let authMode = "login";
let currentShell = "bash";
let searchMode = "all";
let isInitializingTerm = false;
let userSettings = { theme: 'dark', neural_voice: 'Andrew', user_bio: '' };
let voiceWS = null;
let audioCtx = null;
let processor = null;
let micStream = null;
let audioQueue = [];
let isPlaying = false;
let nextPlayTime = 0;
let voiceReconnectTimer = null;

// ========================================
// AUTHENTICATION
// ========================================
async function initAuth() {
  console.log("Initializing Auth state...");
  const logoutBtn = document.getElementById("logoutBtn");
  const authOverlay = document.getElementById("authOverlay");
  
  if (!authToken) { 
    console.log("No token found, showing auth overlay.");
    if (logoutBtn) logoutBtn.classList.add("hidden");
    if (authOverlay) authOverlay.classList.remove("hidden");
    return; 
  }
  
  try {
    const r = await fetch("/api/user/me", {
      headers: { "Authorization": `Bearer ${authToken}` }
    });
    
    if (!r.ok) {
        console.warn("Auth token invalid or expired.");
        throw new Error("Invalid token");
    }
    
    const data = await r.json();
    currentUser = data.user;
    
    const greeting = document.getElementById("userGreeting");
    if (greeting) {
        greeting.textContent = currentUser.account_type === "guest" 
            ? `Guest Session` 
            : `Hi, ${currentUser.username}`;
    }
    
    if (logoutBtn) logoutBtn.classList.remove("hidden");
    if (authOverlay) authOverlay.classList.add("hidden");
    
    loadConversations();
    loadUserProfile();
  } catch (err) { 
    console.error("Auth check failed:", err);
    logout(); 
  }
}

function showAuth() {
  const authOverlay = document.getElementById("authOverlay");
  if (authOverlay) authOverlay.classList.remove("hidden");
  const logoutBtn = document.getElementById("logoutBtn");
  if (logoutBtn) logoutBtn.classList.add("hidden");
}

function logout() {
  console.log("Logging out...");
  window.localStorage.removeItem("neural_token");
  authToken = null;
  currentUser = null;
  
  const logoutBtn = document.getElementById("logoutBtn");
  if (logoutBtn) logoutBtn.classList.add("hidden");
  
  const greeting = document.getElementById("userGreeting");
  if (greeting) greeting.textContent = "";
  
  const authOverlay = document.getElementById("authOverlay");
  if (authOverlay) authOverlay.classList.remove("hidden");
  
  // Clear local state
  conversation = [];
  currentConversationId = null;
  document.getElementById('messages').innerHTML = '';
  
  // Optional: location.href = "/"; // Only if we want a full clean state
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
  const identity = document.getElementById("authEmail")?.value.trim();
  const username = document.getElementById("authUsername")?.value.trim();
  const password = document.getElementById("authPassword")?.value.trim();
  const confirm = document.getElementById("authConfirmPassword")?.value.trim();

  if (!identity || !password) return showToast("Credentials required", "error");
  if (authMode === "signup" && !username) return showToast("Username required for signup", "error");
  if (authMode === "signup" && password !== confirm) return showToast("Passwords do not match", "error");

  const url = authMode === "signup" ? "/api/auth/signup" : "/api/auth/login";
  const body = authMode === "signup" ? { username, password, email: identity } : { username: identity, password };

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
      window.localStorage.setItem("neural_token", data.token);
      authToken = data.token;
      showToast("Access granted", "success");
      document.getElementById("authOverlay").classList.add("hidden");
      const logoutBtn = document.getElementById("logoutBtn");
      if (logoutBtn) logoutBtn.classList.remove("hidden");
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

async function handleMaestroAuth() {
  showToast("Maestro Auth Triggered", "info");
  const codeEl = document.getElementById("maestroCode");
  const code = codeEl?.value?.trim();
  if (!code) return showToast("Maestro Student ID required", "error");

  try {
    showToast("Validating Maestro Pattern...", "info");
    const res = await fetch("/api/auth/maestro", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ invite_code: code })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Access denied");

    window.localStorage.setItem("neural_token", data.token);
    authToken = data.token;
    showToast(`Welcome, Maestrian ${data.user?.username || 'Student'}`, "success");
    document.getElementById("authOverlay")?.classList.add("hidden");
    document.getElementById("logoutBtn")?.classList.remove("hidden");
    
    if (data.user) {
      currentUser = data.user;
      const greeting = document.getElementById("userGreeting");
      if (greeting) greeting.textContent = `Hi, ${currentUser.username}`;
    }
    loadConversations();
  } catch (err) {
    console.error("Maestro Auth Error:", err);
    showToast(err.message || "Authentication failed", "error");
  }
}

async function handleGuestAuth() {
  showToast("Guest Auth Triggered", "info");
  try {
    showToast("Generating Guest Session...", "info");
    const res = await fetch("/api/auth/guest", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Guest access failed");

    window.localStorage.setItem("neural_token", data.token);
    authToken = data.token;
    showToast(`Session active as ${data.user?.username || 'Guest'}`, "success");
    document.getElementById("authOverlay")?.classList.add("hidden");
    document.getElementById("logoutBtn")?.classList.remove("hidden");
    
    if (data.user) {
      currentUser = data.user;
      const greeting = document.getElementById("userGreeting");
      if (greeting) greeting.textContent = `Guest Session: ${currentUser.username}`;
    }
    loadConversations();
  } catch (err) {
    console.error("Guest Auth Error:", err);
    showToast(err.message || "Authentication failed", "error");
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
    // Support both {conversations: []} and direct array []
    const convs = Array.isArray(data) ? data : (data.conversations || []);
    renderConversationList(convs);
  } catch (e) { console.error('Failed to load conversations', e); }
}

async function createNewConversation() {
  try {
    const res = await fetch('/api/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
      body: JSON.stringify({ title: 'New Conversation' })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Server error');
    }
    const data = await res.json();
    if (!data.id) throw new Error('No ID returned');

    currentConversationId = data.id;
    conversation = [];
    const msgsEl = document.getElementById('messages');
    if (msgsEl) msgsEl.innerHTML = '';
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
  lastSeen[id] = new Date().toISOString();
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
  list.innerHTML = convs.map(c => {
    const isUnread = c.id !== currentConversationId && c.updated_at && lastSeen[c.id] && c.updated_at > lastSeen[c.id];
    return `
    <div class="history-item ${currentConversationId === c.id ? 'active' : ''}" onclick="loadConversation('${c.id}')">
      <span class="history-item-text">${escHtml(c.title)}</span>
      ${isUnread ? '<span class="unread-badge">NEW</span>' : ''}
      <div class="history-item-actions">
        <button class="history-item-rename" onclick="event.stopPropagation(); renameConversation('${c.id}')">✏️</button>
        <button class="history-item-delete" onclick="event.stopPropagation(); deleteConversation('${c.id}')">&times;</button>
      </div>
    </div>
  `}).join('') || '<p style="font-size:12px;color:#555;padding:10px;">No logs found.</p>';

  // Track last-seen timestamps for unread detection
  convs.forEach(c => { if (!lastSeen[c.id]) lastSeen[c.id] = c.updated_at; });

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
  console.log("SendMessage called. Override:", textOverride);
  const input = document.getElementById('chatInput');
  const text = textOverride || (input ? input.value.trim() : "");
  
  if (!text) {
    console.warn("SendMessage: No text to send");
    return;
  }
  
  if (isStreaming) {
    console.warn("SendMessage: Already streaming");
    return;
  }

  try {
    if (!currentConversationId) {
      console.log("No current conversation. Creating one...");
      await createNewConversation();
    }
  } catch (e) {
    console.error("Conversation creation failed:", e);
    showToast("Failed to initialize intelligence log", "error");
    return; 
  }

  if (!textOverride && input) {
    input.value = '';
    input.style.height = 'auto';
  }
  
  const welcome = document.getElementById('welcomeScreen');
  if (welcome) welcome.style.display = 'none';

  addMsg('user', text);
  conversation.push({ role: 'user', content: text });
  isStreaming = true;
  
  const sendBtn = document.getElementById('sendBtn');
  if (sendBtn) {
    sendBtn.disabled = true;
    sendBtn.innerHTML = '●●●';
  }

  const assistantMsg = addMsg('assistant', '');
  const bubble = assistantMsg.querySelector('.msg-bubble');
  bubble.innerHTML = '<div class="thinking-dots"><span></span><span></span><span></span></div>';
  const typingLabel = assistantMsg.querySelector('.typing-label');
  if (typingLabel) typingLabel.style.display = 'block';
  abortStream = false;
  showStopButton(true);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
      body: JSON.stringify({ prompt: text, conversation_id: currentConversationId, messages: conversation, use_uplink: uplinkEnabled })
    });
    
    if (!res.ok) {
       const errData = await res.json().catch(() => ({}));
       throw new Error(errData.error || `Server responded with ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let full = '';
    let firstToken = true;
    bubble.innerHTML = '';

    while (true) {
      if (abortStream) {
        await fetch('/api/chat/stop', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
          body: JSON.stringify({ conversation_id: currentConversationId })
        }).catch(() => {});
        break;
      }
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
              if (firstToken) {
                firstToken = false;
                if (typingLabel) typingLabel.style.display = 'none';
              }
              full += data.content;
              // Render tool-cards when an uplink agent emits a 🔧 marker
              if (data.content.includes('🔧') || full.includes('🔧 NeuralAI is processing tool')) {
                bubble.innerHTML = renderToolCard(full);
              } else {
                bubble.innerHTML = fmt(full);
              }
              const msgs = document.getElementById('messages');
              msgs.scrollTop = msgs.scrollHeight;
            }
          } catch {}
        }
      }
    }
    conversation.push({ role: 'assistant', content: full });
    
    // Auto-update 'Recent Intelligence' (sidebar) on first reply to pick up the auto-generated title
    if (conversation.length <= 2) {
      console.log("First interaction complete. Refreshing sidebar...");
      setTimeout(loadConversations, 500); // Small delay to allow DB commit
    }
    
    // Send to Voice TTS if Live Voice is active
    if (voiceWS && voiceWS.readyState === WebSocket.OPEN) {
      console.log('Sending text to voice engine:', full.substring(0, 50) + '...');
      voiceWS.send(JSON.stringify({ type: 'text', data: full }));
    } else {
      console.warn('Voice WebSocket is not open. State:', voiceWS ? voiceWS.readyState : 'null');
      if (document.getElementById('liveVoiceOverlay') && !document.getElementById('liveVoiceOverlay').classList.contains('hidden')) {
        showToast('Voice connection lost. Attempting to reconnect...', 'info');
        initLiveSession(); // Try to re-init if overlay is visible
      }
    }
  } catch (e) {
    console.error("Chat Error:", e);
    bubble.innerHTML = `<span style="color:#ff6b6b">Generation failed: ${e.message}</span>`;
  } finally {
    isStreaming = false;
    abortStream = false;
    showStopButton(false);
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) {
      sendBtn.disabled = false;
      sendBtn.innerHTML = 'Send';
    }
  }
}

function showStopButton(show) {
  const stopBtn = document.getElementById('stopBtn');
  if (stopBtn) stopBtn.style.display = show ? 'inline-flex' : 'none';
}

async function generateImageFromPrompt(prompt) {
  if (isStreaming) return;
  showToast('Generating image…', 'info');
  try {
    const res = await fetch('/api/image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
      body: JSON.stringify({ prompt })
    });
    const data = await res.json();
    if (data.success && data.image_url) {
      addMsg('assistant', `🖼️ **${escHtml(prompt)}**\n\n![generated](${data.image_url})`);
      showToast('Image ready', 'success');
    } else {
      showToast('Image generation failed: ' + (data.error || 'unknown'), 'error');
    }
  } catch (e) {
    showToast('Image generation failed: ' + e.message, 'error');
  }
}

async function runCodeSnippet(code) {
  if (isStreaming) return;
  showToast('Running code…', 'info');
  try {
    const res = await fetch('/api/execute/code', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
      body: JSON.stringify({ code, language: 'python' })
    });
    const data = await res.json();
    const out = data.success ? data.output : (data.error || 'No output');
    addMsg('assistant', '```python\n' + out + '\n```');
  } catch (e) {
    showToast('Code execution failed: ' + e.message, 'error');
  }
}

function renderToolCard(text) {
  // Extract the 🔧 processing line and render a styled tool-card
  const lines = text.split('\n');
  const toolLine = lines.find(l => l.includes('🔧'));
  if (toolLine) {
    const label = toolLine.replace('🔧', '').replace('NeuralAI is processing tool:', '').trim();
    return `<div class="tool-card"><span class="tool-icon">🔧</span><span class="tool-text">Processing tool: ${escHtml(label)}</span></div>` + fmt(lines.filter(l => l !== toolLine).join('\n'));
  }
  return fmt(text);
}

function updateModelStatus() {
  fetch('/api/health').then(r => r.json()).then(d => {
    const dot = document.getElementById('uplinkDot');
    const label = document.getElementById('uplinkLabel');
    if (dot) dot.className = 'status-dot ' + (d.status === 'ready' ? 'online' : 'offline');
    if (label) label.textContent = d.status === 'ready' ? 'Ready' : (d.status || 'Offline');
  }).catch(() => {
    const dot = document.getElementById('uplinkDot');
    const label = document.getElementById('uplinkLabel');
    if (dot) dot.className = 'status-dot offline';
    if (label) label.textContent = 'Offline';
  });
}

function addMsg(role, content) {
  const container = document.getElementById('messages');
  if (!container) return document.createElement('div');
  const div = document.createElement('div');
  div.className = `msg ${role === 'assistant' ? 'ai' : 'user'}`;
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  div.innerHTML = `
    <div class="msg-meta"><span>${role === 'assistant' ? 'NeuralAI' : 'You'}</span><span class="msg-timestamp">${time}</span></div>
    ${role === 'assistant' ? '<div class="typing-label" style="display:none;font-size:12px;color:#8a8a9a;margin-bottom:4px;">NeuralAI is typing…</div>' : ''}
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
    try {
      if (voiceReconnectTimer) clearTimeout(voiceReconnectTimer);
      
      if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }
      
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.hostname;
      
      // Voice WebSocket connects through the main service proxy at /voice/ws
      // This works universally: localhost, ZO Computer, Cloudflare, etc.
      const voiceOrigin = window.location.origin.replace('https:', 'wss:').replace('http:', 'ws:');
      const voiceUrl = `${voiceOrigin}/voice/ws`;
      
      console.log(`[Voice] Connecting to: ${voiceUrl}`);
      showToast('Connecting to NeuralVoice Engine...', 'info');
      
      if (voiceWS) {
        voiceWS.onclose = null;
        voiceWS.close();
      }
      
      try {
        voiceWS = new WebSocket(voiceUrl);
      } catch (wsErr) {
        console.error("[Voice] WebSocket constructor failed:", wsErr);
        throw new Error("Failed to create connection: " + wsErr.message);
      }

      voiceWS.onopen = () => {
        console.log('[Voice] WebSocket Connected');
        showToast('NeuralVoice Engine Online', 'success');
        const selectedVoice = document.getElementById('voiceSelection')?.value || userSettings.neural_voice || 'Andrew';
        voiceWS.send(JSON.stringify({ type: 'config', voice: selectedVoice }));
        startMicCapture().then(resolve).catch(reject);
      };

      voiceWS.onmessage = async (event) => {
        console.log("[Voice] Message received:", event.data.substring(0, 100));
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'audio') {
            const orb = document.querySelector('.live-orb');
            orb?.classList.remove('listening');
            orb?.classList.add('speaking');
            updateLiveStatus('NeuralAI Speaking...');
            
            if (data.data) {
              const audioData = base64ToUint8Array(data.data);
              queueAudioChunk(audioData, data.sampleRate || 16000);
            }
          } else if (data.type === 'turn_complete') {
            const now = audioCtx.currentTime;
            const delay = Math.max(0, (nextPlayTime - now) * 1000);
            setTimeout(() => {
              const orb = document.querySelector('.live-orb');
              if (orb && orb.classList.contains('speaking')) {
                orb.classList.remove('speaking', 'processing');
                orb.classList.add('listening');
                updateLiveStatus('NeuralAI Listening...');
              }
            }, delay);
          } else if (data.type === 'error') {
            console.error('[Voice] Service Error:', data.message);
            showToast('Voice Error: ' + data.message, 'error');
          }
        } catch (err) {
          console.error('[Voice] Handler Error:', err);
        }
      };

      voiceWS.onclose = (e) => {
        console.log(`[Voice] WebSocket Closed (Code: ${e.code}, Reason: ${e.reason})`);
        voiceWS = null;
        if (e.code !== 1000 && document.getElementById('liveVoiceOverlay') && !document.getElementById('liveVoiceOverlay').classList.contains('hidden')) {
          showToast('Voice connection interrupted. Reconnecting...', 'warning');
          voiceReconnectTimer = setTimeout(initLiveSession, 3000);
        }
      };

      voiceWS.onerror = (err) => {
        console.error('[Voice] WebSocket Error:', err);
        showToast('NeuralVoice connection failed.', 'error');
      };
    } catch (err) {
      console.error("[Voice] Initialization failed:", err);
      reject(err);
    }
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
    console.log("[Voice] Initializing SpeechRecognition instance");
    window.recognition = new SpeechRecognition();
    window.recognition.continuous = true;
    window.recognition.interimResults = true; 
    window.recognition.lang = 'en-US'; // Set explicit language
    
    window.recognition.onstart = () => {
      console.log('[Voice] STT Engine Started');
      updateLiveStatus('NeuralAI Listening...');
      showToast('Microphone Active - Start Speaking', 'success');
      const orb = document.querySelector('.live-orb');
      if (orb) orb.classList.add('listening');
    };

    window.recognition.onspeechstart = () => {
      console.log('[Voice] Speech detected');
      updateLiveStatus('Hearing voice...');
    };

    window.recognition.onresult = (event) => {
      let finalTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          const interim = event.results[i][0].transcript;
          console.log("[Voice] Interim transcript:", interim);
          updateLiveStatus('Hearing: ' + interim);
        }
      }

      if (finalTranscript.trim()) {
        const transcript = finalTranscript.trim();
        console.log("[Voice] Final transcript:", transcript);
        updateLiveStatus('You: ' + transcript);
        sendMessage(transcript);
        
        const orb = document.querySelector('.live-orb');
        orb?.classList.remove('listening');
        orb?.classList.add('processing');
      }
    };
    
    window.recognition.onerror = (event) => {
      console.error("[Voice] STT Error:", event.error);
      if (event.error === 'not-allowed') {
        showToast('Microphone access denied. Check browser settings.', 'error');
        toggleLiveVoice(false);
      } else if (event.error === 'network') {
        showToast('STT Network Error.', 'error');
      }
    };
    
    window.recognition.onend = () => {
      console.log("[Voice] STT Engine Ended");
      const overlay = document.getElementById('liveVoiceOverlay');
      if (overlay && !overlay.classList.contains('hidden')) {
        console.log('[Voice] Attempting auto-restart...');
        setTimeout(() => {
          try { 
            if (overlay && !overlay.classList.contains('hidden')) {
              window.recognition.start(); 
            }
          } catch(e) { console.warn('[Voice] STT Restart failed:', e); }
        }, 500);
      }
    };
  }
  
  return new Promise((resolve) => {
    try { 
      window.recognition.stop(); 
    } catch(e) {}

    setTimeout(() => {
      try {
        console.log("[Voice] Triggering STT start");
        window.recognition.start();
        resolve();
      } catch(e) {
        console.warn('[Voice] Initial start failed:', e);
        resolve();
      }
    }, 400);
  });
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

function queueAudioChunk(audioData, sampleRate) {
  audioQueue.push({ data: audioData, sampleRate: sampleRate });
  if (!isPlaying) {
    playNextAudioChunk();
  }
}

async function playNextAudioChunk() {
  if (audioQueue.length === 0) {
    isPlaying = false;
    return;
  }

  isPlaying = true;
  
  // Schedule all currently queued chunks
  while (audioQueue.length > 0) {
    const chunk = audioQueue.shift();
    try {
      const data = chunk.data;
      const sampleRate = chunk.sampleRate || 16000;
      
      const buffer = data.buffer;
      const int16Data = new Int16Array(buffer, data.byteOffset, data.byteLength / 2);
      const float32Data = new Float32Array(int16Data.length);

      for (let i = 0; i < int16Data.length; i++) {
        float32Data[i] = int16Data[i] / 32768.0;
      }

      const audioBuffer = audioCtx.createBuffer(1, float32Data.length, sampleRate);
      audioBuffer.getChannelData(0).set(float32Data);

      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);

      const now = audioCtx.currentTime;
      // Provide a small buffer ahead of current time if we fell behind
      if (nextPlayTime < now) nextPlayTime = now + 0.05;

      source.start(nextPlayTime);
      nextPlayTime += audioBuffer.duration;

      source.onended = () => {
        // If this was the last scheduled chunk and queue is empty, finish
        if (audioQueue.length === 0 && audioCtx.currentTime >= nextPlayTime - 0.1) {
          isPlaying = false;
        } else if (audioQueue.length > 0) {
          // If more chunks arrived while playing, schedule them
          playNextAudioChunk();
        }
      };
    } catch (err) {
      console.error('Playback Error:', err);
    }
  }
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
  const statusEl = document.getElementById('liveStatusText');
  const transcriptEl = document.getElementById('liveTranscript');
  if (text.startsWith('Hearing:') || text.startsWith('You:')) {
    if (transcriptEl) transcriptEl.innerText = text;
  } else {
    if (statusEl) statusEl.innerText = text;
  }
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
  } else if (searchMode === "system" || searchMode === "audit") {
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
  if (termSid || isInitializingTerm) return;
  isInitializingTerm = true;
  try {
    // Clear ALL terminal outputs before connection message to prevent duplicates
    const el = document.getElementById("terminalOutput");
    if (el) el.innerHTML = '';
    const popEl = document.getElementById("popupTerminalOutput");
    if (popEl) popEl.innerHTML = '';

    const res = await fetch("/api/terminal/create", { method: "POST", headers: { "Authorization": `Bearer ${authToken}` } });
    if (!res.ok) throw new Error(`Server responded ${res.status}`);
    const data = await res.json();
    if (!data.session_id) throw new Error('No session ID returned');
    termSid = data.session_id;

    termOut(`Connected to NeuralAI High-Velocity Console [SID: ${termSid}]`, 'info');
  } catch (err) { termOut(`Failed to establish Neural Uplink: ${err.message}`, 'error'); }
  finally { isInitializingTerm = false; }
}

function toggleTerminalPopup() {
  const panel = document.getElementById('terminalPanel');
  if (panel) {
    const isHidden = panel.classList.contains('hidden');
    panel.classList.toggle('hidden');
    if (isHidden) {
      initTerm();
      setTimeout(() => document.getElementById('popupTerminalInput')?.focus(), 100);
    }
  }
}

async function executeTerminalCmd(isPopup = false) {
  const inputId = isPopup ? "popupTerminalInput" : "terminalInput";
  const input = document.getElementById(inputId);
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
  } catch (err) { termOut(`Terminal Error: ${err.message}`, 'error'); }
}

function termOut(txt, cls = '') {
  const targets = ["terminalOutput", "popupTerminalOutput"];
  targets.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    const div = document.createElement('div');
    div.className = `term-line ${cls}`;
    div.innerHTML = txt;
    el.appendChild(div);
    el.scrollTop = el.scrollHeight;
  });
}

function clearTerm() {
  const el = document.getElementById("terminalOutput");
  if (el) el.innerHTML = '';
  const popEl = document.getElementById("popupTerminalOutput");
  if (popEl) popEl.innerHTML = '';
}

function restartTerm() {
  termSid = null;
  isInitializingTerm = false;
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
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    window.neuralFiles = data.files || [];
    renderFiles(window.neuralFiles);
  } catch (err) { 
    console.error("NeuralDrive access error:", err);
    container.innerHTML = `<div class="error-state">Error accessing NeuralDrive: ${err.message}</div>`; 
  }
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

// Upload Logic
function initUploadListeners() {
  const uploadBtn = document.getElementById("uploadBtn");
  const fileInput = document.getElementById("fileInputHidden");
  const dropZone = document.getElementById("dropZone");

  if (uploadBtn && fileInput) {
    uploadBtn.addEventListener("click", () => fileInput.click());
  }

  if (fileInput) {
    fileInput.addEventListener("change", async (e) => {
      await handleFileUploads(e.target.files);
      fileInput.value = ""; // Reset
    });
  }

  if (dropZone && fileInput) {
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.style.borderColor = "#a78bfa";
      dropZone.style.background = "rgba(167, 139, 250, 0.05)";
    });
    dropZone.addEventListener("dragleave", (e) => {
      e.preventDefault();
      dropZone.style.borderColor = "rgba(255, 255, 255, 0.1)";
      dropZone.style.background = "transparent";
    });
    dropZone.addEventListener("drop", async (e) => {
      e.preventDefault();
      dropZone.style.borderColor = "rgba(255, 255, 255, 0.1)";
      dropZone.style.background = "transparent";
      if (e.dataTransfer.files.length > 0) {
        await handleFileUploads(e.dataTransfer.files);
      }
    });
  }
}

async function handleFileUploads(files) {
  if (!files || files.length === 0) return;
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const formData = new FormData();
    formData.append("file", file);
    try {
      showToast(`Uploading ${file.name}...`, 'info');
      const res = await fetch("/api/upload", {
        method: "POST",
        headers: { "Authorization": `Bearer ${authToken}` },
        body: formData
      });
      if (res.ok) {
        showToast(`${file.name} uploaded successfully`, 'success');
      } else {
        const err = await res.json();
        showToast(`Failed: ${err.error || 'Upload error'}`, 'error');
      }
    } catch (e) {
      showToast(`Upload failed for ${file.name}`, 'error');
    }
  }
  loadFiles(); // Refresh
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
    showToast('Updating Identity Vault...', 'info');
    const res = await fetch('/api/user/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
      body: JSON.stringify(data)
    });
    if (res.ok) {
      showToast('Identity Vault Updated Successfully', 'success');
      await initAuth(); // Refresh UI and greeting
      loadUserProfile(); // Reload profile fields
    } else {
      const err = await res.json();
      throw new Error(err.error || 'Update failed');
    }
  } catch (e) { 
    showToast(`Vault Update Error: ${e.message}`, 'error'); 
  }
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
  updateArchitectureStatus();
}

async function updateArchitectureStatus() {
  try {
    const res = await fetch('/api/status', { headers: { 'Authorization': `Bearer ${authToken}` } });
    const data = await res.json();
    const settingDevice = document.getElementById('settingDevice');
    const versionNode = document.getElementById('settingVersion');
    if (settingDevice) settingDevice.textContent = data.device || 'CPU (Optimized)';
    if (versionNode) {
      versionNode.innerHTML = `<span style="color:#10b981; animation: pulse 2s infinite;">●</span> OTA Synced: ${data.version || 'v5.2.1'} (Rules: ${data.active_rules || 0})`;
    }
  } catch {}
}

// Add periodic updates for Architecture status
setInterval(updateArchitectureStatus, 30000);

async function saveVoicePreference() {
  const sel = document.getElementById('voiceSelection');
  const voice = sel?.value;
  if (!voice) return;
  
  try {
    showToast(`Switching persona to ${voice}...`, 'info');
    await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
      body: JSON.stringify({ neural_voice: voice })
    });
    
    // Update local state
    userSettings.neural_voice = voice;
    
    // If Live Session is active, send config update immediately
    if (voiceWS && voiceWS.readyState === WebSocket.OPEN) {
      console.log(`[Voice] Pushing live config update: ${voice}`);
      voiceWS.send(JSON.stringify({ type: 'config', voice: voice }));
      showToast(`${voice} Voice Active`, 'success');
    } else {
      showToast('Persona saved. Start Live Mode to use.', 'success');
    }
  } catch (err) {
    showToast('Failed to save voice preference.', 'error');
  }
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
  
  try {
    showToast('Storing memory...', 'info');
    const res = await fetch('/api/memory', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
      body: JSON.stringify({ fact })
    });
    if (!res.ok) throw new Error('Failed to save memory');
    
    input.value = '';
    await loadMemoryList();
    showToast('Memory Cloud Updated', 'success');
  } catch (err) {
    showToast('Error storing memory', 'error');
  }
}

async function loadMemoryList() {
  const list = document.getElementById('memoryList');
  if (!list) return;
  
  try {
    const res = await fetch('/api/memory', { headers: { 'Authorization': `Bearer ${authToken}` } });
    const data = await res.json();
    
    const items = data.facts || [];
    list.innerHTML = items.map(f => `
      <div class="memory-item">
        <span>${escHtml(f.fact)}</span>
        <button class="delete-btn" onclick="deleteMemoryItem(${f.id})" title="Purge Memory">×</button>
      </div>
    `).join('') || '<p style="font-size:12px;color:#888;padding:10px;">No dynamic memories stored.</p>';
  } catch (err) {
    list.innerHTML = '<p style="color:#ff6b6b;padding:10px;">Failed to sync with Memory Cloud.</p>';
  }
}

async function deleteMemoryItem(id) {
  if (!confirm('Purge this memory fact permanently?')) return;
  try {
    const res = await fetch(`/api/memory/${id}`, { 
      method: 'DELETE', 
      headers: { 'Authorization': `Bearer ${authToken}` } 
    });
    if (!res.ok) throw new Error('Delete failed');
    showToast('Memory Purged', 'success');
    loadMemoryList();
  } catch (err) {
    showToast('Failed to delete memory', 'error');
  }
}

async function addRuleFromTab() {
  const input = document.getElementById('ruleInput');
  const rule = input?.value.trim();
  if (!rule) return;
  
  try {
    showToast('Deploying rule...', 'info');
    const res = await fetch('/api/rules', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
      body: JSON.stringify({ rule })
    });
    if (!res.ok) throw new Error('Failed to save rule');
    
    input.value = '';
    await loadRulesList();
    showToast('Behavioral Rule Active', 'success');
  } catch (err) {
    showToast('Error saving rule', 'error');
  }
}

async function loadRulesList() {
  const list = document.getElementById('rulesList');
  if (!list) return;
  
  try {
    const res = await fetch('/api/rules', { headers: { 'Authorization': `Bearer ${authToken}` } });
    const data = await res.json();
    
    const items = data.rules || [];
    list.innerHTML = items.map(r => `
      <div class="rule-item">
        <span>${escHtml(r.rule)}</span>
        <div class="rule-actions">
          <button class="rule-toggle ${r.active ? 'on' : 'off'}" onclick="toggleRuleItem(${r.id})" title="Toggle Protocol">
            ${r.active ? 'ON' : 'OFF'}
          </button>
          <button class="delete-btn" onclick="deleteRuleItem(${r.id})" title="Decommission Rule">×</button>
        </div>
      </div>
    `).join('') || '<p style="font-size:12px;color:#888;padding:10px;">No behavioral rules defined.</p>';
  } catch (err) {
    list.innerHTML = '<p style="color:#ff6b6b;padding:10px;">Failed to sync Behavioral Rules.</p>';
  }
}

async function toggleRuleItem(id) {
  try {
    const res = await fetch(`/api/rules/${id}/toggle`, { 
      method: 'POST', 
      headers: { 'Authorization': `Bearer ${authToken}` } 
    });
    if (!res.ok) throw new Error('Toggle failed');
    loadRulesList();
  } catch (err) {
    showToast('Failed to toggle rule', 'error');
  }
}

async function deleteRuleItem(id) {
  if (!confirm('Decommission this behavioral protocol?')) return;
  try {
    const res = await fetch(`/api/rules/${id}`, { 
      method: 'DELETE', 
      headers: { 'Authorization': `Bearer ${authToken}` } 
    });
    if (!res.ok) throw new Error('Delete failed');
    showToast('Rule Decommissioned', 'success');
    loadRulesList();
  } catch (err) {
    showToast('Failed to delete rule', 'error');
  }
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
  window.localStorage.setItem('neural_theme', isDark ? 'dark' : 'light');
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
    window.localStorage.setItem("neural_onboarded", "true");
  }
}

// ========================================
// INITIALIZATION
// ========================================
document.addEventListener('DOMContentLoaded', () => {
  console.log("NeuralAI Core UI Initializing...");
  initAuth();
  initUploadListeners();
  
  // Wire Experimental Access Modes - Use addEventListener only
  document.getElementById('maestroSubmit')?.addEventListener('click', handleMaestroAuth);
  document.getElementById('guestSubmit')?.addEventListener('click', handleGuestAuth);
  
  // Wire top search bar (Google-style) — Ask button + Enter key
  document.getElementById('searchBtn')?.addEventListener('click', () => {
    console.log("Search Ask button clicked");
    handleSearch();
  });
  document.getElementById('queryInput')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSearch();
    }
  });

  // Wire bottom chat send button and enter key
  document.getElementById('sendBtn')?.addEventListener('click', () => {
    console.log("Send button clicked");
    sendMessage();
  });
  document.getElementById('chatInput')?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Wire Voice Controls
  document.getElementById('voiceBtn')?.addEventListener('click', () => {
    console.log("Voice button clicked");
    toggleLiveVoice(true);
  });
  document.getElementById('closeLiveBtn')?.addEventListener('click', () => {
    console.log("Close voice button clicked");
    toggleLiveVoice(false);
  });

  // Wire Terminal Panel
  document.getElementById('popupTerminalRun')?.addEventListener('click', () => executeTerminalCmd(true));
  document.getElementById('popupTerminalInput')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') executeTerminalCmd(true);
  });
  document.getElementById('closeTerminal')?.addEventListener('click', toggleTerminalPopup);

  // Wire search tabs properly to change mode AND execute if query exists
  document.querySelectorAll('.search-tabs .tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.search-tabs .tab').forEach(t => t.classList.remove('active'));
      btn.classList.add('active');
      searchMode = btn.dataset.mode;
      // Also shift visual focus
      if (searchMode === 'files') {
        document.getElementById("queryInput").placeholder = "Search NeuralDrive files...";
      } else if (searchMode === 'system') {
        document.getElementById("queryInput").placeholder = "Run system audit or terminal command...";
      } else {
        document.getElementById("queryInput").placeholder = "Ask NeuralAI anything...";
      }
      
      const text = document.getElementById("queryInput")?.value.trim();
      if (text) {
        handleSearch();
      }
    });
  });

  document.getElementById('authSubmit')?.addEventListener('click', handleAuth);
  document.getElementById('authToggle')?.addEventListener('click', toggleAuthMode);
  
  // Wire Experimental Access Modes
  document.getElementById('maestroSubmit')?.addEventListener('click', handleMaestroAuth);
  document.getElementById('guestSubmit')?.addEventListener('click', handleGuestAuth);
  
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

  // Stop button (appears during streaming)
  document.getElementById('stopBtn')?.addEventListener('click', () => {
    abortStream = true;
  });

  // Feature buttons: image + code
  document.getElementById('imageBtn')?.addEventListener('click', () => {
    const prompt = (document.getElementById('chatInput')?.value || '').trim() || 'a serene mountain landscape at sunset';
    generateImageFromPrompt(prompt);
  });
  document.getElementById('codeBtn')?.addEventListener('click', () => {
    const modal = document.getElementById('codeModal');
    const editor = document.getElementById('codeEditor');
    if (modal && editor) {
      const existing = (document.getElementById('chatInput')?.value || '').trim();
      if (existing && !editor.value) editor.value = existing;
      modal.classList.remove('hidden');
      editor.focus();
    }
  });
  document.getElementById('closeCodeModal')?.addEventListener('click', () => {
    document.getElementById('codeModal')?.classList.add('hidden');
  });
  document.getElementById('runCodeBtn')?.addEventListener('click', async () => {
    const editor = document.getElementById('codeEditor');
    const out = document.getElementById('codeOutput');
    if (!editor || !out) return;
    const code = editor.value;
    out.textContent = 'Running…';
    try {
      const res = await fetch('/api/execute/code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
        body: JSON.stringify({ code, language: 'python' })
      });
      const data = await res.json();
      out.textContent = data.success ? data.output : (data.error || 'No output');
      // Also drop the result into the chat
      addMsg('assistant', '```python\n' + (data.success ? data.output : (data.error || 'No output')) + '\n```');
    } catch (e) {
      out.textContent = 'Error: ' + e.message;
    }
  });

  // Uplink toggle
  const uplinkToggle = document.getElementById('uplinkToggle');
  if (uplinkToggle) {
    uplinkToggle.checked = uplinkEnabled;
    uplinkToggle.addEventListener('change', e => {
      uplinkEnabled = e.target.checked;
      window.localStorage.setItem('neural_uplink', uplinkEnabled ? '1' : '0');
    });
  }
  const savedUplink = window.localStorage.getItem('neural_uplink');
  if (savedUplink !== null) uplinkEnabled = savedUplink === '1';

  // Model status polling
  updateModelStatus();
  setInterval(updateModelStatus, 30000);
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
window.sendMessage = sendMessage;
window.toggleLiveVoice = toggleLiveVoice;
window.closeOnboarding = closeOnboarding;
window.toggleTerminalPopup = toggleTerminalPopup;
window.executeTerminalCmd = executeTerminalCmd;
