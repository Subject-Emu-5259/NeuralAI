// NeuralAI v2 — chat demo
// Calls the public HF Inference API. Your token is stored locally for rate limits.

const HF_MODEL = "Subject-Emu-5259/NeuralAI-merged";
// Try router endpoint first (more reliable from within HF Spaces), fallback to classic API
const API_URL  = `https://router.huggingface.co/hf-inference/models/${HF_MODEL}`;
const API_URL_FALLBACK = `https://api-inference.huggingface.co/models/${HF_MODEL}`;

const SYSTEM_PROMPT =
  "You are NeuralAI v2, a helpful, concise assistant. " +
  "Answer in plain text. Use short paragraphs. No markdown unless asked.";

const log       = document.getElementById("log");
const form      = document.getElementById("form");
const input     = document.getElementById("input");
const sendBtn   = document.getElementById("sendBtn");
const clearBtn  = document.getElementById("clearBtn");
const modelLbl  = document.getElementById("modelLabel");
const tokenInput = document.getElementById("tokenInput");
const tokenStatus = document.getElementById("tokenStatus");

let history = []; // [{role, content}]

// ── Token management ──────────────────────────────────────────────
function loadToken() {
  return localStorage.getItem("HF_TOKEN") || "";
}
function saveToken(t) {
  if (t) localStorage.setItem("HF_TOKEN", t);
  else localStorage.removeItem("HF_TOKEN");
  updateTokenUI();
}
function updateTokenUI() {
  const t = loadToken();
  if (tokenInput) tokenInput.value = t;
  if (tokenStatus) {
    tokenStatus.textContent = t ? "✅ Token set" : "⚡ No token (rate-limited)";
    tokenStatus.className = t ? "token-ok" : "token-missing";
  }
  if (modelLbl) {
    modelLbl.textContent = t
      ? "base · SmolLM2-360M · authenticated"
      : "base · SmolLM2-360M-Instruct";
  }
}

// Token input events
if (tokenInput) {
  tokenInput.addEventListener("input", () => saveToken(tokenInput.value.trim()));
  tokenInput.value = loadToken();
}

// ── Chat helpers ──────────────────────────────────────────────────
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}

function addMessage(role, content, opts = {}) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}${opts.streaming ? " streaming" : ""}`;
  wrap.innerHTML = `<div class="role">${role === "user" ? "you" : "neuralai"}</div><div class="body"></div>`;
  wrap.querySelector(".body").textContent = content;
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
  return wrap;
}

function setBusy(busy) {
  sendBtn.disabled = busy;
  sendBtn.textContent = busy ? "…" : "Send";
  input.disabled = busy;
  if (busy) sendBtn.classList.add("busy");
  else sendBtn.classList.remove("busy");
}

// ── Clear ─────────────────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  history = [];
  log.innerHTML = "";
  addMessage(
    "bot",
    "Hey, I'm NeuralAI v2. Ask me anything — this is the merged LoRA adapter (NeuralAI-merged) running on SmolLM2-360M."
  );
});

// Greet
addMessage(
  "bot",
  "Hey, I'm NeuralAI v2. Ask me anything — this is the merged LoRA adapter (NeuralAI-merged) running on SmolLM2-360M."
);
updateTokenUI();

// Auto-grow textarea
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(160, input.scrollHeight) + "px";
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

// ── Submit ────────────────────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  addMessage("user", text);
  history.push({ role: "user", content: text });
  input.value = "";
  input.style.height = "auto";
  setBusy(true);

  const botMsg = addMessage("bot", "", { streaming: true });
  const bodyEl = botMsg.querySelector(".body");

  const prompt =
    `${SYSTEM_PROMPT}\n\n` +
    history.map(m => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`).join("\n") +
    `\nAssistant:`;

  try {
    const headers = { "Content-Type": "application/json" };
    const token = loadToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;

    const body = JSON.stringify({
      inputs: prompt,
      parameters: {
        max_new_tokens: 512,
        temperature: 0.7,
        top_p: 0.9,
        repetition_penalty: 1.1,
        return_full_text: false,
        stop: ["\nUser:", "\nAssistant:"],
      },
      options: { wait_for_model: true, use_cache: false },
    });
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);

    // Try primary endpoint, fallback to secondary on network or HTTP errors
    let resp;
    let usedFallback = false;
    try {
      resp = await fetch(API_URL, {
        method: "POST", headers, body, signal: controller.signal,
      });
      // If router returned an HTTP error, try the classic endpoint instead
      if (!resp.ok) {
        console.warn(`Primary endpoint returned ${resp.status}, trying fallback`);
        resp = await fetch(API_URL_FALLBACK, {
          method: "POST", headers, body, signal: controller.signal,
        });
        usedFallback = true;
      }
    } catch (fetchErr) {
      // DNS/network error on primary — try fallback
      console.warn("Primary endpoint failed, trying fallback:", fetchErr.message);
      resp = await fetch(API_URL_FALLBACK, {
        method: "POST", headers, body, signal: controller.signal,
      });
      usedFallback = true;
    }
    clearTimeout(timeout);

    if (!resp.ok) {
      const errText = await resp.text();
      let msg =
        `⚠️ Inference API error (${resp.status}). `;
      if (resp.status === 401 || resp.status === 403) {
        msg += "Your token may be invalid. Try clearing it (delete the text in the token field) or generate a new one at huggingface.co/settings/tokens.";
      } else if (resp.status === 503) {
        msg += "The model is loading. Please wait ~30s and try again.";
      } else {
        msg += `Details: ${errText.slice(0, 300)}`;
      }
      bodyEl.textContent = msg;
      botMsg.classList.remove("streaming");
      setBusy(false);
      return;
    }

    const data = await resp.json();
    let reply = "";
    if (Array.isArray(data) && data[0]?.generated_text) {
      reply = data[0].generated_text;
    } else if (data?.generated_text) {
      reply = data.generated_text;
    } else if (data?.error) {
      reply = `⚠️ ${data.error}`;
    } else {
      reply = "(no response — the model returned an empty result. Try rephrasing your question.)";
    }
    reply = reply.trim();

    // Animate the response
    bodyEl.textContent = "";
    let i = 0;
    const speed = Math.max(5, Math.min(15, Math.floor(reply.length / 60)));
    const tick = () => {
      i = Math.min(reply.length, i + speed);
      bodyEl.textContent = reply.slice(0, i);
      log.scrollTop = log.scrollHeight;
      if (i < reply.length) requestAnimationFrame(tick);
      else {
        botMsg.classList.remove("streaming");
        history.push({ role: "assistant", content: reply });
      }
    };
    tick();
  } catch (err) {
    if (err.name === "AbortError") {
      bodyEl.textContent = "⚠️ Request timed out after 60s. The model may be overloaded — try again later or add your HF token above.";
    } else if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
      bodyEl.textContent =
        "⚠️ Network error: cannot reach the Inference API (tried both router.huggingface.co and api-inference.huggingface.co). " +
        "This is often a DNS / connectivity issue on Hugging Face's free tier. " +
        "Try adding your HF token above (some endpoints require auth), or try again later. " +
        "You can also try restarting the Space from the HF dashboard.";
    } else {
      bodyEl.textContent = `⚠️ Error: ${err.message}`;
    }
    botMsg.classList.remove("streaming");
  } finally {
    setBusy(false);
  }
});
