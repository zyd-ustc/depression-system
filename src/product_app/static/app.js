const state = {
  token: localStorage.getItem("product_token") || "",
  username: "",
};

const $ = (id) => document.getElementById(id);

function show(view) {
  ["authView", "consentView", "chatView"].forEach((id) => $(id).classList.add("hidden"));
  $(view).classList.remove("hidden");
  $("logoutBtn").classList.toggle("hidden", view === "authView");
}

function setStatus(id, text) {
  $(id).textContent = text || "";
}

function formatApiError(payload, fallback) {
  const detail = payload.detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || JSON.stringify(item)).join("；");
  }
  if (typeof detail === "string") {
    return detail;
  }
  if (detail && typeof detail === "object") {
    return JSON.stringify(detail);
  }
  return fallback;
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(path, { ...options, headers });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload, response.statusText));
  }
  return payload;
}

async function authenticate(mode) {
  setStatus("authStatus", "");
  const username = $("username").value.trim();
  const password = $("password").value;
  if (username.length < 2) {
    setStatus("authStatus", "用户名至少 2 个字符。");
    return;
  }
  if (password.length < 6) {
    setStatus("authStatus", "密码至少 6 位。");
    return;
  }
  try {
    const payload = await api(`/api/auth/${mode}`, {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    state.token = payload.token;
    state.username = payload.username;
    localStorage.setItem("product_token", state.token);
    $("currentUser").textContent = state.username;
    if (payload.consent_required) show("consentView");
    else show("chatView");
  } catch (error) {
    setStatus("authStatus", `失败：${error.message}`);
  }
}

async function loadMe() {
  if (!state.token) {
    show("authView");
    return;
  }
  try {
    const payload = await api("/api/me");
    state.username = payload.username;
    $("currentUser").textContent = state.username;
    if (payload.consent_required) show("consentView");
    else show("chatView");
  } catch {
    localStorage.removeItem("product_token");
    state.token = "";
    show("authView");
  }
}

async function acceptConsent() {
  if (!$("consentCheck").checked) {
    setStatus("consentStatus", "需要勾选同意。");
    return;
  }
  try {
    const payload = await api("/api/consent", {
      method: "POST",
      body: JSON.stringify({ accepted: true }),
    });
    state.token = payload.token;
    localStorage.setItem("product_token", state.token);
    show("chatView");
  } catch (error) {
    setStatus("consentStatus", `失败：${error.message}`);
  }
}

function addMessage(role, content) {
  const node = document.createElement("div");
  node.className = "msg";
  node.innerHTML = `<span class="role">${role}</span>${escapeHtml(content)}`;
  $("messages").appendChild(node);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);
}

async function sendMessage(event) {
  event.preventDefault();
  const text = $("messageInput").value.trim();
  if (!text) return;
  $("messageInput").value = "";
  addMessage("你", text);
  const button = $("chatForm").querySelector("button");
  button.disabled = true;
  try {
    const payload = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message: text }),
    });
    addMessage("系统", payload.assistant_reply);
    $("riskLevel").textContent = payload.risk.level;
  } catch (error) {
    addMessage("系统", `请求失败：${error.message}`);
  } finally {
    button.disabled = false;
  }
}

$("loginBtn").addEventListener("click", () => authenticate("login"));
$("registerBtn").addEventListener("click", () => authenticate("register"));
$("consentBtn").addEventListener("click", acceptConsent);
$("chatForm").addEventListener("submit", sendMessage);
$("logoutBtn").addEventListener("click", () => {
  localStorage.removeItem("product_token");
  state.token = "";
  show("authView");
});

loadMe();
