const API_BASE_URL =
  import.meta.env?.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const TOKEN_KEY = "interviewStudio.accessToken";
let accessToken;

function getStorage() {
  return typeof window === "undefined" ? null : window.sessionStorage;
}

function getAccessToken() {
  if (accessToken !== undefined) return accessToken;
  accessToken = getStorage()?.getItem(TOKEN_KEY) || null;
  return accessToken;
}

function setAccessToken(token) {
  accessToken = token;
  const storage = getStorage();
  if (token) storage?.setItem(TOKEN_KEY, token);
  else storage?.removeItem(TOKEN_KEY);
}

function clearAccessToken() {
  setAccessToken(null);
}

async function request(path, options = {}) {
  const headers = { ...options.headers };
  const token = getAccessToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });
  if (response.status === 401 && token) {
    clearAccessToken();
    if (typeof window !== "undefined") {
      window.dispatchEvent(new Event("interviewstudio:auth-expired"));
    }
  }
  if (response.status === 204) return null;

  const payload = await response.json();
  if (!response.ok) {
    const message =
      typeof payload.detail === "string"
        ? payload.detail
        : "The request could not be completed.";
    throw new Error(message);
  }
  return payload;
}

function postJson(path, body) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const api = {
  getAccessToken,
  setAccessToken,
  clearAccessToken,

  register({ full_name, email, password }) {
    return postJson("/auth/register", { full_name, email, password });
  },

  async login({ email, password }) {
    clearAccessToken();
    return postJson("/auth/login", { email, password });
  },

  getCurrentUser() {
    return request("/auth/me");
  },

  async logout() {
    try {
      return await request("/auth/logout", { method: "POST" });
    } finally {
      clearAccessToken();
    }
  },

  uploadResume(file) {
    const formData = new FormData();
    formData.append("file", file);
    return request("/documents/upload", {
      method: "POST",
      body: formData,
    });
  },

  startInterview({ role, interview_type, difficulty, number_of_questions, document_id }) {
    return postJson("/interview/start", {
      role,
      interview_type,
      difficulty,
      number_of_questions,
      document_id,
    });
  },

  getQuestion(session_id) {
    return postJson("/interview/question", { session_id });
  },

  evaluateAnswer({ session_id, question, answer }) {
    return postJson("/interview/evaluate", { session_id, question, answer });
  },

  getSession(session_id) {
    return request(`/interview/${encodeURIComponent(session_id)}`);
  },

  getReport(session_id) {
    return request(`/interview/${encodeURIComponent(session_id)}/report`);
  },

  listSessions(limit = 50) {
    return request(`/interviews?limit=${encodeURIComponent(limit)}`);
  },
};
