import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import { api } from "./api.js";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
  api.clearAccessToken();
});

test("startInterview sends the complete session configuration", async () => {
  let requestUrl;
  let requestOptions;
  globalThis.fetch = async (url, options) => {
    requestUrl = url;
    requestOptions = options;
    return Response.json({ session_id: "session-1" });
  };

  const result = await api.startInterview({
    role: "Python Developer",
    interview_type: "technical",
    difficulty: "medium",
    number_of_questions: 3,
    document_id: "resume-1",
  });

  assert.equal(requestUrl, "http://127.0.0.1:8000/interview/start");
  assert.equal(requestOptions.method, "POST");
  assert.deepEqual(JSON.parse(requestOptions.body), {
    role: "Python Developer",
    interview_type: "technical",
    difficulty: "medium",
    number_of_questions: 3,
    document_id: "resume-1",
  });
  assert.equal(result.session_id, "session-1");
});

test("getQuestion and evaluateAnswer preserve the session and active question", async () => {
  const captured = [];
  globalThis.fetch = async (url, options) => {
    captured.push({ url, options });
    return Response.json({ status: "active" });
  };

  await api.getQuestion("session-2");
  await api.evaluateAnswer({
    session_id: "session-2",
    question: "What did you build?",
    answer: "A tested API.",
  });

  assert.equal(captured[0].options.method, "POST");
  assert.deepEqual(JSON.parse(captured[0].options.body), {
    session_id: "session-2",
  });
  assert.equal(captured[1].options.method, "POST");
  assert.deepEqual(JSON.parse(captured[1].options.body), {
    session_id: "session-2",
    question: "What did you build?",
    answer: "A tested API.",
  });
});

test("API errors surface the FastAPI detail message", async () => {
  globalThis.fetch = async () =>
    Response.json({ detail: "Interview session not found" }, { status: 404 });

  await assert.rejects(api.getSession("missing"), {
    message: "Interview session not found",
  });
});

test("uploadResume sends the PDF as multipart form data", async () => {
  let requestOptions;
  globalThis.fetch = async (_url, options) => {
    requestOptions = options;
    return Response.json({ document_id: "resume-2" });
  };
  const file = new File(["resume"], "resume.pdf", { type: "application/pdf" });

  const result = await api.uploadResume(file);

  assert.equal(requestOptions.method, "POST");
  assert.equal(requestOptions.body.get("file").name, "resume.pdf");
  assert.equal(result.document_id, "resume-2");
});

test("authenticated API requests attach the bearer token", async () => {
  let requestOptions;
  api.setAccessToken("test-access-token");
  globalThis.fetch = async (_url, options) => {
    requestOptions = options;
    return Response.json({ sessions: [] });
  };

  await api.listSessions();

  assert.equal(
    requestOptions.headers.Authorization,
    "Bearer test-access-token",
  );
});

test("an expired access token is cleared after an unauthorized response", async () => {
  api.setAccessToken("expired-token");
  globalThis.fetch = async () =>
    Response.json({ detail: "Invalid or expired access token." }, { status: 401 });

  await assert.rejects(api.listSessions(), {
    message: "Invalid or expired access token.",
  });
  assert.equal(api.getAccessToken(), null);
});

test("login and registration use the expected authentication endpoints", async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return Response.json({ user: { user_id: "user-1" } });
  };
  await api.register({
    full_name: "Test User",
    email: "test@example.com",
    password: "a-very-long-password",
  });
  await api.login({
    email: "test@example.com",
    password: "a-very-long-password",
  });

  assert.deepEqual(calls.map(({ url }) => url), [
    "http://127.0.0.1:8000/auth/register",
    "http://127.0.0.1:8000/auth/login",
  ]);
  assert.deepEqual(
    JSON.parse(calls[0].options.body),
    {
      full_name: "Test User",
      email: "test@example.com",
      password: "a-very-long-password",
    },
  );
});
