import assert from "node:assert/strict";
import { test } from "node:test";

import {
  clearAccountState,
  clearInterviewState,
  createAccountRequestGuard,
  initialForm,
} from "./appState.js";

function deferred() {
  let resolve;
  const promise = new Promise((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

function createStateSetters() {
  const state = {
    session: { session_id: "old-session" },
    question: { question: "Old question" },
    answer: "Private answer from the previous account.",
    evaluation: { score: 9 },
    report: { overall_score: 9 },
    sessions: [{ session_id: "old-session" }],
    documentId: "old-document",
    documents: [{ document_id: "old-document" }],
    form: { role: "Previous user's role" },
    documentType: "study_material",
  };
  const setters = Object.fromEntries(
    Object.keys(state).map((key) => [
      `set${key[0].toUpperCase()}${key.slice(1)}`,
      (value) => {
        state[key] = value;
      },
    ]),
  );
  return { state, setters };
}

test("logout clears private interview, document, and account form state", () => {
  const { state, setters } = createStateSetters();

  clearAccountState(setters);

  assert.equal(state.session, null);
  assert.equal(state.question, null);
  assert.equal(state.answer, "");
  assert.equal(state.evaluation, null);
  assert.equal(state.report, null);
  assert.deepEqual(state.sessions, []);
  assert.equal(state.documentId, "");
  assert.deepEqual(state.documents, []);
  assert.equal(state.form, initialForm);
  assert.equal(state.documentType, "resume");
});

test("expired-token cleanup clears private state before another account signs in", () => {
  const { state, setters } = createStateSetters();

  clearAccountState(setters);
  state.user = { user_id: "new-account" };

  assert.equal(state.user.user_id, "new-account");
  assert.equal(state.answer, "");
  assert.equal(state.session, null);
  assert.equal(state.documentId, "");
  assert.deepEqual(state.documents, []);
});

test("starting or resuming an interview clears prior interview answers", () => {
  const { state, setters } = createStateSetters();

  clearInterviewState(setters);

  assert.equal(state.session, null);
  assert.equal(state.question, null);
  assert.equal(state.answer, "");
  assert.equal(state.evaluation, null);
  assert.equal(state.report, null);
});

test("late session and document responses cannot restore a previous account", async () => {
  const guard = createAccountRequestGuard();
  guard.activate("account-a");
  const accountARequest = guard.capture();
  const sessionsResponse = deferred();
  const documentsResponse = deferred();
  const state = {
    sessions: [],
    documents: [],
  };

  const sessionsRequest = guard.run(
    accountARequest,
    () => sessionsResponse.promise,
    (result) => {
      state.sessions = result;
    },
  );
  const documentsRequest = guard.run(
    accountARequest,
    () => documentsResponse.promise,
    (result) => {
      state.documents = result;
    },
  );

  guard.invalidate();
  guard.activate("account-b");
  state.sessions = [{ session_id: "account-b-session" }];
  state.documents = [{ document_id: "account-b-document" }];

  sessionsResponse.resolve([{ session_id: "account-a-session" }]);
  documentsResponse.resolve([{ document_id: "account-a-document" }]);

  assert.equal(await sessionsRequest, false);
  assert.equal(await documentsRequest, false);
  assert.deepEqual(state.sessions, [{ session_id: "account-b-session" }]);
  assert.deepEqual(state.documents, [{ document_id: "account-b-document" }]);
});
