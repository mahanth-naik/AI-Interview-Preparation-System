import React, { useCallback, useEffect, useState } from "react";

import { api } from "./api.js";

const initialForm = {
  role: "",
  interview_type: "technical",
  difficulty: "medium",
  number_of_questions: 5,
};

function App() {
  const [page, setPage] = useState("dashboard");
  const [user, setUser] = useState(null);
  const [authReady, setAuthReady] = useState(false);
  const [authMode, setAuthMode] = useState("login");
  const [form, setForm] = useState(initialForm);
  const [documentId, setDocumentId] = useState("");
  const [documentType, setDocumentType] = useState("resume");
  const [documents, setDocuments] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [session, setSession] = useState(null);
  const [question, setQuestion] = useState(null);
  const [answer, setAnswer] = useState("");
  const [evaluation, setEvaluation] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const refreshSessions = useCallback(async () => {
    const result = await api.listSessions();
    setSessions(result.sessions);
  }, []);

  const refreshDocuments = useCallback(async () => {
    const result = await api.listDocuments();
    setDocuments(result.documents);
  }, []);

  useEffect(() => {
    let active = true;
    if (!api.getAccessToken()) {
      setAuthReady(true);
      return () => {
        active = false;
      };
    }

    api.getCurrentUser()
      .then((result) => {
        if (active) setUser(result.user);
      })
      .catch((requestError) => {
        api.clearAccessToken();
        if (active) setError(requestError.message);
      })
      .finally(() => {
        if (active) setAuthReady(true);
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (user) {
      refreshSessions().catch((requestError) => setError(requestError.message));
      refreshDocuments().catch((requestError) => setError(requestError.message));
    } else {
      setSessions([]);
      setDocuments([]);
    }
  }, [refreshDocuments, refreshSessions, user]);

  useEffect(() => {
    function handleExpiredToken() {
      setUser(null);
      setSessions([]);
      setSession(null);
      setQuestion(null);
      setEvaluation(null);
      setReport(null);
      setPage("dashboard");
      setAuthMode("login");
      setError("Your session expired. Please sign in again.");
    }

    window.addEventListener("interviewstudio:auth-expired", handleExpiredToken);
    return () => {
      window.removeEventListener(
        "interviewstudio:auth-expired",
        handleExpiredToken,
      );
    };
  }, []);

  async function authenticate(credentials) {
    setLoading(true);
    setError("");
    try {
      if (authMode === "register") {
        await api.register(credentials);
      }
      const result = await api.login(credentials);
      api.setAccessToken(result.access_token);
      setUser(result.user);
      setPage("dashboard");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function signOut() {
    setLoading(true);
    setError("");
    try {
      await api.logout();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setUser(null);
      setSessions([]);
      setSession(null);
      setQuestion(null);
      setEvaluation(null);
      setReport(null);
      setDocumentId("");
      setDocuments([]);
      setPage("dashboard");
      setAuthMode("login");
      setLoading(false);
    }
  }

  function updateForm(event) {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
  }

  async function uploadDocument(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const result = await api.uploadDocument(file, documentType);
      setDocumentId(result.document_id);
      await refreshDocuments();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
      event.target.value = "";
    }
  }

  async function deleteDocument(documentIdToDelete) {
    if (!documentIdToDelete) return;
    setLoading(true);
    setError("");
    try {
      await api.deleteDocument(documentIdToDelete);
      if (documentId === documentIdToDelete) setDocumentId("");
      await refreshDocuments();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function startInterview(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setEvaluation(null);
    setReport(null);
    try {
      const created = await api.startInterview({
        ...form,
        number_of_questions: Number(form.number_of_questions),
        document_id: documentId || null,
      });
      setSession(created);
      setPage("room");
      const firstQuestion = await api.getQuestion(created.session_id);
      setQuestion(firstQuestion);
      await refreshSessions();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function submitAnswer(event) {
    event.preventDefault();
    if (!session || !question?.question || !answer.trim()) return;
    setLoading(true);
    setError("");
    try {
      const result = await api.evaluateAnswer({
        session_id: session.session_id,
        question: question.question,
        answer: answer.trim(),
      });
      setEvaluation(result);
      setAnswer("");
      await refreshSessions();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadNextQuestion() {
    if (!session) return;
    setLoading(true);
    setError("");
    try {
      const next = await api.getQuestion(session.session_id);
      setQuestion(next);
      setEvaluation(null);
      setAnswer("");
      if (next.status === "completed" || !next.question) {
        await openReport(session.session_id);
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function openReport(sessionId) {
    setLoading(true);
    setError("");
    try {
      const result = await api.getReport(sessionId);
      setReport(result);
      setPage("report");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function resumeInterview(sessionId) {
    setLoading(true);
    setError("");
    setEvaluation(null);
    setReport(null);
    try {
      const restored = await api.getSession(sessionId);
      setSession(restored);
      if (restored.status === "completed") {
        await openReport(sessionId);
      } else {
        setPage("room");
        setQuestion(await api.getQuestion(sessionId));
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  function navigate(nextPage) {
    setError("");
    setPage(nextPage);
  }

  if (!authReady) {
    return (
      <div className="auth-loading">
        <span className="spinner" />
        <span>Loading your workspace…</span>
      </div>
    );
  }

  if (!user) {
    return (
      <AuthPage
        mode={authMode}
        onModeChange={(nextMode) => {
          setAuthMode(nextMode);
          setError("");
        }}
        onSubmit={authenticate}
        loading={loading}
        error={error}
      />
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => navigate("dashboard")} aria-label="Interview Studio home">
          <span className="brand-mark">IS</span>
          <span>Interview Studio</span>
        </button>
        <nav aria-label="Main navigation">
          <button className={page === "dashboard" ? "nav-link active" : "nav-link"} onClick={() => navigate("dashboard")}>Dashboard</button>
          <button className={page === "history" ? "nav-link active" : "nav-link"} onClick={() => navigate("history")}>My interviews</button>
        </nav>
        <div className="account-actions">
          <span className="account-name">{user.full_name}</span>
          <button className="text-button" onClick={signOut} disabled={loading}>Sign out</button>
        </div>
      </header>

      <main className="page-wrap">
        {error && (
          <div className="alert" role="alert">
            <span>{error}</span>
            <button onClick={() => setError("")} aria-label="Dismiss error">×</button>
          </div>
        )}
        {page === "dashboard" && (
          <Dashboard
            form={form}
            updateForm={updateForm}
            uploadDocument={uploadDocument}
            documentType={documentType}
            setDocumentType={setDocumentType}
            documents={documents}
            documentId={documentId}
            setDocumentId={setDocumentId}
            loading={loading}
            sessions={sessions}
            user={user}
            onStart={startInterview}
            onResume={resumeInterview}
            onHistory={() => navigate("history")}
            onDeleteDocument={deleteDocument}
          />
        )}
        {page === "history" && (
          <History
            sessions={sessions}
            loading={loading}
            onResume={resumeInterview}
            onBack={() => navigate("dashboard")}
          />
        )}
        {page === "room" && (
          <InterviewRoom
            session={session}
            question={question}
            answer={answer}
            evaluation={evaluation}
            loading={loading}
            setAnswer={setAnswer}
            onSubmit={submitAnswer}
            onNext={loadNextQuestion}
            onReport={() => openReport(session.session_id)}
            onExit={() => navigate("dashboard")}
          />
        )}
        {page === "report" && (
          <Report
            report={report}
            loading={loading}
            onHistory={() => navigate("history")}
            onDashboard={() => navigate("dashboard")}
          />
        )}
      </main>
      <footer className="footer">Built to make every practice session count.</footer>
    </div>
  );
}

function AuthPage({ mode, onModeChange, onSubmit, loading, error }) {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [passwordVisible, setPasswordVisible] = useState(false);
  const registering = mode === "register";

  function submit(event) {
    event.preventDefault();
    onSubmit({ full_name: fullName, email, password });
  }

  return (
    <main className="auth-shell">
      <section className="auth-card">
        <div className="auth-brand">
          <span className="brand-mark">IS</span>
          <strong>Interview Studio</strong>
        </div>
        <p className="eyebrow">{registering ? "CREATE YOUR ACCOUNT" : "WELCOME BACK"}</p>
        <h1>{registering ? "Start preparing with purpose." : "Sign in to your workspace."}</h1>
        <p className="auth-description">
          {registering
            ? "Build confidence with personalized interview practice and useful feedback."
            : "Continue your interview practice and pick up where you left off."}
        </p>
        {error && <div className="alert" role="alert">{error}</div>}
        <form className="auth-form" onSubmit={submit}>
          {registering && (
            <label className="field">
              Full name
              <input
                autoComplete="name"
                value={fullName}
                onChange={(event) => setFullName(event.target.value)}
                maxLength={100}
                required
              />
            </label>
          )}
          <label className="field">
            Email address
            <input
              type="email"
              autoComplete="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              maxLength={254}
              required
            />
          </label>
          <label className="field">
            Password
            <span className="password-input">
              <input
                type={passwordVisible ? "text" : "password"}
                autoComplete={registering ? "new-password" : "current-password"}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                minLength={registering ? 12 : 1}
                maxLength={128}
                required
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setPasswordVisible((visible) => !visible)}
                aria-label={passwordVisible ? "Hide password" : "Show password"}
              >
                {passwordVisible ? "Hide" : "Show"}
              </button>
            </span>
            {registering && <small className="field-hint">Use at least 12 characters.</small>}
          </label>
          <button className="button button-primary button-full" disabled={loading}>
            {loading ? <><span className="spinner" /> {registering ? "Creating account…" : "Signing in…"}</> : registering ? "Create account" : "Sign in"}
          </button>
        </form>
        <p className="auth-switch">
          {registering ? "Already have an account?" : "New to Interview Studio?"}
          <button
            type="button"
            className="text-button"
            onClick={() => onModeChange(registering ? "login" : "register")}
          >
            {registering ? "Sign in" : "Create an account"}
          </button>
        </p>
      </section>
    </main>
  );
}

function Dashboard({
  form, updateForm, uploadDocument, documentType, setDocumentType,
  documents = [], documentId, setDocumentId, loading, sessions = [], user,
  onStart, onResume, onHistory, onDeleteDocument,
}) {
  const completedSessions = sessions.filter((item) => item.status === "completed");
  const scoredSessions = completedSessions.filter(
    (item) => typeof item.overall_score === "number",
  );
  const averageScore = scoredSessions.length
    ? (
        scoredSessions.reduce((total, item) => total + item.overall_score, 0) /
        scoredSessions.length
      ).toFixed(1)
    : "—";
  const firstName = user?.full_name?.trim().split(/\s+/)[0] || "there";
  const recentSessions = sessions.slice(0, 3);

  return (
    <div className="dashboard">
      <section className="dashboard-hero">
        <div className="hero-copy">
          <span className="hero-kicker"><span className="hero-kicker-dot" /> YOUR INTERVIEW WORKSPACE</span>
          <h1>Prepare with confidence, {firstName}.</h1>
          <p>Practice with purpose. Get useful feedback, track your progress, and walk into your next interview better prepared.</p>
          <button className="hero-button" type="button" onClick={() => document.getElementById("practice-setup")?.scrollIntoView({ behavior: "smooth", block: "start" })}>
            Start practising <span aria-hidden="true">→</span>
          </button>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="hero-art-orbit orbit-one" />
          <div className="hero-art-orbit orbit-two" />
          <div className="hero-art-card">
            <span className="hero-art-spark">✦</span>
            <span className="hero-art-line line-long" />
            <span className="hero-art-line line-short" />
            <span className="hero-art-check">✓</span>
          </div>
          <span className="hero-art-caption">PRACTICE · IMPROVE · REPEAT</span>
        </div>
      </section>

      <section className="progress-section" aria-labelledby="progress-heading">
        <div className="section-heading">
          <div>
            <span className="section-kicker">YOUR ACTIVITY</span>
            <h2 id="progress-heading">Progress at a glance</h2>
          </div>
          <button className="dashboard-text-link" type="button" onClick={onHistory}>
            View interview history <span aria-hidden="true">↗</span>
          </button>
        </div>

        <div className="dashboard-stats">
          <article className="dashboard-stat">
            <span className="stat-symbol stat-symbol-blue" aria-hidden="true">▤</span>
            <div><span className="stat-label">Total interviews</span><strong>{sessions.length}</strong><span className="stat-note">Practice sessions started</span></div>
          </article>
          <article className="dashboard-stat">
            <span className="stat-symbol stat-symbol-green" aria-hidden="true">✓</span>
            <div><span className="stat-label">Completed</span><strong>{completedSessions.length}</strong><span className="stat-note">Finished interview sessions</span></div>
          </article>
          <article className="dashboard-stat">
            <span className="stat-symbol stat-symbol-gold" aria-hidden="true">↗</span>
            <div><span className="stat-label">Average score</span><strong>{averageScore}{averageScore !== "—" && <small> / 10</small>}</strong><span className="stat-note">Across scored interviews</span></div>
          </article>
        </div>
      </section>

      <section className="practice-section" id="practice-setup" aria-labelledby="practice-heading">
        <div className="section-heading practice-heading">
          <div>
            <span className="section-kicker">READY WHEN YOU ARE</span>
            <h2 id="practice-heading">Set up your next interview</h2>
            <p className="section-description">A few details help us shape a focused practice session.</p>
          </div>
          <span className="saved-indicator"><span /> Your sessions are saved automatically</span>
        </div>

        <div className="dashboard-columns">
          <form className="dashboard-panel setup-panel" onSubmit={onStart}>
            <div className="panel-heading">
              <span className="panel-step">01</span>
              <div><h3>Interview setup</h3><p>Choose your role and preferred interview format.</p></div>
            </div>

            <label className="dashboard-field field-full">
              <span>Document type</span>
              <select
                value={documentType}
                onChange={(event) => setDocumentType(event.target.value)}
              >
                <option value="resume">Resume</option>
                <option value="study_material">Study material</option>
              </select>
            </label>
            <label className="resume-dropzone">
              <span className="resume-icon" aria-hidden="true">↑</span>
              <span className="resume-copy">
                <strong>
                  {documentType === "resume" ? "Add a resume" : "Add study material"}
                </strong>
                <small>Upload a PDF to use it as interview context</small>
              </span>
              <span className="resume-action">Choose PDF</span>
              <input
                type="file"
                accept="application/pdf,.pdf"
                onChange={uploadDocument}
                disabled={loading}
              />
            </label>

            <div className="dashboard-form-grid">
              <label className="dashboard-field field-full">
                <span>Target role</span>
                <input name="role" value={form.role} onChange={updateForm} placeholder="e.g. Java Backend Developer" required minLength="1" />
              </label>
              <label className="dashboard-field">
                <span>Interview type</span>
                <select name="interview_type" value={form.interview_type} onChange={updateForm}>
                  <option value="technical">Technical</option>
                  <option value="behavioral">Behavioral</option>
                  <option value="mixed">Mixed</option>
                </select>
              </label>
              <label className="dashboard-field">
                <span>Difficulty</span>
                <select name="difficulty" value={form.difficulty} onChange={updateForm}>
                  <option value="easy">Easy</option>
                  <option value="medium">Medium</option>
                  <option value="hard">Hard</option>
                </select>
              </label>
              <label className="dashboard-field field-full">
                <span>Number of questions</span>
                <select name="number_of_questions" value={form.number_of_questions} onChange={updateForm}>
                  {[3, 5, 7, 10].map((count) => <option key={count} value={count}>{count} questions</option>)}
                </select>
              </label>
            </div>

            <button className="dashboard-start-button" type="submit" disabled={loading}>
              {loading ? <><span className="dashboard-spinner" /> Preparing your interview…</> : <>Start interview <span aria-hidden="true">→</span></>}
            </button>
            <p className="setup-footnote">Only the selected document is used to add context to your practice questions.</p>
          </form>

          <aside className="dashboard-aside">
            <section className="dashboard-panel recent-panel">
              <div className="aside-heading">
                <div><span className="section-kicker">DOCUMENT LIBRARY</span><h3>Your documents</h3></div>
              </div>
              {documents.length === 0 ? (
                <div className="recent-empty">
                  <strong>No documents uploaded yet</strong>
                  <p>Upload a resume or study material to use it as interview context.</p>
                </div>
              ) : (
                <>
                  <label className="dashboard-field">
                    <span>Choose context for this interview</span>
                    <select
                      value={documentId}
                      onChange={(event) => setDocumentId(event.target.value)}
                    >
                      <option value="">No document selected</option>
                      {documents.map((document) => (
                        <option
                          key={document.document_id}
                          value={document.document_id}
                          disabled={document.processing_status !== "ready"}
                        >
                          {document.filename} · {document.document_type === "resume" ? "Resume" : "Study material"} · {document.processing_status}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button
                    className="button button-secondary"
                    type="button"
                    onClick={() => onDeleteDocument(documentId)}
                    disabled={loading || !documentId}
                  >
                    Delete selected document
                  </button>
                </>
              )}
            </section>

            <section className="dashboard-panel recent-panel">
              <div className="aside-heading">
                <div><span className="section-kicker">CONTINUE PRACTISING</span><h3>Recent interviews</h3></div>
                <button className="dashboard-text-link" type="button" onClick={onHistory}>See all</button>
              </div>
              {recentSessions.length === 0 ? (
                <div className="recent-empty">
                  <span aria-hidden="true">◷</span>
                  <strong>Your recent sessions will show here</strong>
                  <p>Complete your first practice interview to start building your history.</p>
                </div>
              ) : (
                <div className="recent-list">
                  {recentSessions.map((item) => (
                    <button className="recent-session" type="button" key={item.session_id} onClick={() => onResume(item.session_id)}>
                      <span className="recent-avatar">{item.role?.slice(0, 1).toUpperCase() || "I"}</span>
                      <span className="recent-session-copy"><strong>{item.role || "Interview practice"}</strong><small>{item.interview_type} · {item.status}</small></span>
                      <span className="recent-session-arrow" aria-hidden="true">→</span>
                    </button>
                  ))}
                </div>
              )}
            </section>

            <section className="dashboard-panel tips-panel">
              <span className="section-kicker">A BETTER WAY TO PREPARE</span>
              <h3>Make each answer count.</h3>
              <div className="tip-row"><span>01</span><p><strong>Think out loud</strong><small>Explain the reasoning behind your answer.</small></p></div>
              <div className="tip-row"><span>02</span><p><strong>Use real examples</strong><small>Connect your answers to projects you have built.</small></p></div>
              <div className="tip-row"><span>03</span><p><strong>Learn from feedback</strong><small>Turn weak areas into your next practice goals.</small></p></div>
            </section>
          </aside>
        </div>
      </section>
    </div>
  );
}

function History({ sessions, loading, onResume, onBack }) {
  return (
    <section className="content-page">
      <div className="page-heading"><div><p className="eyebrow">YOUR PRACTICE</p><h1>Interview history</h1><p>Pick up where you left off or revisit a completed report.</p></div><button className="button button-secondary" onClick={onBack}>← Dashboard</button></div>
      <div className="panel history-panel">
        {sessions.length === 0 ? <div className="empty-state"><span>◷</span><h2>No sessions yet</h2><p>Start an interview practice to see your progress here.</p></div> : sessions.map((item) => (
          <article className="history-item" key={item.session_id}>
            <div className="history-avatar">{item.role.slice(0, 1).toUpperCase()}</div>
            <div className="history-copy"><h3>{item.role}</h3><p>{item.interview_type} interview · {item.question_count} of {item.target_question_count} questions · {item.status}</p></div>
            {item.overall_score != null && <span className="history-score">{item.overall_score}<small>/10</small></span>}
            <button className="button button-secondary" onClick={() => onResume(item.session_id)} disabled={loading}>{item.status === "completed" ? "View report" : "Continue"} <span>→</span></button>
          </article>
        ))}
      </div>
    </section>
  );
}

function InterviewRoom({
  session,
  question,
  answer,
  evaluation,
  loading,
  setAnswer,
  onSubmit,
  onNext,
  onReport,
  onExit,
}) {
  const total = session?.number_of_questions ?? session?.target_question_count ?? 5;
  const current = question?.question_number ?? Math.min((session?.current_question ?? 0) + 1, total);
  const progress = Math.min((current / total) * 100, 100);
  return (
    <section className="content-page room-page">
      <div className="room-topline"><button className="text-button" onClick={onExit}>← Exit practice</button><span className="live-label"><span /> SESSION IN PROGRESS</span></div>
      <div className="room-title"><div><p className="eyebrow">{session?.role} · {session?.interview_type}</p><h1>Interview practice</h1></div><div className="question-count">Question <strong>{current}</strong> <span>of {total}</span></div></div>
      <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
      <div className="room-layout">
        <div className="panel question-panel">
          <div className="question-meta"><span className="step-pill">QUESTION {current}</span>{question?.difficulty && <span className={`difficulty difficulty-${question.difficulty}`}>{question.difficulty} difficulty</span>}</div>
          <h2>{question?.question || (loading ? "Preparing your question…" : "Your next question is ready when you are.")}</h2>
          {!evaluation ? (
            question?.question ? (
              <form onSubmit={onSubmit}>
                <label className="field answer-field">Your answer
                  <textarea value={answer} onChange={(event) => setAnswer(event.target.value)} placeholder="Take a moment to think, then write your answer…" rows="8" required disabled={loading} />
                </label>
                <div className="answer-footer"><span>Tip: explain your approach and include a specific example.</span><button className="button button-primary" disabled={loading || !answer.trim()}>{loading ? "Evaluating…" : "Submit answer"} <span>→</span></button></div>
              </form>
            ) : (
              <button className="button button-primary" onClick={onNext} disabled={loading}>{loading ? "Preparing…" : "Load question"} <span>→</span></button>
            )
          ) : (
            <EvaluationCard evaluation={evaluation} loading={loading} onNext={evaluation.next_question_available ? onNext : onReport} />
          )}
        </div>
        <aside className="panel coach-panel"><span className="coach-icon">✦</span><p className="eyebrow">A LITTLE REMINDER</p><h3>It’s a conversation, not a test.</h3><p>Think out loud, be specific about what you did, and share what you learned along the way.</p><div className="coach-divider" /><div className="coach-detail"><span>◷</span><small>Your progress is saved automatically. You can come back to this session later.</small></div></aside>
      </div>
    </section>
  );
}

function EvaluationCard({ evaluation, loading, onNext }) {
  const score = evaluation.score ?? 0;
  return (
    <section className="evaluation-card" aria-live="polite">
      <div className="evaluation-heading"><div><p className="eyebrow">ANSWER REVIEW</p><h3>Good work. Here’s your feedback.</h3></div><div className="score-badge"><strong>{score}</strong><span>/10</span></div></div>
      <div className="score-grid">
        {[
          ["Technical", evaluation.technical_correctness],
          ["Relevance", evaluation.relevance],
          ["Completeness", evaluation.completeness],
          ["Clarity", evaluation.clarity],
        ].map(([label, value]) => <div key={label}><span>{label}</span><strong>{value ?? "—"}<small>/10</small></strong></div>)}
      </div>
      <div className="feedback-columns">
        <FeedbackList title="What went well" items={evaluation.strengths} tone="positive" />
        <FeedbackList title="Try next time" items={evaluation.weaknesses?.length ? evaluation.weaknesses : evaluation.improvements} tone="improve" />
      </div>
      {evaluation.improvement_suggestions?.length > 0 && <p className="recommendation"><strong>Next step:</strong> {evaluation.improvement_suggestions[0]}</p>}
      <button className="button button-primary" onClick={onNext} disabled={loading}>{loading ? "Loading…" : evaluation.next_question_available ? "Continue to next question" : "View final report"} <span>→</span></button>
    </section>
  );
}

function FeedbackList({ title, items = [], tone }) {
  return <div className={`feedback-list ${tone}`}><h4>{title}</h4>{items?.length ? <ul>{items.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul> : <p>No specific notes for this answer.</p>}</div>;
}

function Report({ report, loading, onHistory, onDashboard }) {
  if (!report) return <section className="content-page"><div className="panel empty-state"><h2>{loading ? "Building your report…" : "Report unavailable"}</h2></div></section>;
  return (
    <section className="content-page report-page">
      <div className="report-banner"><div><p className="eyebrow">INTERVIEW COMPLETE</p><h1>You showed up. That’s progress.</h1><p>Here’s a clear picture of what went well and what to work on next.</p></div><div className="overall-score"><strong>{report.overall_score}</strong><span> / 10<br />overall score</span></div></div>
      <div className="report-summary">
        <div className="panel summary-card"><span className="summary-icon">✓</span><p className="eyebrow">STRENGTHS</p><h2>Keep doing this</h2><FeedbackList title="" items={report.strengths} tone="positive" /></div>
        <div className="panel summary-card"><span className="summary-icon warm">↗</span><p className="eyebrow">GROWTH AREAS</p><h2>Focus your practice</h2><FeedbackList title="" items={report.weaknesses} tone="improve" /></div>
      </div>
      <div className="panel recommendations-panel"><p className="eyebrow">YOUR NEXT STEPS</p><h2>Small improvements, stronger interviews.</h2>{report.recommendations.length ? <ol>{report.recommendations.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ol> : <p className="muted">Keep practicing with role-specific examples and clear reasoning.</p>}</div>
      <div className="panel question-results"><div className="panel-heading compact"><div><p className="eyebrow">SESSION BREAKDOWN</p><h2>{report.role} · {report.question_count} questions</h2></div><span className="step-pill">{report.interview_type}</span></div>{report.question_results.map((item) => <article className="result-item" key={item.question_number}><div className="result-number">{String(item.question_number).padStart(2, "0")}</div><div className="result-body"><h3>{item.question}</h3><p>{item.answer}</p></div><span className="result-score">{item.score}<small>/10</small></span></article>)}</div>
      <div className="report-actions"><button className="button button-secondary" onClick={onHistory}>View interview history</button><button className="button button-primary" onClick={onDashboard}>Start another practice <span>→</span></button></div>
    </section>
  );
}

export default App;
