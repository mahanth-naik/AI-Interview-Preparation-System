# AI Interview Preparation System

ADP is a document-aware interview practice application. Users can upload resumes or study materials, select one as interview context, and practice through the React interface backed by authenticated FastAPI APIs. User accounts and document ownership are stored in SQLite, and document chunks are indexed in ChromaDB.

## Features

- Upload resume or study-material PDFs, extract their text, chunk it, and store owner- and type-tagged chunks in ChromaDB.
- List, inspect, select, and delete owned documents and their associated vectors.
- Create technical, behavioral, or mixed interview sessions for a target role.
- Generate one question at a time using the selected provider and optional selected-document context.
- Evaluate submitted answers, retain structured feedback, and adjust the next question's difficulty.
- Resume in-progress sessions after restarting the backend.
- Review completed reports with an overall score, per-question scores, strengths, weaknesses, and recommendations.
- Browse and continue previous sessions from the frontend.
- Register and sign in with Argon2-hashed passwords and revocable JWT access tokens.
- Keep document retrieval, interview sessions, history, evaluations, and reports scoped to the signed-in user.

## Architecture

```text
Resume or study PDF -> pypdf -> owner/type-tagged chunks -> ChromaDB
                                                            |
React UI -> FastAPI -> JWT identity -> selected-document RAG context
                           |          -> InterviewService -> selected LLM provider
                           |                           -> SQLite session and report data
                           +-> SQLite accounts, document ownership, and token revocation
```

`backend/main.py` defines the HTTP API and applies the authenticated user identity to document and interview operations. `backend/services/interview.py` owns interview progression and report creation. `backend/database/session_store.py` persists the owner and session fields, including JSON question/answer/evaluation history, in `backend/database/interviews.sqlite3` by default. `backend/database/document_store.py` records document owner, type, filename, processing status, and upload time in that same database. Existing document rows are migrated additively as resumes in the ready state. `INTERVIEW_DB_PATH` can override this location. ChromaDB continues to persist document vectors separately under `backend/database/chroma/`; new chunks carry `document_id`, `user_id`, and `document_type` metadata. Searches filter by owner and can additionally filter by document type or document ID.

The frontend lives in `frontend/` and talks to FastAPI. Users choose whether to upload a resume or study material and select one ready document from their library. The selected `document_id` is sent when starting a session, so interview retrieval cannot mix in another uploaded document. The `session_id` returned by session creation is used for question, answer, resume, status, and report requests.

## LLM configuration

Create `.env` from `.env.example` only when you do not already have a local `.env`. The default `LLM_PROVIDER=local` runs deterministic, offline question and evaluation logic. To use Gemini with AutoGen, set:

```text
LLM_PROVIDER=autogen
GEMINI_MODEL=gemini-3-flash-preview
GEMINI_API_KEY=your-key
```

The key is read from the environment and must not be committed. AutoGen builds a `SelectorGroupChat` containing separate interviewer and evaluator agents; both use Gemini through Google's OpenAI-compatible endpoint. `LLM_PROVIDER=gemini` selects the direct Gemini adapter without AutoGen. `LLM_PROVIDER=openai` selects the OpenAI adapter. The example configuration keeps the offline provider as its default to avoid making external calls unexpectedly.

`FRONTEND_ORIGINS` is a comma-separated CORS allow-list and defaults to the Vite development origins `http://localhost:5173,http://127.0.0.1:5173`.

## Authentication and ownership

The backend exposes:

- `POST /auth/register` with `full_name`, `email`, and a password (12–128 characters).
- `POST /auth/login` with `email` and `password`; returns a short-lived bearer access token and public user details.
- `GET /auth/me` with `Authorization: Bearer <access_token>`.
- `POST /auth/logout` with the same bearer token; stores its JWT ID in SQLite so it cannot be reused.

Passwords are hashed with pwdlib's recommended Argon2id hasher. Only the password hash is saved in the `users` table. JWTs use HS256, include an issuer and expiration, and are checked against the SQLite revocation table. All resume, RAG search, interview, history, status, and report endpoints require a bearer token. SQLite checks the current user's ownership before returning or changing a session/document, and ChromaDB searches include a `user_id` metadata filter. Configure `JWT_SECRET_KEY` in the ignored local `.env` with a randomly generated value of at least 32 characters. For example, generate a value locally in PowerShell with:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Set `JWT_ACCESS_TOKEN_MINUTES` to an integer from 1 to 1440; it defaults to 30. Authentication routes return `503` when the JWT secret or lifetime configuration is invalid. Logout revokes that token immediately. The React app keeps the access token in tab-scoped `sessionStorage`, clears it on logout, and returns to sign-in when the API rejects an expired token.

## Setup on Windows

Use two PowerShell terminals from the repository root.

### Backend

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
py -m venv .venv              # only if the environment does not exist
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
Set-Location backend
python -m uvicorn main:app --reload
```

The API is at `http://127.0.0.1:8000`; Swagger UI is at `http://127.0.0.1:8000/docs`.

### Frontend

```powershell
Set-Location frontend
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
npm install
npm run dev
```

Open the Vite URL printed in the terminal (normally `http://localhost:5173`).

## API workflow

### Authentication

1. `POST /auth/register` with `{"full_name":"Ada Lovelace","email":"ada@example.com","password":"a-long-password-value"}` creates a SQLite user.
2. `POST /auth/login` returns an access token; include it as `Authorization: Bearer <access_token>` on protected requests.
3. `GET /auth/me` returns the authenticated user's public profile.
4. `POST /auth/logout` revokes the presented access token.

### Interview and document workflow (authenticated)

Include the bearer token from login in every request below.

1. `POST /documents/upload` with multipart field `file` (PDF) and optional `document_type` (`resume` by default or `study_material`) returns a document ID and processing status. Uploads are limited to 20 MiB and require extractable text.
2. `GET /documents` lists the signed-in user's documents; `GET /documents/{document_id}` returns one owned document; `DELETE /documents/{document_id}` removes its vectors and metadata. Foreign or missing document IDs return `404`.
3. `GET /documents/search` and `GET /ask` accept optional `document_id` and `document_type` filters in addition to the authenticated user filter.
4. `POST /interview/start` with JSON `role`, `interview_type`, optional `difficulty`, `number_of_questions`, and optional owned, ready `document_id` returns `session_id`.
5. `POST /interview/question` with `{"session_id":"..."}` returns the active question. Calling it again after evaluation generates the next question.
6. `POST /interview/evaluate` with `session_id`, the active `question`, and `answer` evaluates and saves the answer.
7. `GET /interview/{session_id}` returns the persisted session state.
8. `GET /interview/{session_id}/report` returns a final report after all configured questions are evaluated. It responds with `409` while the interview is still active.
9. `GET /interviews?limit=50` lists recent sessions for the dashboard/history screen.

The service rejects answers for a question that is not the active session question. Question and answer history, active question, difficulty, and completion status are saved to SQLite.

## Tests and verification

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q backend\tests
.\.venv\Scripts\python.exe -m compileall -q backend\main.py backend\services backend\database backend\tests
Set-Location frontend
npm test
npm run build
```

Backend tests use fake providers and temporary SQLite databases, so they do not call Gemini or consume API credits. Frontend API tests use mocked `fetch`; `npm run build` checks the React production bundle.

## Project structure

```text
backend/
  database/
    session_store.py
    vector_store.py
  services/
    ai_provider.py
    autogen_orchestration.py
    interview.py
    rag.py
    chunking.py
  tests/
frontend/
  src/
    App.jsx
    api.js
    styles.css
```

## Current boundaries

- SQLite and ChromaDB are local persistent stores; configure durable storage/backups for deployment.
- Existing pre-authentication interview rows are preserved by the additive `user_id` migration, but remain unowned and are not exposed to any account. Existing ChromaDB chunks without `user_id` metadata are likewise preserved but excluded from authenticated searches. No automatic account claiming is performed because the original owner cannot be established safely.
- The API does not include refresh tokens or account recovery.
- Resume and study-material uploads accept PDF files up to 20 MiB with extractable text. Scanned PDFs requiring OCR are not supported.
- The offline provider is intended for local demos/tests; realistic LLM feedback requires a configured external provider.
