# AI Interview Preparation System

Backend foundation for a college Additional Project (ADP) that prepares candidates for interviews using document retrieval and an interview-session workflow. The frontend is not implemented yet.

## Current Progress

The backend supports PDF ingestion, text chunking, ChromaDB semantic retrieval, RAG-backed interview turns, an LLM provider abstraction, structured answer evaluation, and AutoGen interviewer/evaluator orchestration. The local provider is the default, so development and tests work without an API key.

## Architecture

```text
PDF resume/document
  -> pypdf text extraction
  -> validated text chunks
  -> ChromaDB persistent vector collection
  -> retrieval for each interview turn
  -> retrieval/context construction
  -> AutoGen SelectorGroupChat -> InterviewerAgent -> question
  -> AutoGen SelectorGroupChat -> EvaluatorAgent -> structured evaluation
```

```text
start session -> retrieve context -> generate question
       ^                                  |
       |                                  v
next question <- evaluate answer <- submit answer
                 (difficulty and follow-up update)
```

The backend keeps route handling in `main.py`. Chunking, retrieval, provider, orchestration, and session logic live under `backend/services/`. Chroma data is persisted under `backend/database/chroma/` and uploaded source files are kept under `backend/uploads/` for local development.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | API status message |
| GET | `/health` | Health check |
| POST | `/documents/upload` | Extract and store a readable PDF |
| GET | `/documents/search?query=Python&n_results=3` | Return relevant chunks |
| GET | `/ask?query=Python&n_results=3` | Backward-compatible context response |
| POST | `/interview/start` | Create a session |
| POST | `/interview/question` | Generate or return the current question |
| POST | `/interview/evaluate` | Evaluate an answer and advance the session |
| GET | `/interview/{session_id}` | Return session state |

Evaluation responses retain `score`, `feedback`, `strengths`, and `improvements`. They also contain validated dimension scores, weaknesses, improvement suggestions, `follow_up_required`, and `next_difficulty`.

## LLM Configuration

Copy `.env.example` to `.env` and set the provider as needed:

```text
LLM_PROVIDER=local
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=
```

Use `LLM_PROVIDER=openai` for direct OpenAI chat completions or `LLM_PROVIDER=autogen` for the AutoGen adapter. The AutoGen adapter creates two distinct `AssistantAgent` instances in one `SelectorGroupChat`: `InterviewerAgent` generates one question and `EvaluatorAgent` returns `EvaluationResult` JSON. Retrieval remains in `InterviewService`; the resulting context is passed into each agent task. The API key is read from the environment only and is never stored in source code. `local` is the offline default.

## Setup and Run (Windows PowerShell)

```powershell
cd C:\Users\mahan\OneDrive\Desktop\AI-Interview-Preparation-System
.\backend\venv\Scripts\Activate.ps1
python -m pip install -r .\backend\requirements.txt
cd .\backend
python -m uvicorn main:app --reload
```

The API is available at `http://127.0.0.1:8000`; interactive documentation is available at `/docs`.

## Testing

```powershell
cd C:\Users\mahan\OneDrive\Desktop\AI-Interview-Preparation-System\backend
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m compileall -q main.py services database tests
```

Tests use fake LLM clients and mocked retrieval, so they do not make paid API calls. They cover missing configuration, malformed responses, RAG-backed question generation, structured evaluation, provider selection, and AutoGen role orchestration. On the current Windows environment, test collection is blocked before any test body runs because Application Control blocks the installed `pydantic_core` DLL. No test pass result is claimed until that environment restriction is resolved by its administrator.

## Project Structure

```text
backend/
  main.py
  requirements.txt
  database/
    vector_store.py
    chroma/                 # ignored generated data
  services/
    chunking.py
    rag.py
    ai_provider.py
    autogen_orchestration.py
    interview.py
  tests/
    test_backend.py
    test_llm_workflow.py
  uploads/                  # ignored local uploads
```

## Remaining Limitations

- Sessions are in memory and are not authenticated or persistent.
- Document extraction currently targets PDFs.
- The frontend is not implemented.
- AutoGen is currently exposed through the existing synchronous provider interface. Its `run_async` path is available for async callers; synchronous FastAPI routes use the boundary adapter without nesting an event loop.
- AutoGen and OpenAI providers require an API key and installed external packages; tests mock those calls.
