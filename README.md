# AI Interview Preparation System

Backend foundation for a college Additional Project (ADP) that prepares candidates
for interviews using document retrieval and an interview-session workflow. The
frontend is not implemented yet.

## Current Progress

Weeks 1-4 are complete for the backend foundation: PDF ingestion, text chunking,
Chroma vector storage, semantic retrieval, offline question generation, answer
evaluation, and in-memory interview sessions. AutoGen and an external LLM are not
currently installed or integrated.

## Architecture

```text
PDF resume/document
	-> pypdf text extraction
	-> validated text chunks
	-> ChromaDB persistent vector collection
	-> semantic retrieval
	-> local interview provider
	-> questions and structured answer feedback
```

The backend keeps route handling in `main.py`. Chunking, retrieval, AI-provider,
and session logic live under `backend/services/`. Chroma data is persisted under
`backend/database/chroma/`, and uploaded source files are kept under
`backend/uploads/` for local development.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | API status message |
| GET | `/health` | Health check |
| POST | `/documents/upload` | Upload a readable PDF, extract text, chunk it, and store vectors |
| GET | `/documents/search?query=Python&n_results=3` | Return relevant chunks with source and distance |
| GET | `/ask?query=Python&n_results=3` | Backward-compatible retrieval context response |
| POST | `/interview/start` | Create a session and generate role-specific questions from retrieved context |
| POST | `/interview/question` | Return the current question for a session |
| POST | `/interview/evaluate` | Evaluate an answer and advance the session |
| GET | `/interview/{session_id}` | Return session state, answers, and evaluations |

Example start request:

```json
{
	"role": "Python Developer",
	"interview_type": "technical",
	"difficulty": "medium",
	"number_of_questions": 5
}
```

The response includes a `session_id`, question count, and `provider: "local"`.
Evaluation responses contain `score`, `feedback`, `strengths`, and `improvements`.

## Technologies

- Python 3.14
- FastAPI and Pydantic
- pypdf
- ChromaDB
- pytest

## Setup and Run (Windows PowerShell)

Use the existing project environment:

```powershell
cd C:\Users\mahan\OneDrive\Desktop\AI-Interview-Preparation-System
.\backend\venv\Scripts\Activate.ps1
python -m pip install -r .\backend\requirements.txt
cd .\backend
python -m uvicorn main:app --reload
```

The API is available at `http://127.0.0.1:8000`; interactive documentation is
available at `/docs`.

## Testing and Validation

```powershell
cd C:\Users\mahan\OneDrive\Desktop\AI-Interview-Preparation-System\backend
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m compileall -q main.py services database tests
```

The test suite uses the local provider and mocks retrieval where appropriate, so
it does not make paid API calls. No API key is required for the current provider.
If a real LLM provider is added later, credentials must come from environment
variables and never from source code.

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
		interview.py
	tests/
		test_backend.py
	uploads/                  # ignored local uploads
```

`backend/.gitignore` excludes `venv/`, caches, uploads, Chroma runtime data, and
`.env`. The root `.venv/` is also ignored by the local environment tooling; it is
not required by this application. The project environment is `backend/venv/`.

## Future Work (Weeks 5-6)

- Integrate and configure a real LLM provider through environment variables.
- Replace the local provider with an LLM-backed question and evaluation provider.
- Add persistent session storage and authentication.
- Add richer document formats and better extraction quality.
- Build the frontend and connect it to these APIs.
