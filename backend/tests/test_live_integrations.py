import importlib.util
import os
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import chromadb
import pytest
from dotenv import load_dotenv
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

import main
from database import vector_store
from database.session_store import SQLiteSessionStore
from services.ai_provider import (
    EvaluationResult,
    GeminiProvider,
    LLMProviderError,
    get_interview_provider,
)
from services.chunking import chunk_pages
from services.interview import InterviewService
from services.rag import retrieve_context


_RUN_LIVE = os.getenv("ADP_RUN_LIVE_INTEGRATION_TESTS") == "1"
_HAS_GEMINI_KEY = bool(os.getenv("GEMINI_API_KEY"))
_HAS_AUTOGEN = all(
    importlib.util.find_spec(name)
    for name in ("autogen_agentchat", "autogen_ext")
)


@pytest.fixture
def real_chroma_collection(tmp_path, monkeypatch):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    name = f"adp-integration-{uuid4().hex}"
    collection = client.create_collection(name=name)
    monkeypatch.setattr(vector_store, "collection", collection)
    try:
        yield collection
    finally:
        client.delete_collection(name)


@pytest.fixture(scope="session")
def valid_gemini_configuration():
    if not _HAS_GEMINI_KEY:
        pytest.skip("GEMINI_API_KEY is not configured")
    try:
        question = GeminiProvider().generate_question(
            "Engineer",
            "technical",
            "easy",
            [],
            "Synthetic integration-test context.",
        )
    except LLMProviderError as exc:
        status_code = getattr(exc.__cause__, "status_code", None)
        reason = (
            f"Gemini preflight failed with HTTP {status_code}"
            if status_code is not None
            else "Gemini preflight failed with a network/provider error"
        )
        pytest.skip(reason)
    if not question.strip():
        pytest.skip("Gemini preflight returned an empty response")
    return True


def _synthetic_pdf(page_texts: list[str]) -> bytes:
    page_ids = list(range(3, 3 + len(page_texts)))
    font_id = 3 + len(page_texts)
    content_ids = list(range(font_id + 1, font_id + 1 + len(page_texts)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Kids "
            f"[{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] "
            f"/Count {len(page_texts)} >>"
        ).encode(),
    ]
    for content_id in content_ids:
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode()
        )
    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )
    for text in page_texts:
        escaped = (
            text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"\nendstream"
        )

    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, item in enumerate(objects, start=1):
        offsets.append(len(data))
        data.extend(
            f"{number} 0 obj\n".encode()
            + item
            + b"\nendobj\n"
        )

    xref_offset = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF"
        ).encode()
    )
    return bytes(data)


def test_synthetic_pdf_builder_preserves_extractable_page_text():
    reader = PdfReader(
        BytesIO(
            _synthetic_pdf(
                ["Synthetic first page.", "Synthetic second page."]
            )
        )
    )

    assert [page.extract_text() for page in reader.pages] == [
        "Synthetic first page.",
        "Synthetic second page.",
    ]


def _expected_next_difficulty(current: str, evaluation: dict) -> str:
    levels = ("easy", "medium", "hard")
    concerns = any(
        isinstance(item, str) and item.strip()
        for field in ("weaknesses", "improvement_suggestions")
        for item in evaluation.get(field, [])
    )
    index = levels.index(current)
    if evaluation["score"] <= 4 or (
        evaluation["score"] <= 6 and concerns
    ):
        index = max(0, index - 1)
    elif evaluation["score"] >= 8 and not concerns:
        index = min(len(levels) - 1, index + 1)
    return levels[index]


@pytest.mark.skipif(
    not _RUN_LIVE,
    reason="set ADP_RUN_LIVE_INTEGRATION_TESTS=1 to run live integrations",
)
@pytest.mark.usefixtures("real_chroma_collection")
def test_real_chromadb_insert_retrieve_and_scope_filtering():
    owner_id = uuid4().hex
    document_id = uuid4().hex
    chunks = chunk_pages(
        [
            (1, "Introduction to synthetic distributed systems notes."),
            (
                2,
                "Replication stores copies of data on nodes; "
                "consensus makes nodes agree on a value.",
            ),
        ]
    )

    vector_store.add_chunks(
        chunks,
        "synthetic-network-notes.pdf",
        user_id=owner_id,
        document_type="study_material",
        document_id=document_id,
    )
    evidence = retrieve_context(
        "How do consensus and replication work?",
        10,
        document_id=document_id,
        user_id=owner_id,
        document_type="study_material",
    )

    assert evidence
    assert {item["page_number"] for item in evidence} == {1, 2}
    assert all(item["document_id"] == document_id for item in evidence)
    assert all(item["document_type"] == "study_material" for item in evidence)
    assert all(item["filename"] == "synthetic-network-notes.pdf" for item in evidence)
    assert all(item["chunk_id"].startswith("chunk-") for item in evidence)
    assert retrieve_context(
        "consensus",
        document_id=document_id,
        user_id="another-synthetic-user",
        document_type="study_material",
    ) == []
    assert retrieve_context(
        "consensus",
        document_id=uuid4().hex,
        user_id=owner_id,
        document_type="study_material",
    ) == []
    assert retrieve_context(
        "consensus",
        document_id=document_id,
        user_id=owner_id,
        document_type="resume",
    ) == []


@pytest.mark.skipif(
    not _RUN_LIVE or not _HAS_GEMINI_KEY,
    reason="requires ADP_RUN_LIVE_INTEGRATION_TESTS=1 and GEMINI_API_KEY",
)
def test_real_gemini_provider_selection_question_and_evaluation(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    provider = get_interview_provider()

    assert isinstance(provider, GeminiProvider)
    try:
        question = provider.generate_question(
            "Network Engineer",
            "technical",
            "medium",
            [],
            (
                "Study-material evidence:\n"
                "TCP uses a congestion window to limit unacknowledged data."
            ),
            context_mode="study_material",
        )
    except LLMProviderError as exc:
        status_code = getattr(exc.__cause__, "status_code", None)
        pytest.fail(
            "Configured Gemini question request failed"
            + (f" (HTTP {status_code})." if status_code else ".")
        )
    assert question.strip()

    try:
        evaluation = provider.evaluate_answer(
            "Network Engineer",
            question,
            "The congestion window limits how much unacknowledged data can be sent.",
            (
                "Study-material evidence:\n"
                "TCP uses a congestion window to limit unacknowledged data."
            ),
            "medium",
            context_mode="study_material",
        )
    except LLMProviderError as exc:
        status_code = getattr(exc.__cause__, "status_code", None)
        pytest.fail(
            "Configured Gemini evaluation request failed"
            + (f" (HTTP {status_code})." if status_code else ".")
        )
    assert isinstance(evaluation, EvaluationResult)
    assert 1 <= evaluation.score <= 10
    assert evaluation.next_difficulty in {"easy", "medium", "hard"}


@pytest.mark.skipif(
    not _RUN_LIVE,
    reason="requires ADP_RUN_LIVE_INTEGRATION_TESTS=1",
)
def test_real_gemini_invalid_api_key_is_reported_without_fallback():
    provider = GeminiProvider(
        api_key="adp-invalid-integration-test-key",
        model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
    )

    with pytest.raises(LLMProviderError) as error:
        provider.generate_question(
            "Engineer",
            "technical",
            "medium",
            [],
            "Synthetic context.",
        )

    assert getattr(error.value.__cause__, "status_code", None) in {
        400,
        401,
        403,
    }
    error_message = str(error.value.__cause__).casefold()
    assert any(
        marker in error_message
        for marker in ("invalid", "auth", "unauthorized")
    )


@pytest.mark.skipif(
    not _RUN_LIVE or not _HAS_GEMINI_KEY,
    reason="requires ADP_RUN_LIVE_INTEGRATION_TESTS=1 and GEMINI_API_KEY",
)
@pytest.mark.usefixtures("valid_gemini_configuration")
def test_real_gemini_unavailable_model_is_reported():
    provider = GeminiProvider(
        model="adp-model-that-does-not-exist",
    )

    with pytest.raises(LLMProviderError) as error:
        provider.generate_question(
            "Engineer",
            "technical",
            "medium",
            [],
            "Synthetic context.",
        )

    assert getattr(error.value.__cause__, "status_code", None) in {
        400,
        404,
    }
    error_message = str(error.value.__cause__).casefold()
    assert "model" in error_message or "not found" in error_message


@pytest.mark.skipif(
    not _RUN_LIVE or not _HAS_GEMINI_KEY,
    reason="requires ADP_RUN_LIVE_INTEGRATION_TESTS=1 and GEMINI_API_KEY",
)
@pytest.mark.usefixtures("real_chroma_collection")
@pytest.mark.usefixtures("valid_gemini_configuration")
@pytest.mark.parametrize(
    ("document_type", "filename", "pages", "role", "answer"),
    [
        (
            "resume",
            "synthetic-resume.pdf",
            [
                "Synthetic candidate profile for integration testing.",
                "Skills: Python, FastAPI, SQLite. Project: built an "
                "inventory analytics API and tested query results.",
            ],
            "Software Engineer",
            "I used Python and FastAPI to build the inventory analytics API, "
            "stored records in SQLite, and tested the query results.",
        ),
        (
            "study_material",
            "synthetic-study-material.pdf",
            [
                "Synthetic introductory notes for integration testing.",
                "A distributed system coordinates independent computers "
                "through messages. Replication stores copies of data on "
                "multiple nodes, and consensus helps nodes agree on a value.",
            ],
            "Computer Networks Student",
            "Replication stores copies on multiple nodes, while consensus "
            "helps the nodes agree on a value.",
        ),
    ],
)
def test_real_api_document_interview_and_report_workflow(
    authenticated_client,
    monkeypatch,
    document_type,
    filename,
    pages,
    role,
    answer,
):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    provider = get_interview_provider()
    assert provider.name == "gemini"

    database_path = main.document_store.database_path
    main.interview_service = InterviewService(
        store=SQLiteSessionStore(database_path),
        provider=provider,
    )

    upload = authenticated_client.post(
        "/documents/upload",
        data={"document_type": document_type},
        files={
            "file": (
                filename,
                _synthetic_pdf(pages),
                "application/pdf",
            )
        },
    )
    assert upload.status_code == 200
    document = upload.json()
    document_id = document["document_id"]
    assert document["document_type"] == document_type
    owner_id = authenticated_client.get("/auth/me").json()["user"]["user_id"]
    assert main.document_store.get_for_user(
        document_id,
        owner_id,
    )["processing_status"] == "ready"

    assert retrieve_context(
        "synthetic document",
        document_id=document_id,
        user_id="another-synthetic-user",
        document_type=document_type,
    ) == []

    started = authenticated_client.post(
        "/interview/start",
        json={
            "role": role,
            "interview_type": "technical",
            "difficulty": "medium",
            "number_of_questions": 2,
            "document_id": document_id,
        },
    )
    assert started.status_code == 200
    assert started.json()["provider"] == "gemini"
    session_id = started.json()["session_id"]

    first = authenticated_client.post(
        "/interview/question",
        json={"session_id": session_id},
    )
    assert first.status_code == 200
    first_question = first.json()
    assert first_question["question_number"] == 1
    assert first_question["is_follow_up"] is False
    assert first_question["evidence"]
    assert all(
        item["document_id"] == document_id
        and item["document_type"] == document_type
        and item["filename"] == filename
        and item["page_number"] in {1, 2}
        and item["chunk_id"].startswith("chunk-")
        for item in first_question["evidence"]
    )

    first_evaluation = authenticated_client.post(
        "/interview/evaluate",
        json={
            "session_id": session_id,
            "question": first_question["question"],
            "answer": answer,
        },
    )
    assert first_evaluation.status_code == 200
    evaluation = first_evaluation.json()
    assert evaluation["evidence"]
    assert evaluation["next_question_available"] is True
    assert evaluation["next_difficulty"] == _expected_next_difficulty(
        "medium",
        evaluation,
    )

    second = authenticated_client.post(
        "/interview/question",
        json={"session_id": session_id},
    )
    assert second.status_code == 200
    second_question = second.json()
    assert second_question["question_number"] == 2
    assert second_question["is_follow_up"] is evaluation["follow_up_required"]
    assert second_question["difficulty"] == evaluation["next_difficulty"]

    second_evaluation = authenticated_client.post(
        "/interview/evaluate",
        json={
            "session_id": session_id,
            "question": second_question["question"],
            "answer": "I cannot add more detail to that response.",
        },
    )
    assert second_evaluation.status_code == 200
    assert second_evaluation.json()["next_question_available"] is False

    report_response = authenticated_client.get(
        f"/interview/{session_id}/report"
    )
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["status"] == "completed"
    assert report["question_count"] == 2
    assert [item["is_follow_up"] for item in report["question_results"]] == [
        False,
        evaluation["follow_up_required"],
    ]
    for result in report["question_results"]:
        assert result["question_evidence"]
        assert result["evaluation_evidence"]
        assert all(
            evidence["document_id"] == document_id
            and evidence["document_type"] == document_type
            and evidence["page_number"] in {1, 2}
            and evidence["chunk_id"].startswith("chunk-")
            for evidence in (
                result["question_evidence"]
                + result["evaluation_evidence"]
            )
        )
    assert owner_id


@pytest.mark.skipif(
    not _RUN_LIVE or not _HAS_GEMINI_KEY or not _HAS_AUTOGEN,
    reason=(
        "requires ADP_RUN_LIVE_INTEGRATION_TESTS=1, GEMINI_API_KEY, "
        "and AutoGen dependencies"
    ),
)
@pytest.mark.usefixtures("valid_gemini_configuration")
def test_real_autogen_provider_selection_and_question_generation(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "autogen")
    provider = get_interview_provider()

    from services.autogen_orchestration import AutoGenOrchestrator

    assert provider.name == "autogen"
    assert isinstance(provider.orchestrator, AutoGenOrchestrator)
    question = provider.generate_question(
        "Network Engineer",
        "technical",
        "medium",
        [],
        "TCP uses a congestion window to limit unacknowledged data.",
        context_mode="study_material",
    )
    assert question.strip()


@pytest.mark.skipif(
    not _RUN_LIVE or not _HAS_GEMINI_KEY or not _HAS_AUTOGEN,
    reason=(
        "requires ADP_RUN_LIVE_INTEGRATION_TESTS=1, GEMINI_API_KEY, "
        "and AutoGen dependencies"
    ),
)
@pytest.mark.usefixtures("valid_gemini_configuration")
def test_real_autogen_provider_answer_evaluation(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "autogen")
    provider = get_interview_provider()

    evaluation = provider.evaluate_answer(
        "Network Engineer",
        "What does a TCP congestion window control?",
        "It limits unacknowledged data in transit.",
        "TCP uses a congestion window to limit unacknowledged data.",
        "medium",
        context_mode="study_material",
    )
    assert isinstance(evaluation, EvaluationResult)
    assert 1 <= evaluation.score <= 10
