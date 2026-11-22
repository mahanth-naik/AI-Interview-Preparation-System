import pytest

import main
from services.interview import InterviewService


def _upload(client, document_type="resume", filename="document.pdf", content=None):
    if content is None:
        content = b"%PDF-1.7\nvalid test document"
    return client.post(
        "/documents/upload",
        data={"document_type": document_type},
        files={"file": (filename, content, "application/pdf")},
    )


def _fake_pdf_reader(monkeypatch, text="Document text for retrieval."):
    class Page:
        def extract_text(self):
            return text

    class Reader:
        pages = [Page()]

    monkeypatch.setattr(main, "PdfReader", lambda _: Reader())


def _fake_vector_add(monkeypatch):
    captured = {}

    def add_chunks(
        chunks,
        filename,
        user_id=None,
        document_type="resume",
        document_id=None,
    ):
        captured.update(
            chunks=chunks,
            filename=filename,
            user_id=user_id,
            document_type=document_type,
            document_id=document_id,
        )
        return {"document_id": document_id, "chunks_stored": len(chunks)}

    monkeypatch.setattr(main, "add_chunks", add_chunks)
    return captured


@pytest.mark.parametrize(
    ("document_type", "filename"),
    [
        ("resume", "resume.pdf"),
        ("study_material", "lecture-notes.pdf"),
    ],
)
def test_upload_supports_resume_and_study_material(
    authenticated_client,
    monkeypatch,
    document_type,
    filename,
):
    _fake_pdf_reader(monkeypatch)
    captured = _fake_vector_add(monkeypatch)

    response = _upload(authenticated_client, document_type, filename)

    assert response.status_code == 200
    result = response.json()
    assert result["document_type"] == document_type
    assert result["processing_status"] == "ready"
    assert captured["document_type"] == document_type
    assert captured["document_id"] == result["document_id"]
    assert captured["chunks"]

    stored = main.document_store.get_for_user(
        result["document_id"],
        captured["user_id"],
    )
    assert stored is not None
    assert stored["filename"] == filename
    assert stored["document_type"] == document_type
    assert stored["processing_status"] == "ready"
    assert stored["created_at"]


def test_document_list_and_detail_are_scoped_to_current_user(
    authenticated_client,
    monkeypatch,
):
    _fake_pdf_reader(monkeypatch)
    _fake_vector_add(monkeypatch)
    uploaded = _upload(authenticated_client, "study_material")
    document_id = uploaded.json()["document_id"]

    listing = authenticated_client.get("/documents")
    detail = authenticated_client.get(f"/documents/{document_id}")

    assert listing.status_code == 200
    assert [item["document_id"] for item in listing.json()["documents"]] == [
        document_id
    ]
    assert detail.status_code == 200
    assert detail.json()["document_type"] == "study_material"
    assert detail.json()["processing_status"] == "ready"
    assert "user_id" not in detail.json()

    registration = authenticated_client.post(
        "/auth/register",
        json={
            "full_name": "Other User",
            "email": "other@example.com",
            "password": "another-long-test-password",
        },
    )
    assert registration.status_code == 201
    login = authenticated_client.post(
        "/auth/login",
        json={
            "email": "other@example.com",
            "password": "another-long-test-password",
        },
    )
    other_headers = {
        "Authorization": f"Bearer {login.json()['access_token']}"
    }

    assert authenticated_client.get(
        "/documents",
        headers=other_headers,
    ).json()["documents"] == []
    assert authenticated_client.get(
        f"/documents/{document_id}",
        headers=other_headers,
    ).status_code == 404


def test_document_deletion_removes_vectors_and_metadata(
    authenticated_client,
    monkeypatch,
):
    _fake_pdf_reader(monkeypatch)
    _fake_vector_add(monkeypatch)
    uploaded = _upload(authenticated_client)
    document_id = uploaded.json()["document_id"]
    current_user = authenticated_client.get("/auth/me").json()["user"]
    deleted_vectors = []
    monkeypatch.setattr(
        main,
        "delete_document_chunks",
        lambda doc_id, user_id: deleted_vectors.append((doc_id, user_id)),
    )

    response = authenticated_client.delete(f"/documents/{document_id}")

    assert response.status_code == 204
    assert deleted_vectors == [(document_id, current_user["user_id"])]
    assert main.document_store.get_for_user(
        document_id,
        current_user["user_id"],
    ) is None
    assert authenticated_client.get(
        f"/documents/{document_id}"
    ).status_code == 404


def test_document_search_filters_by_owner_and_document_type(
    authenticated_client,
    monkeypatch,
):
    current_user = authenticated_client.get("/auth/me").json()["user"]
    main.document_store.register(
        "study-document",
        current_user["user_id"],
        "lecture-notes.pdf",
        document_type="study_material",
    )
    captured = {}

    def fake_retrieve_context(query, n_results, **filters):
        captured.update(query=query, n_results=n_results, filters=filters)
        return []

    monkeypatch.setattr(main, "retrieve_context", fake_retrieve_context)

    response = authenticated_client.get(
        "/documents/search",
        params={
            "query": "operating systems",
            "document_type": "study_material",
        },
    )

    assert response.status_code == 200
    assert captured == {
        "query": "operating systems",
        "n_results": 3,
        "filters": {
            "document_id": None,
            "user_id": current_user["user_id"],
            "document_type": "study_material",
        },
    }


def test_user_cannot_delete_another_users_document(
    authenticated_client,
    monkeypatch,
):
    _fake_pdf_reader(monkeypatch)
    _fake_vector_add(monkeypatch)
    uploaded = _upload(authenticated_client)
    document_id = uploaded.json()["document_id"]
    deleted_vectors = []
    monkeypatch.setattr(
        main,
        "delete_document_chunks",
        lambda *args: deleted_vectors.append(args),
    )

    authenticated_client.post(
        "/auth/register",
        json={
            "full_name": "Other User",
            "email": "other@example.com",
            "password": "another-long-test-password",
        },
    )
    login = authenticated_client.post(
        "/auth/login",
        json={
            "email": "other@example.com",
            "password": "another-long-test-password",
        },
    )
    other_headers = {
        "Authorization": f"Bearer {login.json()['access_token']}"
    }
    response = authenticated_client.delete(
        f"/documents/{document_id}",
        headers=other_headers,
    )

    assert response.status_code == 404
    assert deleted_vectors == []


@pytest.mark.parametrize(
    ("filename", "content", "expected_status"),
    [
        ("notes.txt", b"plain text", 400),
        ("notes.pdf", b"not a PDF", 400),
        ("empty.pdf", b"", 400),
    ],
)
def test_upload_rejects_invalid_pdf_inputs(
    authenticated_client,
    filename,
    content,
    expected_status,
):
    response = _upload(
        authenticated_client,
        filename=filename,
        content=content,
    )

    assert response.status_code == expected_status


def test_upload_rejects_oversized_pdf(authenticated_client, monkeypatch):
    monkeypatch.setattr(main, "MAX_DOCUMENT_UPLOAD_SIZE_BYTES", 8)

    response = _upload(
        authenticated_client,
        content=b"%PDF-1.7\nmore than eight bytes",
    )

    assert response.status_code == 413


def test_upload_rejects_unknown_document_type(authenticated_client):
    response = _upload(authenticated_client, document_type="transcript")

    assert response.status_code == 422


def test_unreadable_pdf_is_marked_failed(
    authenticated_client,
    monkeypatch,
):
    def unreadable_pdf(_):
        raise ValueError("invalid PDF structure")

    monkeypatch.setattr(main, "PdfReader", unreadable_pdf)

    response = _upload(authenticated_client)
    assert response.status_code == 400
    documents = authenticated_client.get("/documents").json()["documents"]
    assert len(documents) == 1
    assert documents[0]["processing_status"] == "failed"
    response = authenticated_client.post(
        "/interview/start",
        json={
            "role": "Engineer",
            "interview_type": "technical",
            "document_id": documents[0]["document_id"],
        },
    )
    assert response.status_code == 409


def test_interview_without_selected_document_does_not_search_all_user_docs(
    tmp_path,
    monkeypatch,
):
    from database.session_store import SQLiteSessionStore

    def unexpected_retrieval(*args, **kwargs):
        raise AssertionError("Unselected documents must not be retrieved")

    monkeypatch.setattr(
        "services.interview.retrieve_context",
        unexpected_retrieval,
    )

    class Provider:
        name = "test"

        def generate_question(
            self,
            role,
            interview_type,
            difficulty,
            history,
            context,
            context_mode="general",
        ):
            return "A question without unrelated document context?"

        def evaluate_answer(self, *args, **kwargs):
            raise AssertionError("This test does not evaluate answers")

    service = InterviewService(
        store=SQLiteSessionStore(tmp_path / "sessions.sqlite3"),
        provider=Provider(),
    )
    session = service.start(
        "Engineer",
        "technical",
        "medium",
        1,
        user_id="user-1",
    )

    assert service.next_question(session.session_id)["question"]
