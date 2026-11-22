from fastapi.testclient import TestClient

import main


def _create_second_user(client):
    registration = client.post(
        "/auth/register",
        json={
            "full_name": "Second User",
            "email": "second@example.com",
            "password": "another-long-test-password",
        },
    )
    assert registration.status_code == 201
    login = client.post(
        "/auth/login",
        json={
            "email": "second@example.com",
            "password": "another-long-test-password",
        },
    )
    assert login.status_code == 200
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_interview_routes_require_authentication():
    client = TestClient(main.app)
    responses = [
        client.post(
            "/documents/upload",
            files={"file": ("resume.pdf", b"pdf", "application/pdf")},
        ),
        client.get("/interviews"),
        client.get("/documents/search", params={"query": "resume"}),
        client.get("/documents"),
        client.get("/documents/private-document"),
        client.delete("/documents/private-document"),
        client.get("/ask", params={"query": "resume"}),
        client.post(
            "/interview/start",
            json={"role": "Engineer", "interview_type": "technical"},
        ),
        client.post("/interview/question", json={"session_id": "unknown"}),
        client.post(
            "/interview/evaluate",
            json={
                "session_id": "unknown",
                "question": "Question",
                "answer": "Answer",
            },
        ),
        client.get("/interview/unknown"),
        client.get("/interview/unknown/report"),
    ]
    assert [response.status_code for response in responses] == [401] * len(
        responses
    )


def test_session_and_history_are_private_to_the_owner(authenticated_client):
    started = authenticated_client.post(
        "/interview/start",
        json={
            "role": "Engineer",
            "interview_type": "technical",
            "number_of_questions": 1,
        },
    )
    assert started.status_code == 200
    session_id = started.json()["session_id"]
    second_user_headers = _create_second_user(authenticated_client)

    outsider_requests = [
        authenticated_client.get(
            f"/interview/{session_id}",
            headers=second_user_headers,
        ),
        authenticated_client.get(
            f"/interview/{session_id}/report",
            headers=second_user_headers,
        ),
        authenticated_client.post(
            "/interview/question",
            json={"session_id": session_id},
            headers=second_user_headers,
        ),
        authenticated_client.post(
            "/interview/evaluate",
            json={
                "session_id": session_id,
                "question": "unknown",
                "answer": "unknown",
            },
            headers=second_user_headers,
        ),
    ]
    assert [response.status_code for response in outsider_requests] == [404] * 4
    assert authenticated_client.get(
        "/interviews",
        headers=second_user_headers,
    ).json()["sessions"] == []
    assert authenticated_client.get("/interviews").json()["sessions"][0][
        "session_id"
    ] == session_id


def test_resume_search_and_interview_creation_check_document_owner(
    authenticated_client, monkeypatch
):
    current_user = authenticated_client.get("/auth/me").json()["user"]
    document_id = "private-resume-id"
    main.document_store.register(
        document_id,
        current_user["user_id"],
        "resume.pdf",
    )
    searched = []
    monkeypatch.setattr(
        main,
        "retrieve_context",
        lambda *args, **kwargs: searched.append(kwargs) or [],
    )

    own_search = authenticated_client.get(
        "/documents/search",
        params={"query": "experience", "document_id": document_id},
    )
    assert own_search.status_code == 200
    assert searched[-1] == {
        "document_id": document_id,
        "user_id": current_user["user_id"],
        "document_type": None,
    }

    second_user_headers = _create_second_user(authenticated_client)
    foreign_search = authenticated_client.get(
        "/documents/search",
        params={"query": "experience", "document_id": document_id},
        headers=second_user_headers,
    )
    foreign_ask = authenticated_client.get(
        "/ask",
        params={"query": "experience", "document_id": document_id},
        headers=second_user_headers,
    )
    foreign_interview = authenticated_client.post(
        "/interview/start",
        json={
            "role": "Engineer",
            "interview_type": "technical",
            "document_id": document_id,
        },
        headers=second_user_headers,
    )
    assert [response.status_code for response in (
        foreign_search,
        foreign_ask,
        foreign_interview,
    )] == [404, 404, 404]


def test_resume_upload_records_owner_and_adds_owner_to_vector_metadata(
    authenticated_client, monkeypatch
):
    captured = {}

    class Page:
        def extract_text(self):
            return "Python developer with interview experience."

    class Reader:
        pages = [Page()]

    def fake_add_chunks(
        chunks,
        filename,
        user_id=None,
        document_type="resume",
        document_id=None,
    ):
        captured["chunks"] = chunks
        captured["filename"] = filename
        captured["user_id"] = user_id
        captured["document_type"] = document_type
        captured["document_id"] = document_id
        return {"document_id": document_id, "chunks_stored": len(chunks)}

    monkeypatch.setattr(main, "add_chunks", fake_add_chunks)
    monkeypatch.setattr(main, "PdfReader", lambda _: Reader())

    response = authenticated_client.post(
        "/documents/upload",
        files={
            "file": (
                "resume.pdf",
                b"%PDF-1.7\nfake pdf",
                "application/pdf",
            )
        },
    )
    user_id = authenticated_client.get("/auth/me").json()["user"]["user_id"]
    assert response.status_code == 200
    document_id = response.json()["document_id"]
    assert captured["filename"] == "resume.pdf"
    assert captured["user_id"] == user_id
    assert captured["document_type"] == "resume"
    assert captured["document_id"] == document_id
    assert captured["chunks"]
    stored = main.document_store.get_for_user(document_id, user_id)
    assert stored is not None
    assert stored["document_type"] == "resume"
    assert stored["processing_status"] == "ready"
