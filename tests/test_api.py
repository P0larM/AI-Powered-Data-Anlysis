import pytest
from fastapi.testclient import TestClient

from app.api import server

client = TestClient(server.app)
HEADERS = {"X-User-Id": "tester"}
CSV = ("people.csv", b"Name, Age\nAna, 39\nBo, ?\n", "text/csv")


class FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def fake_db(monkeypatch):
    seen = {"datasets": []}
    monkeypatch.setattr(server, "get_ingest_connection", lambda: FakeConn())
    monkeypatch.setattr(server, "get_query_connection", lambda: FakeConn())
    monkeypatch.setattr(server, "get_user_datasets", lambda conn, uid: seen["datasets"])

    def fake_ingest(conn, df, user_id, table, original_filename=None, if_exists="fail"):
        seen.update(df=df, user_id=user_id, table=table, if_exists=if_exists)
        return {"schema": f"user_{user_id}", "table": table, "rows": len(df), "columns": []}

    monkeypatch.setattr(server, "ingest_dataframe", fake_ingest)
    return seen


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_user_header_is_required_and_validated():
    assert client.get("/datasets").status_code == 422
    assert client.get("/datasets", headers={"X-User-Id": "Bad User!"}).status_code == 400


def test_upload_cleans_and_names_the_table(fake_db):
    response = client.post("/datasets", headers=HEADERS, files={"file": CSV})
    assert response.status_code == 201
    assert fake_db["table"] == "people" and fake_db["user_id"] == "tester"
    assert list(fake_db["df"].columns) == ["name", "age"]
    assert fake_db["df"]["age"].isna().sum() == 1


def test_duplicate_upload_needs_replace(fake_db):
    fake_db["datasets"] = [{"table_name": "people"}]
    assert client.post("/datasets", headers=HEADERS, files={"file": CSV}).status_code == 409
    response = client.post("/datasets", headers=HEADERS, files={"file": CSV}, data={"replace": "true"})
    assert response.status_code == 201 and fake_db["if_exists"] == "replace"


def test_bad_uploads(fake_db, monkeypatch):
    empty = {"file": ("x.csv", b"", "text/csv")}
    assert client.post("/datasets", headers=HEADERS, files=empty).status_code == 400
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 10)
    assert client.post("/datasets", headers=HEADERS, files={"file": CSV}).status_code == 413


def test_ask_requires_datasets_and_a_key(fake_db, monkeypatch):
    body = {"question": "how many rows?"}
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    assert client.post("/ask", headers=HEADERS, json=body).status_code == 500
    monkeypatch.setenv("COHERE_API_KEY", "test")
    assert client.post("/ask", headers=HEADERS, json=body).status_code == 404
    assert client.post("/ask", headers=HEADERS, json={"question": ""}).status_code == 422
    assert client.post("/ask", headers=HEADERS, json={"question": "   "}).status_code == 422


def test_ask_returns_the_assistant_result(fake_db, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test")
    fake_db["datasets"] = [{"table_name": "people"}]
    monkeypatch.setattr(server, "answer_question",
                        lambda q, uid, ds, conn: {"answer": f"echo: {q}", "queries": [], "analyses": [], "api_calls": 1})
    response = client.post("/ask", headers=HEADERS, json={"question": " hi "})
    assert response.status_code == 200 and response.json()["answer"] == "echo: hi"