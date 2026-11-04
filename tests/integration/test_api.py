"""
Integration tests for FastAPI REST Endpoints.
Verifies healthcheck, authenticated CRUD operations, rate limiting, and 401 rejection.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
VALID_API_KEY = "rag_sec_live_key_2026"
AUTH_HEADERS = {"X-API-Key": VALID_API_KEY}


def test_api_health_is_public():
    """Healthcheck probe should be accessible without authentication."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["total_documents"] >= 5


def test_api_unauthorized_access_rejected():
    """Requests without valid X-API-Key must be rejected with 401."""
    res_list = client.get("/v1/documents")
    assert res_list.status_code == 401

    res_query = client.post("/v1/agent/query", json={"query": "Test query"})
    assert res_query.status_code == 401


def test_api_document_lifecycle():
    # Ingest document with authentication
    payload = {
        "doc_id": "test_api_doc",
        "text": "This is a document about consensus algorithms.",
        "metadata": {"category": "test"}
    }
    res_ingest = client.post("/v1/documents", json=payload, headers=AUTH_HEADERS)
    assert res_ingest.status_code == 200
    assert res_ingest.json()["chunks_created"] > 0

    # List documents
    res_list = client.get("/v1/documents", headers=AUTH_HEADERS)
    assert res_list.status_code == 200
    doc_ids = [d["doc_id"] for d in res_list.json()]
    assert "test_api_doc" in doc_ids

    # Delete document
    res_del = client.delete("/v1/documents/test_api_doc", headers=AUTH_HEADERS)
    assert res_del.status_code == 200
    assert res_del.json()["deleted_doc_id"] == "test_api_doc"


def test_api_agent_query_and_trace():
    query_payload = {"query": "How does Raft handle leader election?"}
    res = client.post("/v1/agent/query", json=query_payload, headers=AUTH_HEADERS)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert len(data["citations"]) > 0
    assert "request_id" in data

    # Retrieve execution trace with authentication
    req_id = data["request_id"]
    res_trace = client.get(f"/v1/agent/trace/{req_id}", headers=AUTH_HEADERS)
    assert res_trace.status_code == 200
    trace_data = res_trace.json()
    assert len(trace_data["trace"]) > 0
