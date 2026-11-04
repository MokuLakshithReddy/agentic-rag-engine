"""
Integration tests for FastAPI REST Endpoints.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["total_documents"] >= 5


def test_api_document_lifecycle():
    # Ingest document
    payload = {
        "doc_id": "test_api_doc",
        "text": "This is a document about consensus algorithms.",
        "metadata": {"category": "test"}
    }
    res_ingest = client.post("/v1/documents", json=payload)
    assert res_ingest.status_code == 200
    assert res_ingest.json()["chunks_created"] > 0

    # List documents
    res_list = client.get("/v1/documents")
    assert res_list.status_code == 200
    doc_ids = [d["doc_id"] for d in res_list.json()]
    assert "test_api_doc" in doc_ids

    # Delete document
    res_del = client.delete("/v1/documents/test_api_doc")
    assert res_del.status_code == 200
    assert res_del.json()["deleted_doc_id"] == "test_api_doc"


def test_api_agent_query():
    query_payload = {"query": "How does Raft handle leader election?"}
    res = client.post("/v1/agent/query", json=query_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert len(data["citations"]) > 0
    assert "request_id" in data

    # Retrieve execution trace
    req_id = data["request_id"]
    res_trace = client.get(f"/v1/agent/trace/{req_id}")
    assert res_trace.status_code == 200
    trace_data = res_trace.json()
    assert len(trace_data["trace"]) > 0

