"""
Production-Hardened FastAPI Routes.
Implements mandatory API authentication, path-traversal proof request ID validation,
sliding-window rate limiting, and comprehensive telemetry redaction across all stored traces.
"""

from typing import List, Dict, Any, Optional
import time
import os
import json
import re
import uuid
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, Path as FastPath, Request, Security, status
from fastapi.security import APIKeyHeader

from api.schemas import (
    IngestDocumentRequest,
    QueryRequest,
    QueryResponse,
    DocumentSummaryResponse,
    HealthResponse
)
from src.ingestion.indexer import KnowledgeBase
from src.tools.registry import ToolRegistry
from src.agents.workflow import AgenticRAGWorkflow
from src.evaluation.dataset import SAMPLE_KNOWLEDGE_CORPUS
from src.evaluation.benchmark import BenchmarkRunner
from src.security.guardrails import SecurityGuardrails

logger = logging.getLogger("agentic_rag.api")
router = APIRouter()

# Secure Storage Directory
TRACE_DIR = Path("./data/traces").resolve()
TRACE_DIR.mkdir(parents=True, exist_ok=True)

import secrets

# Authentication & Session Management
# Server-side administrative API key configured strictly via environment variable
CONFIGURED_API_KEY: Optional[str] = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
session_token_header = APIKeyHeader(name="X-Session-Token", auto_error=False)

# Ephemeral in-memory session registry with TTL for web browser UI clients
ACTIVE_SESSIONS: Dict[str, float] = {}
SESSION_EXPIRATION_SEC = 3600  # 1 hour


def clean_expired_sessions() -> None:
    now = time.time()
    expired = [tok for tok, exp in ACTIVE_SESSIONS.items() if now > exp]
    for tok in expired:
        ACTIVE_SESSIONS.pop(tok, None)


def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
    session_token: Optional[str] = Security(session_token_header)
) -> bool:
    """
    Unified Authentication Validator.
    Protects endpoints against unauthorized access:
    1. Programmatic access: Requires valid X-API-Key matching server environment variable.
    2. Browser clients: Requires active, unexpired X-Session-Token issued by /v1/auth/session.
    Rejects any unauthenticated requests with HTTP 401.
    """
    # 1. Validate programmatic API Key if configured on server
    configured_key = os.getenv("API_KEY") or CONFIGURED_API_KEY
    if api_key and configured_key:
        if secrets.compare_digest(api_key.strip(), configured_key.strip()):
            return True

    # 2. Validate browser session token
    if session_token:
        clean_expired_sessions()
        exp = ACTIVE_SESSIONS.get(session_token.strip())
        if exp and time.time() < exp:
            return True

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized: Valid X-API-Key or active X-Session-Token is required."
    )


# Sliding Window Rate Limiting: 60s window, max 120 calls per IP
RATE_LIMIT_WINDOW_SEC = 60
MAX_REQUESTS_PER_WINDOW = 120
request_counts: Dict[str, List[float]] = {}

guardrails = SecurityGuardrails()

# Global runtime indexer with persistence
kb = KnowledgeBase(storage_dir="./data/indices")
if not kb.raw_documents:
    for item in SAMPLE_KNOWLEDGE_CORPUS:
        kb.index_document(doc_id=item["doc_id"], text=item["text"], metadata={"category": item["category"]})

tool_registry = ToolRegistry(kb)
agent_workflow = AgenticRAGWorkflow(tool_registry)


@router.post("/v1/auth/session", response_model=Dict[str, Any])
def create_browser_session(request: Request):
    """
    Creates an ephemeral, rate-limited browser session token.
    Allows frontend clients to interact with the research API safely without embedding hardcoded API keys.
    """
    check_rate_limit(request)
    clean_expired_sessions()
    session_token = secrets.token_urlsafe(32)
    ACTIVE_SESSIONS[session_token] = time.time() + SESSION_EXPIRATION_SEC
    return {
        "session_token": session_token,
        "expires_in": SESSION_EXPIRATION_SEC
    }


def check_rate_limit(request: Request) -> None:
    """Sliding window rate limiter by client IP."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    timestamps = request_counts.get(client_ip, [])
    valid_ts = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SEC]
    if len(valid_ts) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 120 requests per minute."
        )
    valid_ts.append(now)
    request_counts[client_ip] = valid_ts


def sanitize_request_id(req_id: Optional[str]) -> str:
    """
    Validates request ID against path traversal patterns.
    Allows only alphanumeric, hyphens, and underscores.
    """
    if not req_id:
        return str(uuid.uuid4())
    cleaned = req_id.strip()
    if not re.match(r"^[a-zA-Z0-9_\-]{8,64}$", cleaned):
        return str(uuid.uuid4())
    return cleaned


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Public healthcheck probe (does not expose sensitive data)."""
    return HealthResponse(
        status="HEALTHY",
        total_documents=len(kb.raw_documents),
        total_chunks=len(kb.all_chunks),
        version="1.0.0"
    )


@router.post("/v1/documents", response_model=Dict[str, Any])
def ingest_document(
    req: IngestDocumentRequest,
    request: Request,
    authorized: bool = Security(verify_api_key)
):
    """Ingests and indexes document under thread-safety and authentication."""
    check_rate_limit(request)
    try:
        chunks = kb.index_document(
            doc_id=req.doc_id,
            text=req.text,
            metadata=req.metadata,
            page_number=req.page_number
        )
        return {
            "success": True,
            "doc_id": req.doc_id,
            "chunks_created": len(chunks)
        }
    except Exception as e:
        logger.error(f"Document ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid document payload or ingestion failure.")


@router.get("/v1/documents", response_model=List[DocumentSummaryResponse])
def list_documents(
    request: Request,
    authorized: bool = Security(verify_api_key)
):
    """Lists indexed documents. Requires authentication."""
    check_rate_limit(request)
    return kb.get_document_summary()


@router.delete("/v1/documents/{document_id}", response_model=Dict[str, Any])
def delete_document(
    document_id: str = FastPath(..., description="ID of document to delete"),
    request: Request = None,
    authorized: bool = Security(verify_api_key)
):
    """Removes a document from index. Requires authentication."""
    if request:
        check_rate_limit(request)
    deleted = kb.remove_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return {"success": True, "deleted_doc_id": document_id}


@router.post("/v1/index", response_model=Dict[str, Any])
def rebuild_index(
    request: Request,
    authorized: bool = Security(verify_api_key)
):
    """Forces re-indexing of all registered documents. Requires authentication."""
    check_rate_limit(request)
    kb._rebuild_indices()
    return {
        "success": True,
        "indexed_documents": len(kb.raw_documents),
        "indexed_chunks": len(kb.all_chunks)
    }


@router.post("/v1/agent/query", response_model=QueryResponse)
def execute_agent_query(
    req: QueryRequest,
    request: Request,
    authorized: bool = Security(verify_api_key)
):
    """Executes autonomous agent research workflow with redacted persistence."""
    check_rate_limit(request)
    safe_req_id = sanitize_request_id(req.request_id)
    start_time = time.time()

    try:
        state = agent_workflow.run(req.query, request_id=safe_req_id)
        elapsed_ms = (time.time() - start_time) * 1000

        # Fully redact telemetry (query, trace messages, evidence content)
        trace_payload = {
            "request_id": state.request_id,
            "query": guardrails.redact_telemetry(state.original_query),
            "status": state.response_status.value,
            "confidence": state.confidence_score,
            "trace": [
                {**t.model_dump(), "message": guardrails.redact_telemetry(t.message)}
                for t in state.trace
            ],
            "tool_history": [tc.model_dump() for tc in state.tool_call_history],
            "evidence": [
                {**c.model_dump(), "content": guardrails.redact_telemetry(c.content)}
                for c in state.retrieved_evidence
            ]
        }

        # Safe canonical path resolution
        trace_path = (TRACE_DIR / f"{state.request_id}.json").resolve()
        if trace_path.parent == TRACE_DIR:
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace_payload, f)

        return QueryResponse(
            request_id=state.request_id,
            query=state.original_query,
            answer=state.final_answer,
            citations=state.citations,
            confidence=state.confidence_score,
            status=state.response_status.value,
            classification=state.classification.value,
            tool_calls_count=len(state.tool_call_history),
            execution_time_ms=round(elapsed_ms, 2),
            retrieved_chunk_count=len(state.retrieved_evidence)
        )
    except Exception as e:
        logger.error(f"Workflow execution failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while executing the research workflow."
        )


@router.get("/v1/agent/trace/{request_id}", response_model=Dict[str, Any])
def get_agent_trace(
    request_id: str = FastPath(..., description="Target request ID"),
    request: Request = None,
    authorized: bool = Security(verify_api_key)
):
    """Retrieves redacted execution trace. Requires authentication and validates path safety."""
    if request:
        check_rate_limit(request)

    # Strictly reject path traversal in request_id
    if not re.match(r"^[a-zA-Z0-9_\-]{8,64}$", request_id):
        raise HTTPException(status_code=400, detail="Invalid request_id format.")

    trace_file = (TRACE_DIR / f"{request_id}.json").resolve()
    if trace_file.parent != TRACE_DIR or not trace_file.exists():
        raise HTTPException(status_code=404, detail=f"Trace for request ID '{request_id}' not found.")

    with open(trace_file, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/v1/evaluation", response_model=Dict[str, Any])
def get_evaluation_metrics(
    request: Request = None,
    authorized: bool = Security(verify_api_key)
):
    """Runs or retrieves benchmark metrics. Requires authentication."""
    if request:
        check_rate_limit(request)
    try:
        runner = BenchmarkRunner()
        results = runner.execute_all()
        return {
            "success": True,
            "baseline_csv": results["baseline_csv"],
            "agent_csv": results["agent_csv"],
            "metrics_summary": results["summary"]
        }
    except Exception as e:
        logger.error(f"Benchmark execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to run benchmark suite.")
