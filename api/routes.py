"""
Production-Hardened FastAPI Routes with Multi-Tenant Isolation & Authentication.
Implements:
1. Genuine User Authentication: Registration, Login, and Cryptographic Password Hashing (PBKDF2).
2. Tenant Data Isolation: Each user organization, account, or guest sandbox has its own partitioned KnowledgeBase.
3. Isolated Guest Sandboxes: Demo users receive sandboxed tenant IDs preventing cross-user interference.
4. Path-traversal proof request ID validation.
5. Sliding-window rate limiting.
6. Comprehensive telemetry redaction across all stored traces.
"""

from typing import List, Dict, Any, Optional
import time
import os
import json
import re
import uuid
import logging
import secrets
from pathlib import Path
from fastapi import APIRouter, HTTPException, Path as FastPath, Request, Security, status
from fastapi.security import APIKeyHeader

from api.schemas import (
    IngestDocumentRequest,
    QueryRequest,
    QueryResponse,
    DocumentSummaryResponse,
    HealthResponse,
    UserRegisterRequest,
    UserLoginRequest,
    AuthTokenResponse,
    ChangePasswordRequest,
    CreateInvitationRequest,
    InvitationResponse,
    InvitationRole
)
from api.auth import UserAuthManager, AuthContext
from api.tenant_manager import TenantKnowledgeBaseManager
from src.security.guardrails import SecurityGuardrails
from src.evaluation.benchmark import BenchmarkRunner

logger = logging.getLogger("agentic_rag.api")
router = APIRouter()

# Secure Storage Directory
TRACE_DIR = Path("./data/traces").resolve()
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# Authentication & Tenant Managers
auth_manager = UserAuthManager()
tenant_manager = TenantKnowledgeBaseManager()
guardrails = SecurityGuardrails()

# Administrative API Key Header & Session Token Header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
session_token_header = APIKeyHeader(name="X-Session-Token", auto_error=False)

# Sliding Window Rate Limiting: 60s window, max 120 calls per IP
RATE_LIMIT_WINDOW_SEC = 60
MAX_REQUESTS_PER_WINDOW = 120
request_counts: Dict[str, List[float]] = {}


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


def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
    session_token: Optional[str] = Security(session_token_header)
) -> AuthContext:
    """
    Unified Authentication and Tenant Context Resolver.
    Protects endpoints against unauthorized access:
    1. Programmatic access: Requires valid X-API-Key matching server environment variable.
       Resolves tenant via optional 'X-Tenant-ID' header (defaulting to 'default').
    2. Authenticated user/browser session: Validates active token issued by /v1/auth/login,
       /v1/auth/register, or /v1/auth/session (isolated guest sandbox).
    """
    # 1. Programmatic API Key authentication
    configured_key = os.getenv("API_KEY")
    if api_key and configured_key:
        if secrets.compare_digest(api_key.strip(), configured_key.strip()):
            tenant_header = request.headers.get("X-Tenant-ID", "default").strip()
            return AuthContext(
                user_id="admin_cli",
                username="admin",
                tenant_id=tenant_header if tenant_header else "default",
                role="admin",
                auth_type="api_key",
                must_change_password=False
            )

    # 2. Browser / Client session authentication
    if session_token:
        sess = auth_manager.validate_session(session_token.strip())
        if sess:
            auth_ctx = AuthContext(
                user_id=sess["user_id"],
                username=sess["username"],
                tenant_id=sess["tenant_id"],
                role=sess["role"],
                auth_type=sess.get("role", "session"),
                must_change_password=sess.get("must_change_password", False)
            )

            # Enforce password replacement before accessing protected endpoints
            if auth_ctx.must_change_password:
                allowed_paths = {"/v1/auth/change-password", "/v1/auth/me"}
                if request.url.path not in allowed_paths:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Password replacement required: Initial bootstrap administrator secret must be updated via /v1/auth/change-password before accessing system."
                    )

            return auth_ctx

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized: Valid X-API-Key or active authenticated X-Session-Token is required."
    )


def sanitize_request_id(req_id: Optional[str]) -> str:
    """Validates request ID against path traversal patterns."""
    if not req_id:
        return str(uuid.uuid4())
    cleaned = req_id.strip()
    if not re.match(r"^[a-zA-Z0-9_\-]{8,64}$", cleaned):
        return str(uuid.uuid4())
    return cleaned


# ==========================================
# Authentication Endpoints
# ==========================================

@router.post("/v1/auth/register", response_model=AuthTokenResponse)
def register_user(req: UserRegisterRequest, request: Request):
    """
    Registers a new user account with isolated tenant knowledge base storage.
    Strictly forbids user-supplied tenant IDs to eliminate tenant hijacking and traversal.
    """
    check_rate_limit(request)
    try:
        user = auth_manager.register_user(
            username=req.username,
            password=req.password,
            invitation_code=req.invitation_code
        )
        sess = auth_manager.create_session(user)
        # Pre-seed tenant knowledge base safely
        tenant_manager.get_kb(user.tenant_id)
        return AuthTokenResponse(
            session_token=sess["session_token"],
            user_id=sess["user_id"],
            tenant_id=sess["tenant_id"],
            role=sess["role"],
            must_change_password=sess["must_change_password"],
            expires_in=sess["expires_in"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"User registration error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Registration failed.")


@router.post("/v1/auth/login", response_model=AuthTokenResponse)
def login_user(req: UserLoginRequest, request: Request):
    """
    Authenticates registered user and issues a tenant-bound session token.
    Flags whether mandatory password replacement is required.
    """
    check_rate_limit(request)
    user = auth_manager.authenticate_user(username=req.username, password=req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password."
        )

    sess = auth_manager.create_session(user)
    return AuthTokenResponse(
        session_token=sess["session_token"],
        user_id=sess["user_id"],
        tenant_id=sess["tenant_id"],
        role=sess["role"],
        must_change_password=sess["must_change_password"],
        expires_in=sess["expires_in"]
    )


@router.post("/v1/auth/change-password", response_model=Dict[str, Any])
def change_password(
    req: ChangePasswordRequest,
    request: Request,
    auth: AuthContext = Security(verify_api_key)
):
    """Changes user password and clears initial bootstrap password lock."""
    check_rate_limit(request)
    try:
        success = auth_manager.change_password(
            user_id=auth.user_id,
            current_password=req.current_password,
            new_password=req.new_password
        )
        if not success:
            raise HTTPException(status_code=400, detail="Password change failed.")
        return {"success": True, "message": "Password changed successfully. Bootstrap lock cleared."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/v1/tenants/invites", response_model=InvitationResponse)
def create_tenant_invitation(
    req: CreateInvitationRequest,
    request: Request,
    auth: AuthContext = Security(verify_api_key)
):
    """Issues an invitation code allowing an authorized user to join the tenant organization."""
    check_rate_limit(request)
    try:
        code = auth_manager.create_invitation(
            tenant_id=auth.tenant_id,
            creator_user_id=auth.user_id,
            role=req.role
        )
        return InvitationResponse(
            invite_code=code,
            tenant_id=auth.tenant_id,
            role=req.role
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/v1/auth/session", response_model=Dict[str, Any])
def create_guest_sandbox_session(request: Request):
    """
    Issues an isolated ephemeral Guest Sandbox session token.
    Allocates a dedicated guest tenant storage directory to ensure public demo users
    cannot overwrite, delete, or inspect other users' research corpora.
    """
    check_rate_limit(request)
    sess = auth_manager.create_guest_sandbox_session()
    tenant_manager.get_kb(sess["tenant_id"])
    return sess


@router.get("/v1/auth/me", response_model=Dict[str, Any])
def get_current_user_profile(auth: AuthContext = Security(verify_api_key)):
    """Returns authenticated user profile and assigned tenant isolation ID."""
    return auth.model_dump()


# ==========================================
# Knowledge Base & Agent Endpoints
# ==========================================

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Public healthcheck probe (reports default baseline stats)."""
    default_kb = tenant_manager.get_kb("default")
    return HealthResponse(
        status="HEALTHY",
        total_documents=len(default_kb.raw_documents),
        total_chunks=len(default_kb.all_chunks),
        version="1.0.0"
    )


@router.post("/v1/documents", response_model=Dict[str, Any])
def ingest_document(
    req: IngestDocumentRequest,
    request: Request,
    auth: AuthContext = Security(verify_api_key)
):
    """Ingests and indexes document into the caller's isolated tenant knowledge base."""
    check_rate_limit(request)
    kb = tenant_manager.get_kb(auth.tenant_id)
    try:
        chunks = kb.index_document(
            doc_id=req.doc_id,
            text=req.text,
            metadata=req.metadata,
            page_number=req.page_number
        )
        return {
            "success": True,
            "tenant_id": auth.tenant_id,
            "doc_id": req.doc_id,
            "chunks_created": len(chunks)
        }
    except Exception as e:
        logger.error(f"Document ingestion error for tenant {auth.tenant_id}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid document payload or ingestion failure.")


@router.get("/v1/documents", response_model=List[DocumentSummaryResponse])
def list_documents(
    request: Request,
    auth: AuthContext = Security(verify_api_key)
):
    """Lists indexed documents belonging to the caller's isolated tenant knowledge base."""
    check_rate_limit(request)
    kb = tenant_manager.get_kb(auth.tenant_id)
    return kb.get_document_summary()


@router.delete("/v1/documents/{document_id}", response_model=Dict[str, Any])
def delete_document(
    document_id: str = FastPath(..., description="ID of document to delete"),
    request: Request = None,
    auth: AuthContext = Security(verify_api_key)
):
    """Removes a document from the caller's isolated tenant knowledge base."""
    if request:
        check_rate_limit(request)
    kb = tenant_manager.get_kb(auth.tenant_id)
    deleted = kb.remove_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found in tenant repository.")
    return {"success": True, "tenant_id": auth.tenant_id, "deleted_doc_id": document_id}


@router.post("/v1/index", response_model=Dict[str, Any])
def rebuild_index(
    request: Request,
    auth: AuthContext = Security(verify_api_key)
):
    """Forces re-indexing of documents in caller's tenant partition."""
    check_rate_limit(request)
    kb = tenant_manager.get_kb(auth.tenant_id)
    kb._rebuild_indices()
    return {
        "success": True,
        "tenant_id": auth.tenant_id,
        "indexed_documents": len(kb.raw_documents),
        "indexed_chunks": len(kb.all_chunks)
    }


@router.post("/v1/agent/query", response_model=QueryResponse)
def execute_agent_query(
    req: QueryRequest,
    request: Request,
    auth: AuthContext = Security(verify_api_key)
):
    """
    Executes autonomous research workflow against the caller's isolated tenant knowledge base.
    Persists redacted telemetry into tenant-partitioned trace storage.
    """
    check_rate_limit(request)
    safe_req_id = sanitize_request_id(req.request_id)
    workflow = tenant_manager.get_workflow(auth.tenant_id)
    start_time = time.time()

    try:
        state = workflow.run(req.query, request_id=safe_req_id)
        elapsed_ms = (time.time() - start_time) * 1000

        # Redact telemetry
        trace_payload = {
            "request_id": state.request_id,
            "tenant_id": auth.tenant_id,
            "user_id": auth.user_id,
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

        # Tenant-isolated trace storage directory
        tenant_trace_dir = (TRACE_DIR / auth.tenant_id).resolve()
        tenant_trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = (tenant_trace_dir / f"{state.request_id}.json").resolve()
        if trace_path.is_relative_to(tenant_trace_dir):
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
        logger.error(f"Workflow execution failure for tenant {auth.tenant_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while executing the research workflow."
        )


@router.get("/v1/agent/trace/{request_id}", response_model=Dict[str, Any])
def get_agent_trace(
    request_id: str = FastPath(..., description="Target request ID"),
    request: Request = None,
    auth: AuthContext = Security(verify_api_key)
):
    """
    Retrieves redacted execution trace from caller's tenant partition.
    Strictly validates path traversal safety and enforces cross-tenant boundary isolation.
    """
    if request:
        check_rate_limit(request)

    if not re.match(r"^[a-zA-Z0-9_\-]{8,64}$", request_id):
        raise HTTPException(status_code=400, detail="Invalid request_id format.")

    # Search caller's tenant directory
    tenant_trace_dir = (TRACE_DIR / auth.tenant_id).resolve()
    trace_file = (tenant_trace_dir / f"{request_id}.json").resolve()

    if not trace_file.is_relative_to(tenant_trace_dir) or not trace_file.exists():
        # If admin, allow searching default directory as fallback
        if auth.role == "admin":
            default_trace = (TRACE_DIR / "default" / f"{request_id}.json").resolve()
            if default_trace.exists():
                with open(default_trace, "r", encoding="utf-8") as f:
                    return json.load(f)
            root_trace = (TRACE_DIR / f"{request_id}.json").resolve()
            if root_trace.exists():
                with open(root_trace, "r", encoding="utf-8") as f:
                    return json.load(f)

        raise HTTPException(status_code=404, detail=f"Trace for request ID '{request_id}' not found.")

    with open(trace_file, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/v1/evaluation", response_model=Dict[str, Any])
def get_evaluation_metrics(
    request: Request = None,
    auth: AuthContext = Security(verify_api_key)
):
    """Runs or retrieves benchmark metrics across the standard corpus."""
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
