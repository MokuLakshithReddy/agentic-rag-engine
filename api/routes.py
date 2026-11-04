"""
FastAPI Route Handlers with Authentication, Rate Limiting, Persistence, and Telemetry Redaction.
"""

from typing import List, Dict, Any, Optional
import time
import os
import json
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

# Persistence paths
TRACE_DIR = Path("./data/traces")
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# Authentication & Rate Limiting Config
CONFIGURED_API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Simple IP-based sliding window rate limiter
RATE_LIMIT_WINDOW_SEC = 60
MAX_REQUESTS_PER_WINDOW = 120
request_counts: Dict[str, List[float]] = {}

guardrails = SecurityGuardrails()

# Global singleton runtime instances
kb = KnowledgeBase(storage_dir="./data/indices")
if not kb.raw_documents:
    for item in SAMPLE_KNOWLEDGE_CORPUS:
        kb.index_document(doc_id=item["doc_id"], text=item["text"], metadata={"category": item["category"]})

tool_registry = ToolRegistry(kb)
agent_workflow = AgenticRAGWorkflow(tool_registry)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """Verifies API key if CONFIGURED_API_KEY is set in environment."""
    if CONFIGURED_API_KEY and len(CONFIGURED_API_KEY.strip()) > 0:
        if api_key != CONFIGURED_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key in X-API-Key header."
            )
    return True


def check_rate_limit(request: Request) -> None:
    """Sliding window rate limiter by client IP."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    timestamps = request_counts.get(client_ip, [])
    # Filter timestamps within window
    valid_ts = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SEC]
    if len(valid_ts) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please throttle your requests."
        )
    valid_ts.append(now)
    request_counts[client_ip] = valid_ts


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Returns application health and current index capacity."""
    return HealthResponse(
        status="HEALTHY",
        total_documents=len(kb.raw_documents),
        total_chunks=len(kb.all_chunks),
        version="1.0.0"
    )


@router.post("/v1/documents", response_model=Dict[str, Any])
def ingest_document(req: IngestDocumentRequest, request: Request, authorized: bool = Security(verify_api_key)):
    """Ingests a document and indexes it into Dense, Sparse, and Metadata indices."""
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
def list_documents(request: Request):
    """Lists all indexed documents with metadata and chunk counts."""
    check_rate_limit(request)
    return kb.get_document_summary()


@router.delete("/v1/documents/{document_id}", response_model=Dict[str, Any])
def delete_document(
    document_id: str = FastPath(..., description="ID of document to delete"),
    request: Request = None,
    authorized: bool = Security(verify_api_key)
):
    """Removes a document from the index."""
    if request:
        check_rate_limit(request)
    deleted = kb.remove_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return {"success": True, "deleted_doc_id": document_id}


@router.post("/v1/index", response_model=Dict[str, Any])
def rebuild_index(request: Request, authorized: bool = Security(verify_api_key)):
    """Forces re-indexing of all registered documents."""
    check_rate_limit(request)
    kb._rebuild_indices()
    return {
        "success": True,
        "indexed_documents": len(kb.raw_documents),
        "indexed_chunks": len(kb.all_chunks)
    }


@router.post("/v1/agent/query", response_model=QueryResponse)
def execute_agent_query(req: QueryRequest, request: Request, authorized: bool = Security(verify_api_key)):
    """Executes the autonomous agentic RAG research workflow for a user query."""
    check_rate_limit(request)
    start_time = time.time()
    try:
        state = agent_workflow.run(req.query, request_id=req.request_id)
        elapsed_ms = (time.time() - start_time) * 1000

        # Persist redacted execution trace
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
            "evidence": [c.model_dump() for c in state.retrieved_evidence]
        }
        trace_file = TRACE_DIR / f"{state.request_id}.json"
        with open(trace_file, "w", encoding="utf-8") as f:
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
            detail="An error occurred while executing the research workflow. Please inspect server logs."
        )


@router.get("/v1/agent/trace/{request_id}", response_model=Dict[str, Any])
def get_agent_trace(
    request_id: str = FastPath(..., description="Target request ID"),
    request: Request = None,
    authorized: bool = Security(verify_api_key)
):
    """Retrieves redacted execution trace, tool telemetry, and evidence for a request."""
    if request:
        check_rate_limit(request)
    trace_file = TRACE_DIR / f"{request_id}.json"
    if not trace_file.exists():
        raise HTTPException(status_code=404, detail=f"Trace for request ID '{request_id}' not found.")
    with open(trace_file, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/v1/evaluation", response_model=Dict[str, Any])
def get_evaluation_metrics(request: Request = None):
    """Runs or retrieves benchmark evaluation metrics across RAG architectures."""
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
