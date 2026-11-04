"""
FastAPI Route Handlers for Agentic RAG Engine.
"""

from typing import List, Dict, Any
import time
from fastapi import APIRouter, HTTPException, Path, Query
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

router = APIRouter()

# Global singleton runtime instances
kb = KnowledgeBase()
# Preload initial sample knowledge corpus
for item in SAMPLE_KNOWLEDGE_CORPUS:
    kb.index_document(doc_id=item["doc_id"], text=item["text"], metadata={"category": item["category"]})

tool_registry = ToolRegistry(kb)
agent_workflow = AgenticRAGWorkflow(tool_registry)

# In-memory execution trace storage
trace_store: Dict[str, Dict[str, Any]] = {}


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
def ingest_document(req: IngestDocumentRequest):
    """Ingests a document and indexes it into Dense, Sparse, and Metadata indices."""
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
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/v1/documents", response_model=List[DocumentSummaryResponse])
def list_documents():
    """Lists all indexed documents with metadata and chunk counts."""
    return kb.get_document_summary()


@router.delete("/v1/documents/{document_id}", response_model=Dict[str, Any])
def delete_document(document_id: str = Path(..., description="ID of document to delete")):
    """Removes a document from the index."""
    deleted = kb.remove_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return {"success": True, "deleted_doc_id": document_id}


@router.post("/v1/index", response_model=Dict[str, Any])
def rebuild_index():
    """Forces re-indexing of all registered documents."""
    kb._rebuild_indices()
    return {
        "success": True,
        "indexed_documents": len(kb.raw_documents),
        "indexed_chunks": len(kb.all_chunks)
    }


@router.post("/v1/agent/query", response_model=QueryResponse)
def execute_agent_query(req: QueryRequest):
    """Executes the autonomous agentic RAG research workflow for a user query."""
    start_time = time.time()
    try:
        state = agent_workflow.run(req.query, request_id=req.request_id)
        elapsed_ms = (time.time() - start_time) * 1000

        # Store trace for auditability
        trace_store[state.request_id] = {
            "request_id": state.request_id,
            "query": state.original_query,
            "status": state.response_status.value,
            "confidence": state.confidence_score,
            "trace": [t.model_dump() for t in state.trace],
            "tool_history": [tc.model_dump() for tc in state.tool_call_history],
            "evidence": [c.model_dump() for c in state.retrieved_evidence]
        }

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
        raise HTTPException(status_code=500, detail=f"Workflow execution failure: {str(e)}")


@router.get("/v1/agent/trace/{request_id}", response_model=Dict[str, Any])
def get_agent_trace(request_id: str = Path(..., description="Target request ID")):
    """Retrieves full execution trace, tool telemetry, and evidence for a request."""
    if request_id not in trace_store:
        raise HTTPException(status_code=404, detail=f"Trace for request ID '{request_id}' not found.")
    return trace_store[request_id]


@router.get("/v1/evaluation", response_model=Dict[str, Any])
def get_evaluation_metrics():
    """Runs or retrieves benchmark evaluation metrics across RAG architectures."""
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
        raise HTTPException(status_code=500, detail=f"Benchmark execution error: {str(e)}")

