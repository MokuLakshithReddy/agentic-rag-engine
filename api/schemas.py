"""
API Request and Response Pydantic Schemas.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class IngestDocumentRequest(BaseModel):
    doc_id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., description="Document content text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata tags")
    page_number: Optional[int] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000, description="User research question")
    request_id: Optional[str] = Field(None, description="Optional custom client request ID")


class QueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    citations: List[str]
    confidence: float
    status: str
    classification: str
    tool_calls_count: int
    execution_time_ms: float
    retrieved_chunk_count: int


class DocumentSummaryResponse(BaseModel):
    doc_id: str
    chunk_count: int
    metadata: Dict[str, Any]
    preview: str


class HealthResponse(BaseModel):
    status: str
    total_documents: int
    total_chunks: int
    version: str

