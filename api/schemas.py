"""
API Request and Response Pydantic Schemas.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field


class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    GUEST = "guest"


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


class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, description="Unique username")
    password: str = Field(..., min_length=6, max_length=128, description="User password")
    invitation_code: Optional[str] = Field(None, description="Optional organization invitation code to join an existing tenant")


class UserLoginRequest(BaseModel):
    username: str = Field(..., description="Registered username")
    password: str = Field(..., description="User password")


class AuthTokenResponse(BaseModel):
    session_token: str
    user_id: str
    tenant_id: str
    role: str
    must_change_password: bool = False
    expires_in: int


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., description="Current account password")
    new_password: str = Field(..., min_length=8, max_length=128, description="New strong password")


class CreateInvitationRequest(BaseModel):
    role: UserRole = Field(default=UserRole.USER, description="Assigned role for joining member ('user' or 'admin')")


class InvitationResponse(BaseModel):
    invite_code: str
    tenant_id: str
    role: UserRole



