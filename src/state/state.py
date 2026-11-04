"""
Typed Agent State Model.
Implements validated Pydantic structures for explicit state transitions,
trace observability, budget tracking, and citation records.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
import time


class QueryClassification(str, Enum):
    SIMPLE_FACTUAL = "simple_factual"
    MULTI_DOCUMENT = "multi_document"
    COMPARATIVE = "comparative"
    MULTI_HOP = "multi_hop"
    AMBIGUOUS = "ambiguous"
    OUT_OF_SCOPE = "out_of_scope"
    INSUFFICIENTLY_SPECIFIED = "insufficiently_specified"


class EvidenceGrading(str, Enum):
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    INSUFFICIENT = "insufficient"
    CONTRADICTORY = "contradictory"
    NO_EVIDENCE = "no_evidence"


class ResponseStatus(str, Enum):
    SUCCESS = "SUCCESS"
    ABSTAINED_INSUFFICIENT_EVIDENCE = "ABSTAINED_INSUFFICIENT_EVIDENCE"
    ABSTAINED_OUT_OF_SCOPE = "ABSTAINED_OUT_OF_SCOPE"
    ERROR = "ERROR"


class Subquery(BaseModel):
    subquery_id: str = Field(default_factory=lambda: f"sq_{uuid.uuid4().hex[:6]}")
    query_text: str
    target_entity: Optional[str] = None
    assigned_tool: str = "hybrid_retrieval"
    resolved: bool = False


class EvidenceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    score: float = 0.0
    strategy: str = "dense_retrieval"
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    call_id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:6]}")
    tool_name: str
    query_input: str
    timestamp: float = Field(default_factory=time.time)
    execution_time_ms: float = 0.0
    success: bool = True
    results_count: int = 0
    error_message: Optional[str] = None


class TraceStep(BaseModel):
    step_index: int
    stage: str
    timestamp: float = Field(default_factory=time.time)
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """
    Central validated state object for the Agentic RAG state machine.
    No hidden chain-of-thought is preserved; all transitions are inspectable.
    """
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str
    normalized_query: str = ""
    classification: QueryClassification = QueryClassification.SIMPLE_FACTUAL
    topics: List[str] = Field(default_factory=list)
    subqueries: List[Subquery] = Field(default_factory=list)
    selected_tools: List[str] = Field(default_factory=list)
    
    # Retrieved & graded evidence
    retrieved_evidence: List[EvidenceChunk] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    evidence_grading: EvidenceGrading = EvidenceGrading.NO_EVIDENCE
    evidence_relevance_score: float = 0.0
    missing_information: List[str] = Field(default_factory=list)
    
    # Execution budgets & iteration control
    execution_attempt_count: int = 0
    planning_revision_count: int = 0
    max_planning_revisions: int = 2
    tool_budget_remaining: int = 6
    tool_call_history: List[ToolCallRecord] = Field(default_factory=list)
    
    # Synthesis & Verification
    synthesis_draft: Optional[str] = None
    final_answer: str = ""
    citations: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    response_status: ResponseStatus = ResponseStatus.SUCCESS
    
    # Observability & Trace Telemetry
    trace: List[TraceStep] = Field(default_factory=list)
    error: Optional[str] = None

    def add_trace(self, stage: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        step = TraceStep(
            step_index=len(self.trace) + 1,
            stage=stage,
            message=message,
            metadata=metadata or {}
        )
        self.trace.append(step)

    def record_tool_call(self, record: ToolCallRecord, cost: int = 1) -> None:
        self.tool_call_history.append(record)
        self.tool_budget_remaining = max(0, self.tool_budget_remaining - cost)

