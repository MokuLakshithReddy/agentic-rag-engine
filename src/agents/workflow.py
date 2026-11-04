"""
Agentic RAG State Machine & Workflow Orchestrator.
Coordinates query analysis, planning, tool execution, evidence evaluation,
bounded re-retrieval loops, answer synthesis, and citation verification.
"""

from typing import Dict, Any, List, Optional
import time
from src.state.state import (
    AgentState,
    QueryClassification,
    EvidenceGrading,
    ResponseStatus,
    EvidenceChunk
)
from src.planner.analyzer import QueryAnalyzer
from src.planner.decomposer import QueryDecomposer
from src.planner.planner import RetrievalPlanner
from src.planner.reformulator import QueryReformulator
from src.evaluation.evaluator import EvidenceEvaluator
from src.synthesis.generator import AnswerGenerator
from src.verification.verifier import AnswerVerifier
from src.tools.registry import ToolRegistry
from src.security.sanitizer import SecuritySanitizer
from src.security.guardrails import SecurityGuardrails


class AgenticRAGWorkflow:
    """
    Production-grade stateful Agentic RAG Orchestrator.
    Executes a bounded, observable, multi-step research loop.
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.analyzer = QueryAnalyzer()
        self.decomposer = QueryDecomposer()
        self.planner = RetrievalPlanner()
        self.reformulator = QueryReformulator(max_revisions=2)
        self.evaluator = EvidenceEvaluator()
        self.generator = AnswerGenerator()
        self.verifier = AnswerVerifier()
        self.sanitizer = SecuritySanitizer()
        self.guardrails = SecurityGuardrails()

    def run(self, user_query: str, request_id: Optional[str] = None) -> AgentState:
        """
        Executes the full agentic workflow pipeline.
        """
        state = AgentState(original_query=user_query)
        if request_id:
            state.request_id = request_id

        state.add_trace("START", f"Initialized request {state.request_id} for query: '{user_query}'")

        # 1. SECURITY & INPUT SANITIZATION
        sanitized_q = self.sanitizer.sanitize_query(user_query)
        state.normalized_query = sanitized_q
        state.add_trace("INPUT_SANITIZED", f"Sanitized query length: {len(sanitized_q)}")

        # 2. QUERY ANALYSIS & CLASSIFICATION
        classification, topics = self.analyzer.analyze(sanitized_q)
        state.classification = classification
        state.topics = topics
        state.add_trace(
            "QUERY_ANALYSIS",
            f"Classified as '{classification.value}' with detected topics: {topics}",
            {"classification": classification.value, "topics": topics}
        )

        # Immediate abstention for out-of-scope queries
        if classification == QueryClassification.OUT_OF_SCOPE:
            state.final_answer = "This query is out of scope for the indexed knowledge domain."
            state.response_status = ResponseStatus.ABSTAINED_OUT_OF_SCOPE
            state.add_trace("TERMINATED", "Terminated early: query marked OUT_OF_SCOPE")
            return state

        # 3. QUERY DECOMPOSITION
        subqueries = self.decomposer.decompose(sanitized_q, classification, topics)
        state.subqueries = subqueries
        state.add_trace(
            "DECOMPOSITION",
            f"Generated {len(subqueries)} targeted subqueries.",
            {"subqueries": [sq.query_text for sq in subqueries]}
        )

        # 4. INITIAL RETRIEVAL PLANNING
        selected_tools = self.planner.plan(state)
        state.add_trace("PLANNING", f"Selected initial tools: {selected_tools}", {"tools": selected_tools})

        # 5. RETRIEVAL & EVIDENCE EVALUATION LOOP (Bounded Iteration)
        while state.tool_budget_remaining > 0:
            state.execution_attempt_count += 1
            state.add_trace("EXECUTION_LOOP", f"Starting retrieval iteration {state.execution_attempt_count}")

            # Execute tools for active subqueries
            retrieved_pool: List[EvidenceChunk] = list(state.retrieved_evidence)
            seen_ids = set(state.evidence_ids)

            if "multi_query_retrieval" in state.selected_tools and len(state.subqueries) > 1:
                sq_texts = [sq.query_text for sq in state.subqueries]
                res = self.tool_registry.execute_tool("multi_query_retrieval", subqueries=sq_texts, top_k_per_query=3)
                if res.get("record"):
                    # Multi-query executes N underlying subquery searches -> deduct N calls from budget
                    state.record_tool_call(res["record"], cost=len(sq_texts))
                if res.get("success"):
                    for chunk in res.get("data", []):
                        if chunk.chunk_id not in seen_ids:
                            if not self.guardrails.inspect_evidence_chunk(chunk.content):
                                seen_ids.add(chunk.chunk_id)
                                retrieved_pool.append(chunk)

            else:
                primary_tool = state.selected_tools[0] if state.selected_tools else "hybrid_retrieval"
                for sq in state.subqueries:
                    if state.tool_budget_remaining <= 0:
                        break
                    res = self.tool_registry.execute_tool(primary_tool, query=sq.query_text, top_k=4)
                    if res.get("record"):
                        state.record_tool_call(res["record"], cost=1)
                    if res.get("success"):
                        for chunk in res.get("data", []):
                            if chunk.chunk_id not in seen_ids:
                                if not self.guardrails.inspect_evidence_chunk(chunk.content):
                                    seen_ids.add(chunk.chunk_id)
                                    retrieved_pool.append(chunk)

            # Optional reranking
            if "rerank_evidence" in state.selected_tools and retrieved_pool:
                res_rerank = self.tool_registry.execute_tool(
                    "rerank_evidence", query=state.normalized_query, candidates=retrieved_pool, top_k=6
                )
                if res_rerank.get("record"):
                    state.record_tool_call(res_rerank["record"], cost=1)
                if res_rerank.get("success"):
                    retrieved_pool = res_rerank.get("data", retrieved_pool)

            state.retrieved_evidence = retrieved_pool
            state.evidence_ids = list(seen_ids)

            # Evaluate Evidence
            grade, rel_score, gaps = self.evaluator.evaluate(state)
            state.evidence_grading = grade
            state.evidence_relevance_score = rel_score
            state.missing_information = gaps

            state.add_trace(
                "EVIDENCE_EVALUATED",
                f"Evidence graded as '{grade.value}' (Relevance: {rel_score:.2f}). Gaps: {len(gaps)}",
                {"grade": grade.value, "score": rel_score, "gaps": gaps}
            )

            # Decide whether to reformulate and re-retrieve
            if self.reformulator.can_reformulate(state):
                state.add_trace(
                    "RE_RETRIEVAL_DECISION",
                    f"Evidence is {grade.value}; reformulating query (Revision {state.planning_revision_count + 1}/{state.max_planning_revisions})"
                )
                revised = self.reformulator.reformulate(state)
                state.selected_tools = ["hybrid_retrieval"]
                state.add_trace(
                    "QUERY_REFORMULATED",
                    f"Generated {len(revised)} revised subqueries.",
                    {"revised_subqueries": [r.query_text for r in revised]}
                )
                continue
            else:
                # Evidence is either sufficient or budget is exhausted
                break

        # 6. ANSWER SYNTHESIS
        draft_answer, citations, status = self.generator.generate(state)
        state.synthesis_draft = draft_answer
        state.citations = citations
        state.response_status = status
        state.add_trace(
            "ANSWER_SYNTHESIZED",
            f"Drafted answer with {len(citations)} citations. Status: {status.value}"
        )

        # 7. CITATION & ANSWER VERIFICATION
        final_answer, confidence, final_status = self.verifier.verify(state)
        state.final_answer = final_answer
        state.confidence_score = confidence
        state.response_status = final_status
        state.add_trace(
            "VERIFICATION_COMPLETED",
            f"Verified answer (Confidence: {confidence:.2f}, Status: {final_status.value})",
            {"confidence": confidence, "status": final_status.value, "citations": state.citations}
        )

        return state

