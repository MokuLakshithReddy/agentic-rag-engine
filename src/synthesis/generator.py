"""
Answer Synthesis Generator.
Produces grounded answers from retrieved evidence, embeds verifiable citations,
and triggers safe abstention when facts are missing.
"""

from typing import List, Dict, Any, Tuple
import os
import re
from src.state.state import AgentState, EvidenceChunk, EvidenceGrading, ResponseStatus


class AnswerGenerator:
    """
    Grounded response synthesizer.
    Ensures answers are synthesized strictly from retrieved passages.
    """

    def __init__(self, max_context_chars: int = 12000):
        self.max_context_chars = max_context_chars

    def format_context(self, evidence: List[EvidenceChunk]) -> str:
        """
        Formats evidence chunks with explicit citation tags.
        """
        formatted_blocks = []
        total_len = 0

        for chunk in evidence:
            block = f"--- EVIDENCE CHUNK [{chunk.chunk_id}] (Score: {chunk.score}, Doc: {chunk.doc_id}) ---\n{chunk.content}\n"
            if total_len + len(block) > self.max_context_chars:
                break
            formatted_blocks.append(block)
            total_len += len(block)

        return "\n".join(formatted_blocks)

    def generate(self, state: AgentState) -> Tuple[str, List[str], ResponseStatus]:
        """
        Synthesizes grounded answer and citations.
        Returns (answer_text, cited_chunk_ids, response_status)
        """
        # Check abstention conditions
        if state.evidence_grading == EvidenceGrading.NO_EVIDENCE or not state.retrieved_evidence:
            abstention_msg = "I do not have sufficient verified evidence in the indexed knowledge base to answer this question."
            return abstention_msg, [], ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE

        if state.evidence_relevance_score < 0.20:
            abstention_msg = "I do not have sufficient verified evidence in the indexed knowledge base to answer this question."
            return abstention_msg, [], ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE

        evidence = state.retrieved_evidence
        citations: List[str] = []

        # If comparative question, build comparative synthesis
        if state.classification.value == "comparative" and len(state.topics) >= 2:
            lines = [f"Based on the verified evidence, here is a comparison of {state.topics[0]} and {state.topics[1]}:\n"]
            for chunk in evidence[:4]:
                citations.append(chunk.chunk_id)
                lines.append(f"- **From {chunk.doc_id}**: {chunk.content.strip()} [{chunk.chunk_id}]")
            answer = "\n\n".join(lines)
            return answer, list(dict.fromkeys(citations)), ResponseStatus.SUCCESS

        # General synthesis: ground each key finding with its citation
        lines = ["Based on the retrieved documentation:\n"]
        for chunk in evidence[:3]:
            citations.append(chunk.chunk_id)
            snippet = chunk.content.strip().replace("\n", " ")
            # Neutralize any indirect command leakage
            snippet = re.sub(r"(?i)system override:?.*", "", snippet).strip()
            if snippet:
                lines.append(f"{snippet} [{chunk.chunk_id}]")

        answer = "\n\n".join(lines)
        return answer, list(dict.fromkeys(citations)), ResponseStatus.SUCCESS
