"""
Answer Synthesis Generator.
Executes LLM-driven generation using system prompt templates and XML delimiters,
embeds verifiable citations, and enforces strict context grounding.
"""

from typing import List, Dict, Any, Tuple, Optional
import os
import re
from src.state.state import AgentState, EvidenceChunk, EvidenceGrading, ResponseStatus
from src.synthesis.prompt_templates import SYNTHESIS_SYSTEM_PROMPT, SYNTHESIS_USER_TEMPLATE
from src.llm.base import BaseLLMClient, LLMMessage
from src.llm.client import get_llm_client


class AnswerGenerator:
    """
    LLM-powered grounded response synthesizer.
    Feeds untrusted evidence safely within XML delimiters and enforces verifiable citations.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None, max_context_chars: int = 12000):
        self.llm = llm_client or get_llm_client()
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
        Synthesizes grounded answer using the LLM with strict context boundaries.
        Returns (answer_text, cited_chunk_ids, response_status)
        """
        # Strict abstention triggers
        if state.evidence_grading == EvidenceGrading.NO_EVIDENCE or not state.retrieved_evidence:
            abstention_msg = "I do not have sufficient verified evidence in the indexed knowledge base to answer this question."
            return abstention_msg, [], ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE

        if state.evidence_relevance_score < 0.20:
            abstention_msg = "I do not have sufficient verified evidence in the indexed knowledge base to answer this question."
            return abstention_msg, [], ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE

        context = self.format_context(state.retrieved_evidence)
        user_prompt = SYNTHESIS_USER_TEMPLATE.format(query=state.normalized_query, context=context)

        messages = [
            LLMMessage(role="system", content=SYNTHESIS_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt)
        ]

        try:
            resp = self.llm.generate(messages)
            answer_text = resp.content.strip()
        except Exception as e:
            # Safe deterministic fallback
            lines = ["Based on the retrieved documentation:\n"]
            for chunk in state.retrieved_evidence[:3]:
                snippet = chunk.content.strip().replace("\n", " ")
                lines.append(f"{snippet} [{chunk.chunk_id}]")
            answer_text = "\n\n".join(lines)

        # Extract cited chunk IDs
        cited_ids = re.findall(r"\[([a-zA-Z0-9_\-\.]+#[a-zA-Z0-9_\-]+)\]", answer_text)
        citations = list(dict.fromkeys(cited_ids))

        # Check if model chose to abstain in generated text
        if "not have sufficient" in answer_text.lower() or "cannot answer" in answer_text.lower():
            return answer_text, [], ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE

        return answer_text, citations, ResponseStatus.SUCCESS
