"""
Semantic Entailment and Factual Claim-to-Evidence Verifier.
Decomposes responses into factual claims, verifies semantic entailment,
detects polarity contradictions (e.g. 'does not use' vs 'uses'),
and validates factual alignment using LLM and calibrated deterministic safeguards.
"""

from typing import Tuple, List, Dict, Any, Set, Optional
import re
import logging
from pydantic import BaseModel
from src.state.state import AgentState, ResponseStatus, EvidenceChunk
from src.verification.citation_checker import CitationChecker
from src.llm.base import BaseLLMClient, LLMMessage
from src.llm.client import get_llm_client, DeterministicReasoningLLMClient

logger = logging.getLogger("agentic_rag.verifier")


class EntailmentResult(BaseModel):
    is_entailed: bool = True
    contradiction: bool = False
    confidence: float = 0.9
    rationale: str = ""


class AnswerVerifier:
    """
    Validates factual claims against cited evidence passages.
    Prevents polarity inversions, hallucinated citations, and contradictory assertions.
    Provides explicit verification strategy tracking and neural NLI / LLM observability.
    """

    DIRECT_NEGATIONS = {
        "not", "never", "no", "neither", "nor", "cannot", "doesn't", "doesnt",
        "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt", "didn't", "didnt"
    }

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        min_confidence_threshold: float = 0.60,
        min_claim_support_ratio: float = 0.45,
        verification_strategy: str = "auto"
    ):
        self.llm = llm_client or get_llm_client()
        self.min_confidence_threshold = min_confidence_threshold
        self.min_claim_support_ratio = min_claim_support_ratio
        self.verification_strategy = verification_strategy
        self.checker = CitationChecker()

    def _split_into_claims(self, text: str) -> List[Tuple[str, Optional[str]]]:
        """
        Splits answer into individual sentence-level claims and resolves associated citation.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        claims_with_cites: List[Tuple[str, Optional[str]]] = []

        for line in lines:
            if line.startswith("#") or line.lower().startswith("based on"):
                continue

            line_cites = self.checker.extract_citations(line)
            default_cite = line_cites[-1] if line_cites else None

            sentences = re.split(r"(?<=[.!?])\s+", line)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) > 15:
                    s_cites = self.checker.extract_citations(s_clean)
                    cite = s_cites[0] if s_cites else default_cite
                    claims_with_cites.append((s_clean, cite))

        return claims_with_cites

    def _has_polarity_contradiction(self, claim: str, chunk_text: str) -> bool:
        """
        Detects if claim negates an affirmative statement in chunk or vice versa.
        E.g. Claim: 'Raft does not use leader election' vs Chunk: 'Raft uses leader election'.
        Evaluates the specific matching sentence within chunk to prevent cross-sentence false positives.
        """
        claim_lower = claim.lower()
        claim_tokens = set(re.findall(r"\b\w+\b", claim_lower))
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "in", "of", "and", "or"}
        claim_content = claim_tokens - self.DIRECT_NEGATIONS - stopwords
        if len(claim_content) < 2:
            return False

        claim_negated = any(re.search(rf"\b{neg}\b", claim_lower) for neg in self.DIRECT_NEGATIONS)

        chunk_sentences = re.split(r"(?<=[.!?])\s+", chunk_text)
        best_sentence = ""
        best_overlap: Set[str] = set()

        for s in chunk_sentences:
            s_lower = s.lower().strip()
            if not s_lower:
                continue
            s_tokens = set(re.findall(r"\b\w+\b", s_lower))
            s_content = s_tokens - self.DIRECT_NEGATIONS - stopwords
            overlap = claim_content.intersection(s_content)
            if len(overlap) > len(best_overlap):
                best_overlap = overlap
                best_sentence = s_lower

        if len(best_overlap) < 3:
            return False

        sentence_negated = any(re.search(rf"\b{neg}\b", best_sentence) for neg in self.DIRECT_NEGATIONS)
        if claim_negated != sentence_negated:
            return True

        return False

    def _lexical_support(
        self,
        claim_clean: str,
        chunk_text: str,
        cited_id: str
    ) -> Tuple[bool, str, float, bool]:
        """Deterministic lexical overlap assessment fallback."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "in",
            "of", "for", "with", "that", "this", "by", "from", "on", "it", "its"
        }
        claim_words = [w.lower() for w in re.findall(r"\b\w+\b", claim_clean) if len(w) > 2 and w.lower() not in stopwords]
        if not claim_words:
            return True, cited_id, 1.0, False

        chunk_lower = chunk_text.lower()
        supported_words = sum(1 for w in claim_words if w in chunk_lower)
        support_ratio = supported_words / len(claim_words)

        is_supported = support_ratio >= self.min_claim_support_ratio
        return is_supported, cited_id, round(support_ratio, 3), False

    def verify_claim_grounding(
        self,
        claim: str,
        cited_id: Optional[str],
        evidence_map: Dict[str, EvidenceChunk]
    ) -> Tuple[bool, Optional[str], float, bool, str]:
        """
        Verifies if a specific claim is factually entailed by its cited evidence chunk
        using genuine NLI/LLM reasoning, contradiction detection, and calibrated deterministic safeguards.
        Returns: (is_grounded, cited_chunk_id, support_score, is_contradiction, model_used)
        """
        if not cited_id:
            return False, None, 0.0, False, "none"

        if cited_id not in evidence_map:
            return False, cited_id, 0.0, False, "none"

        chunk = evidence_map[cited_id]
        chunk_text = chunk.content
        claim_clean = re.sub(r"\[[a-zA-Z0-9_\-\.]+#[a-zA-Z0-9_\-]+\]", "", claim).strip()

        # 1. Deterministic polarity contradiction guard
        if self._has_polarity_contradiction(claim_clean, chunk_text):
            return False, cited_id, 0.0, True, "polarity_contradiction_guard"

        # 2. Strict Deterministic strategy if requested
        if self.verification_strategy == "deterministic":
            is_supp, cid, score, is_contra = self._lexical_support(claim_clean, chunk_text, cited_id)
            return is_supp, cid, score, is_contra, "deterministic_lexical_fallback"

        # 3. Neural NLI Cross-Encoder (Preferred offline & when strategy is neural_nli or auto with offline LLM)
        use_neural_first = (
            self.verification_strategy == "neural_nli" or
            (self.verification_strategy == "auto" and isinstance(self.llm, DeterministicReasoningLLMClient))
        )

        if use_neural_first:
            from src.verification.neural_nli import NeuralNLIClassifier
            clf = NeuralNLIClassifier.get_instance()
            if clf._is_ready and clf.model is not None:
                nli_res = clf.classify(chunk_text, claim_clean)
                model_name = clf.model_name
                if nli_res["contradiction"]:
                    return False, cited_id, 0.0, True, model_name
                if not nli_res["is_entailed"]:
                    return False, cited_id, round(nli_res["confidence"], 3), False, model_name
                return True, cited_id, round(nli_res["confidence"], 3), False, model_name
            else:
                # Local weights unavailable -> calibrated fallback
                is_supp, cid, score, is_contra = self._lexical_support(claim_clean, chunk_text, cited_id)
                return is_supp, cid, score, is_contra, "deterministic_lexical_fallback"

        # 4. External Cloud LLM structured evaluation (when strategy is 'llm' or 'auto' with real cloud client)
        try:
            nli_messages = [
                LLMMessage(
                    role="system",
                    content=(
                        "You are an impartial natural language inference (NLI) and factual verification auditor. "
                        "Evaluate whether the Claim is entailed by the Premise (Evidence Chunk). "
                        "Return strict JSON matching the EntailmentResult schema with fields:\n"
                        "- is_entailed (bool): True if and only if all factual claims are supported by premise.\n"
                        "- contradiction (bool): True if the claim asserts something contrary to the premise.\n"
                        "- confidence (float): Between 0.0 and 1.0.\n"
                        "- rationale (str): Brief explanation."
                    )
                ),
                LLMMessage(
                    role="user",
                    content=f"Premise (Evidence Chunk):\n\"\"\"{chunk_text}\"\"\"\n\nClaim to Verify:\n\"\"\"{claim_clean}\"\"\""
                )
            ]
            entailment: EntailmentResult = self.llm.generate_structured(nli_messages, EntailmentResult)
            model_name = getattr(self.llm, "model_name", "cloud_llm")
            if entailment.contradiction:
                return False, cited_id, 0.0, True, model_name
            if not entailment.is_entailed:
                return False, cited_id, round(entailment.confidence, 3), False, model_name
            return True, cited_id, round(entailment.confidence, 3), False, model_name
        except Exception as e:
            logger.warning(f"LLM entailment evaluation failed: {e}. Falling back to Neural NLI / Heuristic.")

        # 5. Neural NLI Cross-Encoder Fallback
        try:
            from src.verification.neural_nli import NeuralNLIClassifier
            clf = NeuralNLIClassifier.get_instance()
            if clf._is_ready and clf.model is not None:
                nli_res = clf.classify(chunk_text, claim_clean)
                model_name = clf.model_name
                if nli_res["contradiction"]:
                    return False, cited_id, 0.0, True, model_name
                if not nli_res["is_entailed"]:
                    return False, cited_id, round(nli_res["confidence"], 3), False, model_name
                return True, cited_id, round(nli_res["confidence"], 3), False, model_name
        except Exception:
            pass

        # 6. Final Deterministic lexical fallback
        is_supp, cid, score, is_contra = self._lexical_support(claim_clean, chunk_text, cited_id)
        return is_supp, cid, score, is_contra, "deterministic_lexical_fallback"

    def verify(self, state: AgentState) -> Tuple[str, float, ResponseStatus]:
        """
        Verifies all claims and citations in state.synthesis_draft against state.retrieved_evidence.
        Returns (verified_answer, confidence_score, response_status)
        """
        if state.response_status in {
            ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE,
            ResponseStatus.ABSTAINED_OUT_OF_SCOPE
        }:
            return state.final_answer or state.synthesis_draft or "", 0.0, state.response_status

        draft = state.synthesis_draft or state.final_answer
        if not draft:
            return "No response generated.", 0.0, ResponseStatus.ERROR

        if "not have sufficient verified evidence" in draft.lower() or "out of scope" in draft.lower():
            return draft, 0.0, ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE

        evidence_map: Dict[str, EvidenceChunk] = {c.chunk_id: c for c in state.retrieved_evidence}
        all_cited_ids = self.checker.extract_citations(draft)
        state.citations = all_cited_ids

        # If zero citations provided, reject immediately
        if not all_cited_ids:
            state.unsupported_claims = ["Answer contains factual assertions without supporting citations."]
            return (
                "I do not have sufficient verified evidence in the indexed knowledge base to answer this question.",
                0.0,
                ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE
            )

        claims_with_cites = self._split_into_claims(draft)
        if not claims_with_cites:
            claims_with_cites = [(draft, all_cited_ids[0])]

        supported_count = 0
        unsupported_claims: List[str] = []
        has_critical_contradiction = False
        models_used: List[str] = []

        for claim, cited_id in claims_with_cites:
            is_supported, cid, score, is_contradiction, model_used = self.verify_claim_grounding(claim, cited_id, evidence_map)
            models_used.append(model_used)
            if is_contradiction:
                has_critical_contradiction = True
                unsupported_claims.append(f"Direct contradiction detected in claim: '{claim[:80]}' against chunk {cid}")
            elif is_supported:
                supported_count += 1
            else:
                unsupported_claims.append(f"Unverified claim (Cited: {cid}, Support: {score:.2f}): '{claim[:80]}'")

        state.unsupported_claims = unsupported_claims
        grounding_ratio = supported_count / max(1, len(claims_with_cites))

        # Check citation ID validity
        _, valid_cites, invalid_cites, cit_precision = self.checker.verify_citations(
            all_cited_ids, state.retrieved_evidence
        )

        confidence = (0.50 * grounding_ratio) + (0.30 * state.evidence_relevance_score) + (0.20 * cit_precision)
        confidence = round(min(1.0, max(0.0, confidence)), 4)

        primary_model = max(set(models_used), key=models_used.count) if models_used else "none"
        state.verification_metadata = {
            "strategy": self.verification_strategy,
            "model_used": primary_model,
            "models_used": list(set(models_used)),
            "claims_evaluated": len(claims_with_cites),
            "grounding_ratio": round(grounding_ratio, 4),
            "contradiction_detected": has_critical_contradiction
        }
        logger.info(f"Verification completed using model '{primary_model}' (strategy: {self.verification_strategy}, grounding: {grounding_ratio:.2f})")
        state.add_trace(
            "VERIFICATION_METADATA",
            f"Verification executed via '{primary_model}' ({self.verification_strategy}). Grounding: {grounding_ratio:.2f}",
            {"verification_metadata": state.verification_metadata}
        )

        if has_critical_contradiction or grounding_ratio < 0.40 or confidence < self.min_confidence_threshold or len(invalid_cites) > 0:
            return (
                "I do not have sufficient verified evidence in the indexed knowledge base to answer this question.",
                confidence if not has_critical_contradiction else 0.0,
                ResponseStatus.ABSTAINED_INSUFFICIENT_EVIDENCE
            )

        return draft, confidence, ResponseStatus.SUCCESS
