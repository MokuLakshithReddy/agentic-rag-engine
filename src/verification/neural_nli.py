"""
Neural Natural Language Inference (NLI) Classifier using DeBERTa-v3 Cross-Encoder.
Provides genuine model-based entailment, contradiction, and neutrality classification
offline without external LLM API dependencies.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger("agentic_rag.nli")

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False


class NeuralNLIClassifier:
    """
    Local neural cross-encoder NLI classifier.
    Uses 'cross-encoder/nli-deberta-v3-small' to perform deep semantic inference
    between premises (evidence chunks) and hypotheses (factual claims).
    
    Label Mapping for cross-encoder/nli-deberta-v3-small:
      0: contradiction
      1: entailment
      2: neutral
    """

    _instance: Optional["NeuralNLIClassifier"] = None

    @classmethod
    def get_instance(cls, model_name: str = "cross-encoder/nli-deberta-v3-small") -> "NeuralNLIClassifier":
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        self.model_name = model_name
        self.model = None
        self._is_ready = False
        self._load_model()

    def _load_model(self) -> None:
        if HAS_CROSS_ENCODER:
            try:
                self.model = CrossEncoder(self.model_name)
                self._is_ready = True
                logger.info(f"Loaded neural NLI model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Could not load neural NLI cross-encoder ({e}). Will use fallback.")
                self.model = None
                self._is_ready = False
        else:
            self._is_ready = False

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp_vals = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    def _select_eval_premise(self, premise: str, hypothesis: str) -> str:
        """
        Extracts the most relevant sentence from multi-sentence chunks.
        Prevents full-paragraph context dilution where extra sentences turn specific factual claims neutral.
        """
        import re
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", premise) if s.strip()]
        if len(sentences) <= 1:
            return premise

        stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "in", "of", "and", "or", "for", "with", "that", "it", "its"}
        hypo_tokens = set(re.findall(r"\b\w+\b", hypothesis.lower())) - stopwords

        best_s = sentences[0]
        best_ol = -1
        for s in sentences:
            s_tokens = set(re.findall(r"\b\w+\b", s.lower()))
            ol = len(hypo_tokens.intersection(s_tokens))
            if ol > best_ol:
                best_ol = ol
                best_s = s

        return best_s if best_ol > 0 else premise

    def classify(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Classifies semantic relationship between premise and hypothesis.
        Returns:
            {
                "is_entailed": bool,
                "contradiction": bool,
                "confidence": float,
                "label": str,  # 'entailment', 'contradiction', 'neutral'
                "probabilities": {"contradiction": float, "entailment": float, "neutral": float}
            }
        """
        if not self._is_ready or self.model is None:
            return self._heuristic_fallback(premise, hypothesis)

        try:
            eval_premise = self._select_eval_premise(premise, hypothesis)
            raw_scores = self.model.predict([(eval_premise, hypothesis)])
            logits = np.array(raw_scores[0]) if isinstance(raw_scores, (list, np.ndarray)) else np.array(raw_scores)
            probs = self._softmax(logits)

            p_contra = float(probs[0])
            p_entail = float(probs[1])
            p_neutral = float(probs[2])

            is_contradiction = (p_contra >= 0.45) or (p_contra > p_entail and p_contra > p_neutral)
            is_entailed = (p_entail >= 0.35 and p_entail > p_contra)

            # Assign dominant label and confidence
            if is_contradiction:
                label = "contradiction"
                conf = round(p_contra, 4)
            elif is_entailed:
                label = "entailment"
                conf = round(p_entail, 4)
            else:
                label = "neutral"
                conf = round(p_neutral, 4)

            return {
                "is_entailed": is_entailed and not is_contradiction,
                "contradiction": is_contradiction,
                "confidence": conf,
                "label": label,
                "probabilities": {
                    "contradiction": round(p_contra, 4),
                    "entailment": round(p_entail, 4),
                    "neutral": round(p_neutral, 4)
                }
            }
        except Exception as e:
            logger.warning(f"Neural NLI inference failed: {e}. Falling back.")
            return self._heuristic_fallback(premise, hypothesis)

    def _heuristic_fallback(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """Calibrated fallback when neural weights are unavailable."""
        import re
        premise_lower = premise.lower()
        hypo_lower = hypothesis.lower()

        negations = {"not", "never", "no", "cannot", "does not", "doesn't", "is not", "isn't"}
        hypo_neg = any(re.search(rf"\b{neg}\b", hypo_lower) for neg in negations)

        premise_sentences = re.split(r"(?<=[.!?])\s+", premise_lower)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "in", "of", "and", "or", "for", "with", "that"}
        hypo_words = [w for w in re.findall(r"\b\w+\b", hypo_lower) if len(w) > 2 and w not in stopwords and w not in negations]

        best_sentence = ""
        best_overlap = 0
        for s in premise_sentences:
            s_words = set(re.findall(r"\b\w+\b", s))
            overlap = len(set(hypo_words).intersection(s_words))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = s

        sentence_neg = any(re.search(rf"\b{neg}\b", best_sentence) for neg in negations)
        if best_overlap >= 3 and hypo_neg != sentence_neg:
            return {
                "is_entailed": False,
                "contradiction": True,
                "confidence": 0.95,
                "label": "contradiction",
                "probabilities": {"contradiction": 0.95, "entailment": 0.02, "neutral": 0.03}
            }

        premise_all_words = set(re.findall(r"\b\w+\b", premise_lower))
        supported = sum(1 for w in hypo_words if w in premise_all_words or any(w.rstrip('s') in p for p in premise_all_words))
        ratio = supported / max(1, len(hypo_words))

        is_entailed = ratio >= 0.45
        return {
            "is_entailed": is_entailed,
            "contradiction": False,
            "confidence": round(ratio, 4),
            "label": "entailment" if is_entailed else "neutral",
            "probabilities": {
                "contradiction": 0.05,
                "entailment": round(ratio, 4),
                "neutral": round(1.0 - ratio, 4)
            }
        }
