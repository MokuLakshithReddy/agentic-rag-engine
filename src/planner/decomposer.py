"""
Query Decomposer Module.
Breaks multi-document and comparative queries into targeted, discrete subqueries.
"""

from typing import List
import re
from src.state.state import QueryClassification, Subquery


class QueryDecomposer:
    """
    Decomposes queries into subqueries based on intent classification and entity boundaries.
    """

    def decompose(self, query: str, classification: QueryClassification, topics: List[str]) -> List[Subquery]:
        """
        Generates targeted subqueries.
        """
        q = query.strip()
        lower_q = q.lower()

        # If simple factual or ambiguous, maintain single subquery
        if classification in {QueryClassification.SIMPLE_FACTUAL, QueryClassification.AMBIGUOUS, QueryClassification.INSUFFICIENTLY_SPECIFIED, QueryClassification.OUT_OF_SCOPE}:
            return [Subquery(query_text=q, target_entity=topics[0] if topics else None)]

        # If comparative: look for "between X and Y", "X vs Y", "compare X and Y"
        if classification == QueryClassification.COMPARATIVE:
            match = re.search(r"(?:between|compare)\s+(.*?)\s+(?:and|with|versus|vs)\s+(.*?)(?:\?|$|\.|\,)", q, re.IGNORECASE)
            if not match:
                match = re.search(r"(.*?)\s+(?:vs\.?|versus)\s+(.*?)(?:\?|$|\.|\,)", q, re.IGNORECASE)

            if match:
                entity_a = match.group(1).strip()
                entity_b = match.group(2).strip()
                return [
                    Subquery(query_text=f"{entity_a} features overview specifications", target_entity=entity_a),
                    Subquery(query_text=f"{entity_b} features overview specifications", target_entity=entity_b),
                    Subquery(query_text=f"comparison differences trade-offs between {entity_a} and {entity_b}", target_entity=f"{entity_a}_vs_{entity_b}")
                ]
            elif len(topics) >= 2:
                return [
                    Subquery(query_text=f"{topics[0]} specifications and architecture", target_entity=topics[0]),
                    Subquery(query_text=f"{topics[1]} specifications and architecture", target_entity=topics[1]),
                    Subquery(query_text=f"differences between {topics[0]} and {topics[1]}", target_entity=f"{topics[0]}_{topics[1]}")
                ]

        # If multi-hop or multi-document
        if classification in {QueryClassification.MULTI_HOP, QueryClassification.MULTI_DOCUMENT}:
            subqueries = [Subquery(query_text=q)]
            if len(topics) >= 2:
                for t in topics[:3]:
                    subqueries.append(Subquery(query_text=f"{t} details and context", target_entity=t))
            return subqueries

        return [Subquery(query_text=q)]

