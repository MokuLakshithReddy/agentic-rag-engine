"""
LLM Interface Package.
Provides standardized abstract client contracts, structured message formats,
and multi-provider integrations.
"""

from src.llm.base import BaseLLMClient, LLMMessage, LLMResponse

__all__ = ["BaseLLMClient", "LLMMessage", "LLMResponse"]
