"""
Base LLM Client Interface and Data Models.
Defines unified contracts for prompt generation, structured outputs, and embeddings.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message text content")


class LLMResponse(BaseModel):
    content: str
    raw_response: Optional[Dict[str, Any]] = None
    model: str = "default"
    usage: Dict[str, int] = Field(default_factory=dict)


class BaseLLMClient(ABC):
    """
    Abstract interface for LLM providers.
    All models (OpenAI, Gemini, Ollama, local fallbacks) inherit from this class.
    """

    def __init__(self, model_name: str = "default", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    @abstractmethod
    def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generates completion text for a list of messages."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        messages: List[LLMMessage],
        response_model: Type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """Generates validated structured output matching a Pydantic schema."""
        pass

