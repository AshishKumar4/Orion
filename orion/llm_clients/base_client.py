from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Type, Callable, Iterator
from pydantic import BaseModel
import enum

class LLMMessage(BaseModel):
    """
    A single chat message: role can be 'user', 'system', or 'assistant'.
    """
    role: str
    content: str

class LLMTool(Callable):
    """
    Record of a tool invocation
    """
    
class LLMToolCall(BaseModel):
    """
    A tool invocation.
    """
    name: str
    arguments: Dict[str, Any]
    function: Callable
    id: Optional[str] = None
    model_config = {
        "arbitrary_types_allowed": True
    }

class NonStreamingResult(BaseModel):
    """
    The unified result object from a non streaming `predict(...)` call.
    """
    text: Optional[str]
    tool_calls: List[LLMToolCall] = []
    structured: Optional[Any] = None
    model_config = {
        "arbitrary_types_allowed": True
    }
    
class StreamingResult(BaseModel):
    """
    The unified result object from a streaming `predict(...)` call.
    """
    text: Optional[Iterator[str]]
    tool_calls: Iterator[List[LLMToolCall]]
    structured: Optional[Any] = None
    model_config = {
        "arbitrary_types_allowed": True
    }

PredictResult = Union[NonStreamingResult, StreamingResult]

class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM clients in Orion.
    Subclasses must implement a single `predict(...)` method that:
      - Possibly calls the LLM with function calling
      - Possibly does streaming or non-streaming
      - Possibly outputs structured data (if response_format is provided)
      - Returns a PredictResult
    """
    
    def __init__(
        self, 
        model: str,
    ):
        super().__init__()
        self.model = model

    @abstractmethod
    def predict(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[LLMTool]] = None,
        stream: bool = False,
        response_format: Optional[Union[Type[BaseModel], Type[enum.Enum]]] = None
    ) -> PredictResult:
        """
        Unified API for producing an LLM response.

        :param model: Which LLM model to use.
        :param messages: List of chat messages (context).
        :param stream: Whether to stream internally; this method ultimately returns a single final result.
        :param response_format: A Pydantic model type or an Enum type for structured output.

        :return: A PredictResult which can be either a NonStreamingResult or a StreamingPredictResult.
        """
        pass

    @abstractmethod
    def cancel(self):
        """
        Request to cancel the current run if possible.
        """
        pass