from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Optional, List, Union, Any, Dict, Callable, Iterator
import threading

from orion.llm_clients.base_client import BaseLLMClient, LLMMessage, PredictResult
from orion.utils.concurrency import BackgroundStream

class AgentState(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"

class MessageStatus(str, Enum):
    COMPLETED = "completed"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ConversationItem:
    """
    Represents one message in the conversation with a status field.
    """
    role: str
    content: str
    status: MessageStatus = MessageStatus.COMPLETED

class BaseAgent(ABC):
    """
    Abstract base class for an Orion Agent.

    An Agent:
      - Has a name, role, a description prompt, a model_name + API key
      - Manages a conversation history (list of ConversationItem)
      - Has concurrency via an Executor
      - Has a cancel mechanism: `_cancel_requested`
      - Maintains an AgentState: IDLE or PROCESSING
      - All interactions are streaming (this design).
    """

    def __init__(
        self,
        name: str,
        role: str,
        description: str,
        model_name: str,
        api_key: str,
        api_url: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        executor: Optional[Executor] = None,
    ):
        self.name = name
        self.role = role
        self.description = description
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url
        self.tools = tools or []

        # Concurrency
        self.executor = executor or ThreadPoolExecutor(max_workers=32)

        self.llm_client: Optional[BaseLLMClient] = None

        # The conversation starts with a system message if description is provided
        self.conversation_history: List[ConversationItem] = []
        if description:
            self.conversation_history.append(
                ConversationItem(role="system", content=description, status=MessageStatus.COMPLETED)
            )

        # Agent state
        self.state: AgentState = AgentState.IDLE
        self.lock = threading.Lock()

        # Cancelation flag
        self._cancel_requested = False

    def cancel(self):
        """
        Request to cancel the current run (either chat or do).
        """
        with self.lock:
            self._cancel_requested = True

    def chat(self, user_message: str, background: bool = False, **kwargs) -> Iterator[Any]:
        """
        A streaming conversation method. Tools are optional, so we'll pass call_tools=False
        (or let `_perform_interaction` decide). 
        We do NOT provide a non-streaming path here.
        """
        if background:
            return self._run_in_background(self._perform_interaction,
                user_message=user_message,
                force_tools=False,
                is_task=False,
                **kwargs
            )
        else:
            return self._perform_interaction(
                user_message=user_message,
                force_tools=False,
                is_task=False,
                **kwargs
            )

    def do(self, task: str, background: bool = False, stateless: bool = False, **kwargs) -> Iterator[Any]:
        """
        A streaming 'do' method. Tools are forcibly allowed. 
        We do NOT provide a non-streaming path.
        """
        if background:
            return self._run_in_background(self._perform_interaction,
                user_message=task,
                force_tools=True,
                stateless=stateless,
                **kwargs
            )
        else:
            return self._perform_interaction(
                user_message=task,
                force_tools=True,
                stateless=stateless,
                **kwargs
            )

    @abstractmethod
    def _perform_interaction(
        self,
        user_message: str,
        force_tools: bool,
        is_task: bool,
        **kwargs
    ) -> Iterator[Any]:
        """
        Subclasses must produce a streaming response (an iterator of text),
        check for cancellation, etc.
        """
        pass

    def _run_in_background(self, func: Callable, *args, **kwargs) -> Iterator[Any]:
        """
        Schedules the agent's operation on an Executor's worker thread by
        actively consuming the generator in that thread. Returns a
        BackgroundStream for iteration.

        :param func: A function that returns the generator (our _perform_interaction).
        :returns: A BackgroundStream object (iterable) that yields items from the generator in real-time.
        """
        # return self.executor.submit(func, *args, **kwargs)
        def generator_fn() -> Iterator[Any]:
            return func(*args, **kwargs)

        return BackgroundStream(generator_fn, self.executor)
