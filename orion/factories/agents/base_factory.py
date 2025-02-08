from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from orion.agents.base_agent import BaseAgent

class AgentFactory(ABC):
    """
    Abstract base class for creating agents on-the-fly.
    """

    @abstractmethod
    def create(
        self,
        task_description: str,
        **kwargs
    ) -> BaseAgent:
        """
        Given a task description and additional kwargs (e.g. context, user preferences),
        produce a fully initialized agent instance:
          - Decide the agent's type
          - Possibly pick the LLM model
          - Generate or select system prompts
          - Assign or generate tools
        """
        pass
