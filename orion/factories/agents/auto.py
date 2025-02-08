import json
from orion.utils import logger
import re
from typing import Optional, Dict, Any, List, Callable

from pydantic import BaseModel, Field, ValidationError

from orion.llm_clients import BaseLLMClient, LLMMessage, OpenAIClient
from orion.agents.base_agent import BaseAgent
from orion.agents.normal_agent import NormalAgent
from .base_factory import AgentFactory
from orion.factories.clients import AutoClientFactory, DEFAULT_CLIENT_FACTORY

# ----------------------------------------------------------------------
# 1. Define a Data Model for the Meta LLM’s Structured Output
# ----------------------------------------------------------------------

class MetaOutputToolSpec(BaseModel):
    """Definition for a single tool that the meta LLM says we should create."""
    name: str
    docstring: str
    code: str  # The Python code that implements the function

class MetaOutput(BaseModel):
    """Definition for the structured plan from the meta LLM."""
    agent_class: str = Field(..., description="Name of the agent class to create.")
    model: str = Field(..., description="LLM model name to use for the new agent.")
    system_prompt: str = Field(..., description="System prompt for the new agent.")
    tools: List[MetaOutputToolSpec] = Field(
        default_factory=list,
        description="List of custom tools to generate."
    )

# ----------------------------------------------------------------------
# 2. AutoAgentFactory Implementation Using Structured Outputs
# ----------------------------------------------------------------------

class AutoAgentFactory(AgentFactory):
    """
    A specialized factory that uses a meta LLM to analyze a task description and decide
    how to create a new agent. It determines:
    
      - Which Agent class to use (e.g. NormalAgent, ManagerAgent, etc.)
      - Which LLM model the agent should use
      - The system prompt (instructions) for the agent
      - Which tools (if any) the agent should be allowed to use
      
    If the meta LLM fails to provide valid structured output, the factory falls back to defaults.
    """

    def __init__(
        self,
        client_factory: AutoClientFactory = DEFAULT_CLIENT_FACTORY,
        default_agent_class: Callable[..., BaseAgent] = NormalAgent,
        meta_model: str = "gpt-4o",
        default_model: str = "gpt-4o",
        registry: Optional[Dict[str, BaseAgent]] = None,
        tool_registry: Optional[Dict[str, Callable]] = None,
    ):
        """
        :param meta_llm: An LLM client used to reason about tasks and produce an agent creation plan.
        :param default_agent_class: The fallback Agent class if the meta LLM’s output is invalid or missing info.
        :param default_model: The default LLM model to use if none is specified.
        :param registry: Optional dictionary to store references to created agents by name.
        :param tool_registry: Optional dictionary to store or retrieve dynamically generated tool functions by name.
        """
        self.client_factory = client_factory
        meta_llm = OpenAIClient(meta_model)
        self.meta_llm = meta_llm
        self.default_agent_class = default_agent_class
        self.default_model = default_model
        self.agent_registry = registry if registry else {}
        self.tool_registry = tool_registry if tool_registry else {}

    def create(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> BaseAgent:
        """
        1. Prompts the meta LLM with the task description.
        2. Expects structured JSON output (using the MetaOutput schema) describing:
           - agent_class
           - model
           - system_prompt
           - tools
        3. If parsing fails, falls back to default settings.
        4. Optionally generates new Python tool functions from provided code.
        5. Instantiates the agent, attaches the tools, and registers the agent.
        """
        if context is None:
            context = {}

        # 1. Compose a detailed meta prompt.
        meta_prompt = self._compose_meta_prompt(task_description, context)
        meta_messages = [
            LLMMessage(role="system", content="You are an Orion meta-agent that decides how to create new agents."),
            LLMMessage(role="user", content=meta_prompt)
        ]

        # 2. Call the meta LLM with structured output enabled.
        meta_result = self.meta_llm.predict(
            model=self.default_model,
            messages=meta_messages,
            tools=None,
            stream=False,
            response_format=MetaOutput  # Use the MetaOutput schema for structured output.
        )

        agent_config = None
        if meta_result.structured:
            # Use the structured output directly.
            agent_config = MetaOutput(**meta_result.structured.dict())
        else:
            logger.error("Meta LLM failed to provide structured output")    
            raise ValueError("Meta LLM failed to provide structured output")

        # 3. Decide on agent configuration.
        agent_class = self.default_agent_class
        model = self.default_model
        system_prompt = f"System instructions for fallback: Task => {task_description}"
        generated_tools = []

        if agent_config:
            agent_class = self._resolve_agent_class(agent_config.agent_class)
            model = agent_config.model or self.default_model
            system_prompt = agent_config.system_prompt or system_prompt
            # 4. Optionally generate tools from the provided specifications.
            generated_tools = self._generate_tools(agent_config.tools)

        # 5. Instantiate the new agent.
        agent_name = self._unique_agent_name(agent_class)
        new_agent = agent_class(
            name=agent_name,
            role="auto-generated",
            description=system_prompt,
            model_name=model,
            api_key="",  # Replace with your API key or fetch from configuration.
            tools=generated_tools
        )

        # 6. Register and return the new agent.
        self.agent_registry[agent_name] = new_agent
        return new_agent

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------
    def _compose_meta_prompt(self, task_description: str, context: Dict[str, Any]) -> str:
        """
        Compose a detailed prompt for the meta LLM.

        This prompt instructs the meta LLM to analyze the given task description and
        return a valid JSON object with the following exact schema:
        
        {
            "agent_class": <string>,
            "model": <string>,
            "system_prompt": <string>,
            "tools": [
                {
                    "name": <string>,
                    "docstring": <string>,
                    "code": <string>
                },
                ... (zero or more tool definitions)
            ]
        }
        
        Field Explanations:
          - agent_class: The exact name of the Python class for the agent to create 
            (e.g., "NormalAgent", "ManagerAgent").
          - model: The name of the LLM model to use (e.g., "gpt-4", "gpt-4o", "gpt-3.5-turbo").
          - system_prompt: A detailed prompt that instructs the new agent on its role and objectives.
          - tools: An array of tool definitions. Each tool must include:
                - name: The function name to be used for the tool.
                - docstring: A clear description of what the tool does.
                - code: The complete Python code (as a string) defining the tool function.
        
        IMPORTANT:
          - Output ONLY the JSON object. Do not include any extra text, markdown formatting, or commentary.
        
        Task Description: {task_description}
        """
        detailed_prompt = (
            "You are an Orion Meta-Agent. Your job is to analyze the given task description and "
            "determine the best configuration for creating a new agent in the Orion framework. "
            "Based on the task, decide which agent class to use, what LLM model the agent should use, "
            "what system prompt (instructions) should be assigned to the agent, and which tools (if any) "
            "the agent should be allowed to use. If tools are required, you must also provide the full Python "
            "code for each tool as a string.\n\n"
            
            "Your output must be a valid JSON object that strictly conforms to the following schema:\n\n"
            
            "{\n"
            '  "agent_class": <string>,\n'
            '  "model": <string>,\n'
            '  "system_prompt": <string>,\n'
            '  "tools": [\n'
            "    {\n"
            '      "name": <string>,\n'
            '      "docstring": <string>,\n'
            '      "code": <string>\n'
            "    },\n"
            "    ...\n"
            "  ]\n"
            "}\n\n"
            
            "Field Explanations:\n"
            "- agent_class: The exact name of the Python class for the agent to create (e.g., 'NormalAgent').\n"
            "- model: The name of the LLM model to use (e.g., 'gpt-4', 'gpt-4o', 'gpt-3.5-turbo').\n"
            "- system_prompt: A detailed prompt that instructs the new agent on its role and objectives. "
            "This should be clear enough so that the agent understands its task fully.\n"
            "- tools: An array of tool definitions. Each tool is an object with three fields:\n"
            "    - name: The function name to be used for the tool.\n"
            "    - docstring: A description of what the tool does.\n"
            "    - code: The complete Python code (as a string) defining the tool function. The code should include "
            "the function signature and body. If no tools are needed, return an empty array for 'tools'.\n\n"
            
            "IMPORTANT: Output ONLY the JSON object. Do not include any markdown code blocks, explanations, or extra text.\n\n"
            
            "Task Description: " + task_description
        )
        return detailed_prompt

    def _extract_json(self, text: str) -> str:
        """
        A simple method that attempts to extract a JSON object from the provided text.
        """
        json_pattern = r'(\{.*\})'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _resolve_agent_class(self, class_name: str) -> Callable[..., BaseAgent]:
        """
        Convert a string name (e.g., 'NormalAgent') into the actual Python class object.
        """
        known_agents = {
            "NormalAgent": NormalAgent,
            # "ManagerAgent": ManagerAgent,
            # "WorkerAgent": WorkerAgent,
            # Add additional agent classes as needed.
        }
        return known_agents.get(class_name, self.default_agent_class)

    def _generate_tools(self, tool_specs: List[MetaOutputToolSpec]) -> List[Callable]:
        """
        For each tool specification, compile the provided Python code into a callable function,
        attach its docstring, and store it in the tool registry.
        """
        generated = []
        for tool_def in tool_specs:
            if tool_def.name in self.tool_registry:
                logger.info(f"Tool '{tool_def.name}' already exists; reusing it.")
                generated.append(self.tool_registry[tool_def.name])
                continue

            try:
                local_dict = {}
                exec(tool_def.code, {}, local_dict)
                func = local_dict.get(tool_def.name, None)
                if func is None or not callable(func):
                    raise ValueError(f"Could not find callable named '{tool_def.name}' after executing the code.")
                func.__doc__ = tool_def.docstring
                self.tool_registry[tool_def.name] = func
                generated.append(func)
                logger.info(f"Generated tool: {tool_def.name}")
            except Exception as e:
                logger.warning(f"Failed to generate tool '{tool_def.name}': {e}")
        return generated

    def _unique_agent_name(self, agent_class: Callable[..., BaseAgent]) -> str:
        """
        Generates a unique name for the newly created agent (e.g., "NormalAgent_5").
        """
        count = sum(
            1
            for name, agent in self.agent_registry.items()
            if isinstance(agent, agent_class)
        )
        return f"{agent_class.__name__}_{count+1}"
