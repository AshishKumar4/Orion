import json
import logging
import re
from typing import Optional, Dict, Any, List, Callable

from pydantic import BaseModel, Field, ValidationError

from orion.llm_clients.base_client import BaseLLMClient, LLMMessage
from orion.agents.base_agent import BaseAgent
from orion.agents.normal_agent import NormalAgent
from .base_factory import AgentFactory

# ----------------------------------------------------------------------
# 1. Define a Data Model for the Meta LLM’s Structured Output
#    (If you want a JSON-based approach, for instance)
# ----------------------------------------------------------------------

class MetaOutputToolSpec(BaseModel):
    """Definition for a single tool that the meta LLM says we should create."""
    name: str
    docstring: str
    code: str  # the Python code that implements the function

class MetaOutput(BaseModel):
    """Definition for the structured plan from the meta LLM."""
    agent_class: str = Field(..., description="Name of the agent class to create.")
    model: str = Field(..., description="LLM model name to use for the new agent.")
    system_prompt: str = Field(..., description="System prompt for the new agent.")
    tools: List[MetaOutputToolSpec] = Field(default_factory=list, description="List of custom tools to generate.")

# ----------------------------------------------------------------------
# 2. AutoAgentFactory Implementation
# ----------------------------------------------------------------------

class AutoAgentFactory(AgentFactory):
    """
    A specialized factory that uses a "meta" LLM to parse a task description
    and decide how to create a new agent:

      - Which Agent class (e.g. NormalAgent, ManagerAgent, etc.)
      - Which model to use
      - System prompt
      - Which tools to attach (some may be auto-generated on-the-fly)

    If the meta LLM fails to provide valid JSON or leaves out fields,
    we fallback to a default agent.
    """

    def __init__(
        self,
        meta_llm: BaseLLMClient,
        default_agent_class: Callable[..., BaseAgent] = NormalAgent,
        default_model: str = "gpt-4o",
        registry: Optional[Dict[str, BaseAgent]] = None,
        tool_registry: Optional[Dict[str, Callable]] = None,
    ):
        """
        :param meta_llm: An LLM client used to reason about tasks and produce an agent creation plan.
        :param default_agent_class: The fallback Agent class if the meta LLM’s output is invalid or missing info.
        :param default_model: If the meta LLM doesn’t specify a model, fallback to this.
        :param registry: Optional dictionary to store references to created agents by name.
        :param tool_registry: Optional dictionary to store or retrieve code-generated tool functions by name.
        """
        self.meta_llm = meta_llm
        self.default_agent_class = default_agent_class
        self.default_model = default_model
        self.agent_registry = registry if registry else {}
        self.tool_registry = tool_registry if tool_registry else {}

    def create_agent(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> BaseAgent:
        """
        1. Prompt the meta LLM with the task description.
        2. Expect JSON output describing the agent class, model, system prompt, and tools.
        3. Parse the JSON. If invalid, fallback to default.
        4. Optionally generate Python tools from the code snippets provided.
        5. Instantiate the agent, attach tools, and store in registry.
        """
        if context is None:
            context = {}

        # 1. Prompt the meta LLM
        meta_prompt = self._compose_meta_prompt(task_description, context)
        meta_messages = [
            LLMMessage(role="system", content="You are an Orion meta-agent that decides how to create new agents."),
            LLMMessage(role="user", content=meta_prompt)
        ]

        meta_result = self.meta_llm.predict(
            model=self.default_model,  # meta agent uses default model
            messages=meta_messages,
            tools=None,
            stream=False,
            response_format=None  # We'll parse JSON ourselves
        )

        plan_text = meta_result.text or ""
        logging.debug(f"Meta LLM plan text = {plan_text!r}")

        # 2. Parse the JSON
        agent_config = None
        try:
            # We assume the meta LLM outputs valid JSON inside a code block or something.
            # We'll do a simple regex to extract the JSON portion if needed
            plan_json = self._extract_json(plan_text)
            # Validate with MetaOutput
            agent_config = MetaOutput.parse_raw(plan_json)
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Failed to parse meta LLM output, using fallback agent. Error: {e}")

        # 3. Decide agent class, model, prompt, tools
        agent_class = self.default_agent_class
        model = self.default_model
        system_prompt = f"System instructions for fallback: Task => {task_description}"
        generated_tools = []

        if agent_config:
            agent_class = self._resolve_agent_class(agent_config.agent_class)
            model = agent_config.model or self.default_model
            system_prompt = agent_config.system_prompt or system_prompt
            # 4. Possibly generate new tools
            generated_tools = self._generate_tools(agent_config.tools)

        # 5. Instantiate the agent
        agent_name = self._unique_agent_name(agent_class)
        new_agent = agent_class(
            name=agent_name,
            role="auto-generated",
            description=system_prompt,
            model_name=model,
            api_key="",  # Provide real API key or read from config
            tools=generated_tools
        )

        # 6. Save to registry, return it
        self.agent_registry[agent_name] = new_agent
        return new_agent

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------
    def _compose_meta_prompt(self, task_description: str, context: Dict[str, Any]) -> str:
        """
        Compose a detailed prompt for the meta LLM.

        This prompt instructs the meta LLM to analyze the provided task description and return a valid JSON object
        with the following exact schema:
        
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

        Where:
          - "agent_class": is the name of the Orion agent class best suited to handle the task (e.g., "NormalAgent", "ManagerAgent").
          - "model": is the LLM model to be used for this new agent (e.g., "gpt-4", "gpt-4o", "gpt-3.5-turbo").
          - "system_prompt": is the set of instructions that defines the agent's role and objectives.
          - "tools": is an array of tool specifications. If no tools are needed, this should be an empty array.
            Each tool specification must include:
              - "name": the function name for the tool.
              - "docstring": a clear description of what the tool does.
              - "code": the complete Python code (as a string) that defines the function.
        
        IMPORTANT:
          - Do not include any extra text, markdown formatting, or commentary in your output.
          - Output only the JSON object in a format that can be directly parsed.
        
        Include any additional context from the provided context dictionary if needed.
        
        TASK DESCRIPTION: {task_description}
        """
        # Here we construct a multi-line string that clearly instructs the meta LLM.
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
            "- agent_class: The exact name of the Python class for the agent to create (for example, 'NormalAgent').\n"
            "- model: The name of the LLM model to use (for example, 'gpt-4', 'gpt-4o', or 'gpt-3.5-turbo').\n"
            "- system_prompt: A detailed prompt that instructs the new agent on its role and objectives. This should be "
            "clear enough so that the agent understands its task fully.\n"
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
        A naive approach that tries to find a JSON object in the text.
        You might do better by instructing the LLM to return raw JSON (no code blocks).
        """
        # If the LLM is well-behaved and returns raw JSON, we can just do:
        # return text.strip()
        # If it might return triple-backtick code blocks, we can do something like:
        json_pattern = r'(\{.*\})'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()  # fallback

    def _resolve_agent_class(self, class_name: str) -> Callable[..., BaseAgent]:
        """
        Convert a string name (like 'NormalAgent' or 'ManagerAgent') 
        into the actual Python class object.
        This can be done via a registry or a dictionary of known agent classes.
        """
        known_agents = {
            "NormalAgent": NormalAgent,
            # "ManagerAgent": ManagerAgent,
            # "WorkerAgent": WorkerAgent,
            # ...
        }
        return known_agents.get(class_name, self.default_agent_class)

    def _generate_tools(self, tool_specs: List[MetaOutputToolSpec]) -> List[Callable]:
        """
        For each tool specification, compile the python code into a function object,
        attach the docstring, and store it in `self.tool_registry`.
        This is advanced usage—be mindful of security if code is untrusted.
        """
        generated = []
        for tool_def in tool_specs:
            if tool_def.name in self.tool_registry:
                logging.info(f"Tool '{tool_def.name}' already exists in registry; reusing.")
                generated.append(self.tool_registry[tool_def.name])
                continue

            # 1. Compile the code
            try:
                # We'll define a local dict for execution
                local_dict = {}
                exec(tool_def.code, {}, local_dict)
                # We assume the function is named exactly tool_def.name
                func = local_dict.get(tool_def.name, None)
                if func is None or not callable(func):
                    raise ValueError(f"Could not find callable named '{tool_def.name}' after exec.")
                # 2. Attach docstring if needed
                func.__doc__ = tool_def.docstring
                # 3. Store in registry
                self.tool_registry[tool_def.name] = func
                generated.append(func)
                logging.info(f"Generated tool: {tool_def.name}")
            except Exception as e:
                logging.warning(f"Failed to generate tool '{tool_def.name}': {e}")
        return generated

    def _unique_agent_name(self, agent_class: Callable[..., BaseAgent]) -> str:
        """
        Generates a unique name for the newly created agent, e.g. "NormalAgent_5".
        """
        count = sum(
            1
            for name, agent in self.agent_registry.items()
            if isinstance(agent, agent_class)
        )
        return f"{agent_class.__name__}_{count+1}"
