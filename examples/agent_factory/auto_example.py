#!/usr/bin/env python
"""
orion/examples/agent_factory/auto_example.py

This example demonstrates how to use the AutoAgentFactory in Orion.
A dummy meta LLM client is used to return a structured output (using the MetaOutput schema)
that instructs the factory on which agent to create, what model to use, which system prompt
to assign, and what tools to attach.

Author: Ashish Kumar Singh
"""

import logging
from typing import Any, List

from orion.agent_factory.auto import AutoAgentFactory, MetaOutput, MetaOutputToolSpec
from orion.llm_clients.base_client import BaseLLMClient, LLMMessage, NonStreamingResult
from orion.agents.normal_agent import NormalAgent
from orion.agents.base_agent import BaseAgent

# ----------------------------------------------------------------------
# Dummy Meta LLM Client Implementation
# ----------------------------------------------------------------------
class DummyMetaLLMClient(BaseLLMClient):
    """
    A dummy meta LLM client that returns a fixed structured output using the MetaOutput schema.
    This simulates the behavior of a powerful meta agent that, given a task description,
    decides which agent to create.
    """

    def __init__(self):
        super().__init__()

    def predict(
        self,
        model: str,
        messages: List[LLMMessage],
        tools: Any = None,
        stream: bool = False,
        response_format: Any = None
    ) -> NonStreamingResult:
        # Create a dummy structured output for demonstration.
        meta_output = MetaOutput(
            agent_class="NormalAgent",
            model="gpt-3.5-turbo",
            system_prompt="You are a dummy agent created to handle marketing tasks. "
                          "Provide insights and analysis on planning a marketing campaign.",
            tools=[
                MetaOutputToolSpec(
                    name="dummy_tool",
                    docstring="This tool returns a fixed string for demonstration purposes.",
                    code="""
def dummy_tool():
    return "This is a dummy tool result."
"""
                )
            ]
        )
        # Return a NonStreamingResult with the structured field populated.
        return NonStreamingResult(
            text="Dummy meta LLM response",
            tool_calls=[],
            structured=meta_output
        )

# ----------------------------------------------------------------------
# Main Example Using AutoAgentFactory
# ----------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.DEBUG)
    
    # Instantiate the dummy meta LLM client.
    dummy_meta_llm = DummyMetaLLMClient()
    
    # Create the AutoAgentFactory with the dummy meta LLM.
    factory = AutoAgentFactory(
        meta_llm=dummy_meta_llm,
        default_agent_class=NormalAgent,
        default_model="gpt-4o"
    )
    
    # Provide a task description.
    task_description = "Scrape the data from the website https://ashishkumarsingh.com, and provide a detailed summary and insights."
    
    # Use the factory to create a new agent based on the task description.
    new_agent = factory.create_agent(task_description)
    
    # Print details about the newly created agent.
    print("=== New Agent Created ===")
    print(f"Name         : {new_agent.name}")
    print(f"Role         : {new_agent.role}")
    print(f"Model        : {new_agent.model_name}")
    print(f"System Prompt: {new_agent.description}")
    
    if new_agent.tools:
        print("\nAttached Tools:")
        for tool in new_agent.tools:
            # Assuming the tool is a Python callable.
            tool_name = getattr(tool, "__name__", "unknown_tool")
            tool_doc = getattr(tool, "__doc__", "").strip()
            print(f"  - {tool_name}: {tool_doc}")
    else:
        print("\nNo tools attached to the agent.")
    
    # Simulate a simple chat with the new agent.
    # (Assuming NormalAgent implements a chat() method returning an iterator of text chunks.)
    print("\n=== Simulating Agent Chat ===")
    try:
        chat_prompt = "Hello"
        chat_stream = new_agent.chat(chat_prompt, background=False)
        for chunk in chat_stream:
            print(chunk, end="")
        print()  # Newline after the streaming output.
    except Exception as e:
        print(f"Error during chat simulation: {e}")

if __name__ == "__main__":
    main()
