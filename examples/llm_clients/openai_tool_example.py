"""
Example: Using Python functions as tools for OpenAI function calling.
Shows how to define and pass your functions for dynamic function calls.
"""

import json
from orion.config import config
from orion.llm_clients.openai_client import OpenAIClient
from orion.llm_clients.base_client import LLMMessage, LLMTool

# Define some Python callables you'd like to expose as LLM tools

def get_weather(location: str) -> str:
    """
    Returns the weather in the given location.
    For demonstration, it returns a dummy string.
    """
    return f"The weather in {location} is sunny and 25C."

def add_numbers(x: int, y: int) -> int:
    """
    Adds two numbers x and y.
    """
    return x + y

def main():
    client = OpenAIClient(
        api_key=config.OPENAI_API_KEY,
        default_model="gpt-4o"
    )

    # Prompt that encourages the LLM to use one of the tools
    messages = [
        LLMMessage(role="user", content="What is the sum of 2 and 3? Please call add_numbers.")
    ]

    # Pass our functions as 'tools'
    tools = [get_weather, add_numbers]

    # Non-streaming predict with tools
    result = client.predict(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        stream=False,
        response_format=None,
    )

    print("=== Tools Example (Non-Streaming) ===")
    print("Text:", result.text)
    print("Tool Calls:")
    for tool_call in result.tool_calls:
        print("  - Name:", tool_call.name)
        print("    Arguments:", json.dumps(tool_call.arguments, indent=2))
        print("    Function:", tool_call.function.__name__)
    print("Structured:", result.structured)

if __name__ == "__main__":
    main()
