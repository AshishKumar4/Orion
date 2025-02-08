"""
Example: Basic usage of OpenAIClient without tools or structured outputs.
"""

from orion.config import config
from orion.llm_clients.openai_client import OpenAIClient
from orion.llm_clients.base_client import LLMMessage

def main():
    # Initialize the OpenAIClient with your API key from config
    client = OpenAIClient(api_key=config.OPENAI_API_KEY, default_model="gpt-4")

    messages = [
        LLMMessage(role="user", content="Hello, how are you today?")
    ]

    # Non-streaming predict call, no tools, no structured output
    result = client.predict(
        model="gpt-4o",
        messages=messages,
        tools=None,
        stream=False,
        response_format=None,
    )

    print("=== Non-Streaming Basic Example ===")
    print("Text:", result.text)
    print("Tool Calls:", result.tool_calls)
    print("Structured:", result.structured)

if __name__ == "__main__":
    main()
