"""
Example: Simple streaming chat without tools or structured output.
Demonstrates how partial text is yielded in chunks.
"""

from orion.config import config
from orion.llm_clients.openai_client import OpenAIClient
from orion.llm_clients.base_client import LLMMessage

def main():
    client = OpenAIClient(
        model_name="gemini-2.0-flash"
    )

    messages = [
        LLMMessage(role="user", content="Tell me a short story about a brave knight in 50 words.")
    ]

    # Streaming usage, no tools, no structured output
    result = client.predict(
        messages=messages,
        tools=None,
        stream=True,
        response_format=None,
    )

    # result is a StreamingResult with .text, .tool_calls, .structured
    print("=== Streaming Simple Chat Example ===")

    print("Text chunks:")
    for chunk in result.text:
        # chunk is a partial string from the LLM
        print(chunk, end="", flush=True)
    print("\n--- End of Text ---")

    # We expect no tool calls here
    print("Tool calls per chunk:")
    for i, tool_calls_chunk in enumerate(result.tool_calls):
        print(f"  Chunk {i}: {tool_calls_chunk}")
    print("--- End of Tool Calls ---")

if __name__ == "__main__":
    main()
