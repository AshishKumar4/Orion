"""
Example: Streaming usage + function calling with Python tools.
The LLM can invoke these tools partially, and we finalize them when the chunk finish reason is found.
"""

import json
from orion.config import config
from orion.llm_clients.openai_client import OpenAIClient
from orion.llm_clients.base_client import LLMMessage, LLMTool


def multiply(a: int, b: int) -> int:
    """
    Multiply two integers a and b.
    """
    return a * b

def greet(name: str) -> str:
    """
    Return a greeting for the given name.
    """
    return f"Hello, {name}!"

def short_story(story: str) -> str:
    """
    Return a short story about a baby lion.
    """
    return story

def main():
    client = OpenAIClient(
        api_key=config.OPENAI_API_KEY,
        default_model="gpt-4o"
    )

    messages = [
        LLMMessage(role="user", content="Please write a short story about a baby lion. Also, you can maybe  call greet on me. My name is Alice. Also maybe you can try multiply 3 and 4. ")
    ]

    # We pass in our Python tools
    tools = [multiply, greet, short_story]

    # Stream the partial outputs
    result = client.predict(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        stream=True,
        response_format=None,
    )

    print("=== Streaming + Tools Example ===")

    collected_text = []
    print("Text chunks:")
    for chunk in result.text:
        collected_text.append(chunk)
        print(chunk, end="", flush=True)
    print("\n--- End of Text ---")

    # Now check partial tool calls
    print("Tool calls per chunk:")
    for i, tcs in enumerate(result.tool_calls):
        if tcs:
            print(f"  Chunk {i} => {len(tcs)} completed tool call(s):")
            for tc in tcs:
                print(f"    Name: {tc.name}")
                print(f"    Arguments: {json.dumps(tc.arguments, indent=2)}")
                if tc.function:
                    print(f"    Python function: {tc.function.__name__}")
                # Execute the tool function and print the result
                print(f"    Result: {tc.function(**tc.arguments)}")
        else:
            print(f"  Chunk {i} => no completed tool calls")

if __name__ == "__main__":
    main()
