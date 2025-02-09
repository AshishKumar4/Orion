"""
Example: Using the `response_format` param for structured outputs with a Pydantic model.
OpenAIClient will use the .beta.chat.completions.parse(...) under the hood.
"""

from pydantic import BaseModel
from orion.config import config
from orion.llm_clients.openai_client import OpenAIClient
from orion.llm_clients.base_client import LLMMessage

class MovieReview(BaseModel):
    """
    Example Pydantic model representing a structured output from the LLM:
    e.g. a sentiment analysis on a movie review.
    """
    sentiment: str
    rating: int

def main():
    client = OpenAIClient(
        # model_name="gpt-4o"
        model_name="gemini-2.0-flash"
    )

    messages = [
        LLMMessage(
            role="user",
            content=(
                "Perform a sentiment analysis on the following review:\n"
                "\"I absolutely loved this movie! The plot was thrilling, "\
                "the performances were top-notch, and I'd watch it again.\"\n"
                "Return your answer in the structured format."
            )
        )
    ]

    # We'll parse the LLM's response into the MovieReview Pydantic model
    result = client.predict(
        messages=messages,
        tools=None,
        stream=False,
        response_format=MovieReview,
    )

    print("=== Structured Output Example ===")
    print("Text:", result.text)  # Possibly a summary or explanation
    print("Tool Calls:", result.tool_calls)
    if result.structured:
        print("Parsed MovieReview:", result.structured.dict(), MovieReview(**result.structured.dict()))
    else:
        print("No structured data returned...")

if __name__ == "__main__":
    main()
