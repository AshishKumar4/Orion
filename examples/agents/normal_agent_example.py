"""
Demonstration of NormalAgent with concurrency, streaming, multiple tools, and cancellation.
"""

from typing import Iterator
import time
from concurrent.futures import Future

from orion.agents.normal_agent import NormalAgent
from orion.config import config

# Define multiple tool functions the LLM might call:

def greet_user(name: str) -> str:
    """Returns a greeting for the specified user."""
    print(f"[TOOL] greet_user called with name={name}")
    return f"Hello, {name}! Nice to meet you."

def multiply_numbers(a: int, b: int) -> int:
    """Multiplies a and b."""
    print(f"[TOOL] multiply_numbers called with {a} and {b}")
    return a * b

def random_wait(seconds: int) -> str:
    """Waits for the specified number of seconds, then returns a message."""
    print(f"[TOOL] random_wait is sleeping for {seconds} second(s).")
    time.sleep(seconds)
    return f"Slept for {seconds} second(s)."

def main():
    # 1) Instantiate the agent
    agent = NormalAgent(
        name="MultiToolAgent",
        role="Assistant",
        description="You are a helpful agent that can greet users, multiply numbers, and wait as needed.",
        model_name="gpt-4o",  # or "gemini"
        tools=[greet_user, multiply_numbers, random_wait],  # multiple tools
    )

    # 2) Demonstrate basic chat usage (tools not forced)
    print("\n=== 2) Basic Chat Usage ===")
    chat_stream = agent.chat("Hi, who are you?", background=False)
    for chunk in chat_stream:
        # Each chunk is text. We'll print them as they arrive.
        print(chunk, end="", flush=True)
    print()  # newline

    # 3) Demonstrate 'do' usage with multiple tools
    # We'll ask it to greet user "Alice" and multiply 7 and 3,
    # hoping the LLM calls both greet_user() and multiply_numbers().
    print("\n=== 3) 'do' Usage with Tools ===")
    do_stream = agent.do("Please greet user named Alice and also multiply 7 and 3.", background=False)
    for chunk in do_stream:
        # Could be text or final tool results
        print(chunk, end="", flush=True)
    print()  # newline

    # 4) Concurrency: Run 'do' in the background while we do something else.
    print("\n=== 4) Concurrency Demonstration ===")
    future_result = agent.do("Now do random_wait(2) in the background. Please also greet me as Bob!", background=True)
    stream = future_result
    # Meanwhile, let's do something else here
    for i in range(5):
        print(f"[MAIN THREAD] Doing other work... iteration {i}")
        time.sleep(2)

    # Now retrieve the final result from the agent
    print("[MAIN THREAD] Let's see what the agent produced in the background...")
    if isinstance(future_result, Future):
        for chunk in stream:
            print(chunk, end="", flush=True)
        print()
    else:
        print("ERROR: Expected a Future but didn't get one.")

    # 5) Cancellation demonstration
    print("\n=== 5) Cancellation Demonstration ===")
    # We'll spawn a 'do' request that calls random_wait(10), but we will cancel after a few seconds.
    result_after_cancel = agent.do("Please do random_wait(10) while greeting me as Charlie", background=True)
    print("[MAIN THREAD] Launched a long running 'do' call, let's wait ~2 seconds then cancel.")
    time.sleep(1)
    agent.cancel()  # triggers self._cancel_requested = True
    time.sleep(5)

    # The background thread should stop streaming
    # We'll see partial output if the LLM had streamed something
    # or an immediate stop if it checks the cancel flag quickly.
    # It's a generator, so let's read any partial output
    chunks = []
    for chunk in result_after_cancel:
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    if chunks:
        print("\n[MAIN THREAD] Surprising that we got chunks after cancel? Depends on timing...")
    else:
        print("[MAIN THREAD] The streaming was canceled, so no final output was returned.")

if __name__ == "__main__":
    main()
