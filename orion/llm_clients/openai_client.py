import inspect
from collections import defaultdict
import itertools
from typing import Any, Dict, List, Optional, Union, Type, Callable, Iterator
from pydantic import BaseModel
import enum

try:
    from openai import OpenAI  # As per your snippet usage
    from openai.types.chat import ChatCompletion, ParsedChatCompletion, ChatCompletionChunk, ParsedFunctionToolCall
    from openai import Stream
except ImportError:
    OpenAI = None

from .base_client import (
    BaseLLMClient,
    LLMMessage,
    StreamingResult,
    NonStreamingResult,
    PredictResult,
    LLMToolCall,
    LLMTool,
)
import json

def _parse_tool_call(tool_call: ParsedFunctionToolCall, toolmap: Dict[str, LLMTool]) -> LLMToolCall:
    """
    Convert a parsed tool call to an LLMToolCall object.
    """
    function = toolmap.get(tool_call.function.name)
    args = json.loads(tool_call.function.arguments)
    return LLMToolCall(
        name=tool_call.function.name,
        arguments=args,
        function=function
    )

def _handle_parsed_completion(completion: ParsedChatCompletion, toolmap: Dict[str, LLMTool]) -> NonStreamingResult:
    """
    Handles non-streaming parsed completions using OpenAI's beta parse API.
    """
    choice = completion.choices[0]
    message = choice.message
    text = message.content
    invoked_tools = []
    if hasattr(message, "tool_calls"):
        for tc in message.tool_calls:
            invoked_tools.append(_parse_tool_call(tc, toolmap))
    structured_obj = message.parsed
    return NonStreamingResult(text=text, tool_calls=invoked_tools, structured=structured_obj)


def _handle_normal_completion(completion: ChatCompletion, toolmap: Dict[str, LLMTool]) -> NonStreamingResult:
    """
    Handles non-streaming completions from chat.completions.create.
    """
    choice = completion.choices[0]
    message = choice.message
    text = message.content
    invoked_tools = []
    if message.tool_calls:
        for tc in message.tool_calls:
            invoked_tools.append(_parse_tool_call(tc, toolmap))
    # print(f'text: {text}, invoked_tools: {invoked_tools}')
    return NonStreamingResult(text=text, tool_calls=invoked_tools, structured=None)


def _function_to_schema(func: Callable) -> Dict[str, Any]:
    """
    Auto-generate a JSON schema from a Python function using docstring and type hints.
    """
    sig = inspect.signature(func)
    doc = (func.__doc__ or "").strip()
    from typing import get_type_hints
    type_hints = get_type_hints(func)
    properties = {}
    required = []
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        is_required = (param.default is inspect._empty)
        py_type = type_hints.get(param_name, str)
        schema_type = _map_python_to_json_type(py_type)
        properties[param_name] = {"type": schema_type}
        if is_required:
            required.append(param_name)
    params_schema = {"type": "object", "properties": properties}
    if required:
        params_schema["required"] = required
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc,
            "parameters": params_schema
        }
    }


def _map_python_to_json_type(py_type: Any) -> str:
    """
    Minimal mapping from Python type to JSON schema type.
    """
    if py_type == int:
        return "integer"
    elif py_type == float:
        return "number"
    elif py_type == bool:
        return "boolean"
    elif py_type == str:
        return "string"
    return "string"


class OpenAIClient(BaseLLMClient):
    """
    Implements the unified `predict(...)` for the OpenAI SDK.
    Supports both streaming and non-streaming outputs, function calling,
    and structured output (via response_format).
    """

    def __init__(self, api_key: str, api_url: str = None, default_model: str = "gpt-4"):
        if not OpenAI:
            raise ImportError("OpenAI library not found.")
        self._api_key = api_key
        self._default_model = default_model
        self._client = OpenAI(api_key=self._api_key, base_url=api_url)

    def predict(
        self,
        model: str,
        messages: List[LLMMessage],
        tools: Optional[List[LLMTool]] = None,
        stream: bool = False,
        response_format: Optional[Union[Type[BaseModel],
            Type[enum.Enum]]] = None,
    ) -> PredictResult:
        use_model = model or self._default_model
        raw_messages = [{"role": m.role, "content": m.content}
            for m in messages]

        # Convert python callables to JSON schemas.
        tool_schemas = []
        toolmap = {}
        if tools:
            for tool in tools:
                schema = _function_to_schema(tool)
                # print(f'schema: {schema}')
                tool_schemas.append(schema)
                toolmap[schema["function"]["name"]] = tool

        common_args = {
            "model": use_model,
            "messages": raw_messages,
            "tools": tool_schemas if len(tool_schemas) > 0 else None,
        }

        if response_format is not None:
            # Use beta.parse for structured output (non-streaming)
            completion = self._client.beta.chat.completions.parse(
                **common_args,
                response_format=response_format,
            )
            return _handle_parsed_completion(completion, toolmap)
        else:
            if stream:
                response_iter = self._client.chat.completions.create(
                    **common_args,
                    stream=True
                )
                # Create a generator that processes each chunk.
                chunk_gen = self._stream_chunks(response_iter, toolmap)
                # Use itertools.tee to split the generator into two iterators:
                iter1, iter2 = itertools.tee(chunk_gen, 2)
                # Map each chunk to extract text and tool_calls separately.
                text_iter = (chunk["text"] for chunk in iter1)
                tool_calls_iter = (chunk["tool_calls"] for chunk in iter2 if "tool_calls" in chunk)
                return StreamingResult(
                    text=text_iter,
                    tool_calls=tool_calls_iter,
                    structured=None
                )
            else:
                completion = self._client.chat.completions.create(
                    **common_args,
                    stream=False
                )
                return _handle_normal_completion(completion, toolmap)

    def _stream_chunks(
        self, response_iter: Stream[ChatCompletionChunk], toolmap: Dict[str, LLMTool]
    ) -> Iterator[Dict[str, Any]]:
        """
        Processes streaming chunks from OpenAI's response.
        Yields a dictionary with keys:
          - "text": a text chunk (possibly empty)
          - "tool_calls": a list of LLMToolCall objects ONLY when the tool call is complete,
            otherwise an empty list.
        """
        accumulated_tool_calls: Dict[int, Dict[str, str]] = {}  # index -> {"name": str, "arguments": str}
        
        for chunk in response_iter:
            for choice in chunk.choices:
                delta = choice.delta
                # Extract text content (if any)
                text_chunk = delta.content if delta.content is not None else ""
                
                # Process tool_calls if present (for APIs that support multiple calls per chunk)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {"name": "", "arguments": ""}
                        if tc.function and tc.function.name:
                            accumulated_tool_calls[idx]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            accumulated_tool_calls[idx]["arguments"] += tc.function.arguments

                # Check if the chunk has a finish_reason (indicating the tool call is complete)
                finish_reason = choice.finish_reason
                if finish_reason is not None:
                    # Finalize tool calls: convert accumulated data to LLMToolCall objects.
                    complete_tool_calls = []
                    for idx, info in accumulated_tool_calls.items():
                        if info["name"]:
                            complete_tool_calls.append(
                                LLMToolCall(
                                    name=info["name"],
                                    arguments=json.loads(info["arguments"]),
                                    function=toolmap.get(info["name"])
                                )
                            )
                    # Yield this chunk with its text and the complete tool calls.
                    yield {"text": text_chunk, "tool_calls": complete_tool_calls}
                    # Clear the accumulator for future tool calls.
                    accumulated_tool_calls.clear()
                else:
                    # Not final—yield the text and an empty list (tool call not yet complete)
                    yield {"text": text_chunk}