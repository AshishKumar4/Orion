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
    from openai import NOT_GIVEN
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
import threading

from orion.utils import logger

from .supported_models import DEFAULT_SUPPORTED_MODELS

def _parse_tool_call(tool_call: ParsedFunctionToolCall, toolmap: Dict[str, LLMTool]) -> LLMToolCall:
    """
    Convert a parsed tool call to an LLMToolCall object.
    """
    function = toolmap.get(tool_call.function.name)
    args = json.loads(tool_call.function.arguments)
    return LLMToolCall(
        name=tool_call.function.name,
        id=tool_call.id,
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
    return NonStreamingResult(text=text, tool_calls=invoked_tools, structured=None)

def _function_to_schema(func: Callable) -> Dict[str, Any]:
    """
    Auto-generate a JSON schema from a Python function using docstring & type hints,
    ALWAYS using strict mode:
      - "strict": true
      - "additionalProperties": false
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

    # Force additionalProperties = false for strict mode
    params_schema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False
    }
    if required:
        params_schema["required"] = required

    # Return a function schema that has "strict": True
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc,
            "parameters": params_schema,
            "strict": True  # strictly enforce valid JSON
        }
    }

def _map_python_to_json_type(py_type: Any) -> Union[str, List[str]]:
    """
    Minimal mapping from Python type -> JSON schema type.
    For example, if you want to allow null for optional fields, you could do
    ["string", "null"] or something similar. But for now we do a simple approach:
    """
    # If you want to allow optional fields, you must define
    # param: Optional[str] or param: str|None, etc. in your code,
    # then interpret that here. For brevity, we'll do a naive approach.
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

    def __init__(self, model_name: str = 'gpt-4o'):
        if not OpenAI:
            raise ImportError("OpenAI library not found.")
        self.model_name = model_name
        info = DEFAULT_SUPPORTED_MODELS.get(model_name)
        if not info:
            raise ValueError(f"Model {model_name} not found in supported models.")
        self._api_key = info.api_key
        self._api_url = info.api_url
        self._client = OpenAI(api_key=self._api_key, base_url=self._api_url)
        self.lock = threading.Lock()

    def predict(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[LLMTool]] = None,
        stream: bool = False,
        response_format: Optional[Union[Type[BaseModel],
            Type[enum.Enum]]] = None,
    ) -> PredictResult:
        raw_messages = [{"role": m.role, "content": m.content} for m in messages]

        # Convert python callables to JSON schemas (strict).
        tool_schemas = []
        toolmap = {}
        if tools is not None:
            for tool in tools:
                schema = _function_to_schema(tool)
                tool_schemas.append(schema)
                toolmap[schema["function"]["name"]] = tool

        # The OpenAI library has 'tools' or 'functions' param (depending on your snippet).
        # We'll pass them as 'tools' if not empty.
        common_args = {
            "model": self.model_name,
            "messages": raw_messages,
            "tools": tool_schemas if tool_schemas else NOT_GIVEN,
        }
        
        with self.lock:
            client = self._client

        # If user wants a structured output => parse(...) call
        if response_format is not None:
            # Non-streaming only
            completion = client.beta.chat.completions.parse(
                **common_args,
                response_format=response_format,
            )
            return _handle_parsed_completion(completion, toolmap)
        else:
            # Normal
            if stream:
                response_iter = client.chat.completions.create(
                    **common_args,
                    stream=True
                )
                chunk_gen = self._stream_chunks(response_iter, toolmap)
                # We "tee" the generator so we can produce separate text & tool_calls iterators
                iter1, iter2 = itertools.tee(chunk_gen, 2)
                text_iter = (c["text"] for c in iter1)
                # If there's "tool_calls" in the chunk, yield them, else empty list
                tool_calls_iter = (c["tool_calls"] for c in iter2 if "tool_calls" in c)

                return StreamingResult(
                    text=text_iter,
                    tool_calls=tool_calls_iter,
                    structured=None
                )
            else:
                completion = client.chat.completions.create(
                    **common_args,
                    stream=False
                )
                return _handle_normal_completion(completion, toolmap)

    def _stream_chunks(
        self,
        response_iter: Stream[ChatCompletionChunk],
        toolmap: Dict[str, LLMTool]
    ) -> Iterator[Dict[str, Any]]:
        """
        Processes streaming chunks from OpenAI's response.
        Strict mode is enforced in the function schemas,
        so the model must produce valid arguments that match the schema.
        
        Yields a dict:
          {
            "text": str,
            "tool_calls": list[LLMToolCall] (only on final chunk if there's a finish_reason)
          }
        """
        accumulated_tool_calls: Dict[int, Dict[str, str]] = {}
        
        for chunk in response_iter:
            for choice in chunk.choices:
                delta = choice.delta
                text_chunk = delta.content or ""
                # If we see tool_calls in the chunk
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {"name": "", "arguments": "", "id": ""}
                        if tc.function and tc.function.name:
                            accumulated_tool_calls[idx]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            accumulated_tool_calls[idx]["arguments"] += tc.function.arguments
                        if tc.id:
                            accumulated_tool_calls[idx]["id"] = tc.id

                finish_reason = choice.finish_reason
                if finish_reason is not None:
                    # We finalize the tool calls
                    final_tool_calls = []
                    for idx, info in accumulated_tool_calls.items():
                        if info["name"]:
                            # parse the JSON from info["arguments"]
                            try:
                                parsed_args = json.loads(info["arguments"])
                            except json.JSONDecodeError:
                                # If the model gave malformed JSON, it fails in strict mode anyway
                                parsed_args = {}
                            final_tool_calls.append(
                                LLMToolCall(
                                    name=info["name"],
                                    arguments=parsed_args,
                                    function=toolmap.get(info["name"]),
                                    id=info["id"]
                                )
                            )
                    yield {"text": text_chunk, "tool_calls": final_tool_calls}
                    accumulated_tool_calls.clear()
                else:
                    # intermediate chunk => yield text only
                    yield {"text": text_chunk}
                    
    def cancel(self):
        """
        Cancels the current request.
        """
        with self.lock:
            logger.debug("Cancelling OpenAI request.")
            self._client.close()
            self._client = OpenAI(api_key=self._api_key, base_url=self._api_url)
            logger.debug("OpenAI client reset.")