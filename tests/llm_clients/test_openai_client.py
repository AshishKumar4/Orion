import pytest
from unittest.mock import MagicMock, patch

from pydantic import BaseModel
import enum

from orion.llm_clients.base_client import (
    LLMMessage,
    LLMToolCall,
    PredictResult,
)
from orion.llm_clients.openai_client import (
    OpenAIClient,
    _handle_normal_completion,
    _handle_parsed_completion,
)

from openai.types.chat import ChatCompletion, ParsedChatCompletion, ParsedFunctionToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction


######################################
# Mocked Data & Helpers
######################################
class MockParsedModel(BaseModel):
    field: str


class MockEnum(enum.Enum):
    CHOICE1 = "choice1"
    CHOICE2 = "choice2"


def mock_tool_one(param: str) -> str:
    """Mock tool docstring"""
    return f"Result {param}"


def mock_tool_two(x: int, y: int) -> int:
    """Another mock tool"""
    return x + y


@pytest.fixture
def messages():
    return [
        LLMMessage(role="user", content="Hello, world!"),
    ]


@pytest.fixture
def openai_client():
    # Create an OpenAIClient instance with dummy creds
    return OpenAIClient(api_key="test-key")


######################################
# Tests for Class Initialization
######################################
def test_openai_client_init_no_openai(monkeypatch):
    """
    If openai is missing, we expect an ImportError from the constructor.
    """
    with monkeypatch.context() as m:
        m.setattr("orion.llm_clients.openai_client.OpenAI", None)  # Force no OpenAI
        with pytest.raises(ImportError):
            OpenAIClient(api_key="does_not_matter")


######################################
# Tests for Non-streaming Predictions
######################################
def test_predict_non_streaming_no_tools(mocker, openai_client, messages):
    """
    When no tools and no response format, we should do a normal non-streaming
    chat.completions.create call. We'll mock the OpenAI client response.
    """
    mock_create = mocker.patch.object(
        openai_client._client.chat.completions, "create", autospec=True
    )

    # Mock response: fill all required fields
    mock_response = ChatCompletion(
        id="mock-id",
        object="chat.completion",
        created=1690000000,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Hello user!"
                }
            }
        ]
    )
    mock_create.return_value = mock_response

    result = openai_client.predict(
        model="gpt-4",
        messages=messages,
        tools=None,
        stream=False,
        response_format=None,
    )
    mock_create.assert_called_once_with(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello, world!"}],
        functions=[],
        stream=False
    )
    assert result.text == "Hello user!"
    assert result.tool_calls == []
    assert result.structured is None


def test_predict_non_streaming_with_tools(mocker, openai_client, messages):
    """
    When tools are provided, we auto-generate JSON schemas and pass them to create(...).
    Then if the response has tool_calls, we map them to LLMToolCall objects.
    """
    mock_create = mocker.patch.object(
        openai_client._client.chat.completions, "create", autospec=True
    )
    # Mock response with tool_calls
    mock_response = ChatCompletion(
        id="mock-id-2",
        object="chat.completion",
        created=1690000001,
        model="gpt-3.5-turbo",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Sure, calling the tool now...",
                    "tool_calls": [
                        {
                            "id": "tool-call-1",
                            "type": "function",
                            "function": {
                                "name": "mock_tool_one",
                                "arguments": '{"param":"value"}'
                            }
                        }
                    ],
                }
            }
        ]
    )
    mock_create.return_value = mock_response

    # Tools
    tools = [mock_tool_one, mock_tool_two]

    result = openai_client.predict(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        stream=False,
        response_format=None,
    )

    # The name & docstring from mock_tool_one -> schema
    # Check the client call:
    assert mock_create.called
    call_args = mock_create.call_args[1]
    assert call_args["model"] == "gpt-3.5-turbo"
    assert len(call_args["functions"]) == 2  # two tools -> two schemas

    # Validate the result
    assert result.text == "Sure, calling the tool now..."
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.name == "mock_tool_one"
    assert tc.arguments == {"param": "value"}
    # The function attribute is the actual Python tool
    assert tc.function == mock_tool_one


######################################
# Tests for Non-streaming + Response Format (Structured parse)
######################################
def test_predict_non_streaming_structured_parse(mocker, openai_client, messages):
    """
    When a response_format is given, we call .beta.chat.completions.parse(...) instead
    of create(...). We get a structured object and text.
    """
    parse_mock = mocker.patch.object(
        openai_client._client.beta.chat.completions, "parse", autospec=True
    )
    # Mock a ParsedChatCompletion
    mock_completion = ParsedChatCompletion(
        id="parsed-id",
        object="chat.completion",
        created=1690000002,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Here is some structured info",
                    "tool_calls": [
                        {
                            "id": "tool-call-xyz",
                            "type": "function",
                            "function": {
                                "name": "mock_tool_one",
                                "arguments": '{"param":"abc"}'
                            }
                        }
                    ],
                    "parsed": {
                        "field": "some_data"
                    },
                }
            }
        ]
    )
    parse_mock.return_value = mock_completion

    result = openai_client.predict(
        model="gpt-4",
        messages=messages,
        tools=[mock_tool_one],
        stream=False,
        response_format=MockParsedModel,
    )

    parse_mock.assert_called_once_with(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello, world!"}],
        functions=[
            {
                "type": "function",
                "function": {
                    "name": "mock_tool_one",
                    "description": "Mock tool docstring",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param": {"type": "string"}
                        },
                        "required": ["param"]
                    }
                }
            }
        ],
        response_format=MockParsedModel,
    )

    assert result.text == "Here is some structured info"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "mock_tool_one"
    assert result.tool_calls[0].arguments == {"param": "abc"}
    assert result.tool_calls[0].function == mock_tool_one
    # structured
    assert result.structured['field'] == "some_data"


######################################
# Tests for Streaming Predictions
######################################
@pytest.mark.parametrize("finish_reason", [None, "function_call"])
def test_predict_streaming_text_only(mocker, openai_client, messages, finish_reason):
    """
    Test streaming scenario with text chunks only. We'll produce partial text from each chunk.
    No tool_calls. We finalize text on last chunk if finish_reason is set.
    """
    mock_create = mocker.patch.object(
        openai_client._client.chat.completions, "create", autospec=True
    )
    # Prepare a generator of streaming responses
    def stream_responses():
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello "), finish_reason=None)])
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="World"), finish_reason=None)])
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="!"), finish_reason=finish_reason)])

    mock_create.return_value = stream_responses()

    streaming_result = openai_client.predict(
        model="gpt-4",
        messages=messages,
        tools=None,
        stream=True,
        response_format=None,
    )

    # streaming_result is a PredictResult where .text is an iterator of strings, .tool_calls is an iterator of lists
    assert hasattr(streaming_result.text, "__iter__")
    assert hasattr(streaming_result.tool_calls, "__iter__")

    # Collect text
    collected_text = "".join(list(streaming_result.text))
    assert collected_text == "Hello World!"

    # Tools
    collected_tools = list(streaming_result.tool_calls)
    # We expect no tool calls
    for chunk_tool_calls in collected_tools:
        assert chunk_tool_calls == []

def test_predict_streaming_with_tool_calls(mocker, openai_client, messages):
    """
    Test streaming scenario with partial tool_call accumulation using delta.tool_calls.
    We'll produce partial tool_call data across multiple chunks, then yield them
    only once we see a finish_reason in the chunk.
    """
    mock_create = mocker.patch.object(
        openai_client._client.chat.completions, "create", autospec=True
    )

    # We'll produce 4 chunks:
    #  1) partial text, no tool_calls
    #  2) partial tool_call #0: name = mock_tool_two
    #  3) partial tool_call #0: more arguments
    #  4) final chunk with finish_reason => finalize tool call
    # In the final chunk, we assume we have the entire 'arguments' for the call.

    def stream_responses():
        # chunk1: text "The tool says: " with no tool_calls
        chunk1 = MagicMock()
        chunk1.choices = [
            MagicMock(
                delta=MagicMock(
                    content="The tool says: ",
                    tool_calls=None
                ),
                finish_reason=None
            )
        ]
        yield chunk1

        # chunk2: partial name "mock_tool_two", no arguments
        chunk2 = MagicMock()
        chunk2.choices = [
            MagicMock(
                delta=MagicMock(
                    content=None,
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id="tc-1",
                            function=ChoiceDeltaToolCallFunction(
                                name="mock_tool_two",
                                arguments=""
                            )
                        )
                    ]
                ),
                finish_reason=None
            )
        ]
        yield chunk2

        # chunk3: partial arguments
        chunk3 = MagicMock()
        chunk3.choices = [
            MagicMock(
                delta=MagicMock(
                    content=None,
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id="tc-1",
                            function=ChoiceDeltaToolCallFunction(
                                name=None,
                                arguments='{"x":1'
                            )
                        )
                    ]
                ),
                finish_reason=None
            )
        ]
        yield chunk3

        # chunk4: final arguments, finish_reason => finalize
        chunk4 = MagicMock()
        chunk4.choices = [
            MagicMock(
                delta=MagicMock(
                    content=None,
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id="tc-1",
                            function=ChoiceDeltaToolCallFunction(
                                name=None,
                                arguments=', "y":2}'
                            )
                        )
                    ]
                ),
                finish_reason="stop"
            )
        ]
        yield chunk4

    mock_create.return_value = stream_responses()

    # Tools
    tools = [mock_tool_one, mock_tool_two]

    result = openai_client.predict(
        model="gpt-4",
        messages=messages,
        tools=tools,
        stream=True,
        response_format=None,
    )

    # Collect text chunks
    text_chunks = list(result.text)  # each iteration is partial text
    assert "".join(text_chunks) == "The tool says: "

    # Collect tool_calls from each chunk
    tool_calls_per_chunk = list(result.tool_calls)

    # chunk1 => no finished tool calls => []
    # chunk2 => partial data => not finished => []
    # chunk3 => partial arguments => still not finished => []
    # chunk4 => finish_reason => finalize => single tool call
    assert len(tool_calls_per_chunk) == 4
    assert tool_calls_per_chunk[0] == []
    assert tool_calls_per_chunk[1] == []
    assert tool_calls_per_chunk[2] == []

    final_calls = tool_calls_per_chunk[3]
    assert len(final_calls) == 1
    fc = final_calls[0]
    assert fc.name == "mock_tool_two"
    assert fc.arguments == {"arguments": '{"x":1, "y":2}'}
    # The function attribute should be the actual callable
    assert fc.function == mock_tool_two
