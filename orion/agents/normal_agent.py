from typing import Iterator, List, Any
from itertools import zip_longest
import logging
from concurrent.futures import Future

from orion.config import config
from orion.llm_clients.base_client import (
    LLMMessage,
    StreamingResult,
    NonStreamingResult,
    LLMToolCall,
    PredictResult
)
from orion.llm_clients.openai_client import OpenAIClient
from .base_agent import (
    BaseAgent,
    AgentState,
    MessageStatus,
    ConversationItem
)
from orion.utils import logger
import json

class NormalAgent(BaseAgent):
    """
    A streaming-only implementation of an agent. 
    - Updates agent state to PROCESSING during `_perform_interaction`.
    - If canceled mid-stream, stops yielding text and sets message status to CANCELLED.
    - If is_task=True, auto-executes tool calls after streaming ends (unless canceled).
    """

    def __init__(
        self,
        name: str,
        role: str,
        description: str,
        model_name: str,
        tools=None,
        executor=None
    ):
        super().__init__(
            name=name,
            role=role,
            description=description,
            tools=tools,
            executor=executor
        )

        self.llm_client = OpenAIClient(model_name)
        
        self.model_name = model_name
        
    def cancel(self):
        self.llm_client.cancel()
        return super().cancel()

    def _perform_interaction(
        self,
        user_message: str,
        force_tools: bool,
        stateless: bool = False,
        **kwargs
    ) -> Iterator[Any]:
        """
        Returns a streaming iterator of text. If canceled mid-stream, stops immediately.
        At the end, runs tool calls in parallel.
        """
        self._start_run()


        # 1. Create an "assistant" conversation item with empty content, 
        #    which we'll fill as we stream text.
        assistant_item = ConversationItem(role="assistant", content="", status=MessageStatus.IN_PROGRESS)
        
        # 2. Add the user message to conversation_history (COMPLETED by default)
        if not stateless:
            user_item = ConversationItem(role="user", content=user_message, status=MessageStatus.COMPLETED)
            self.conversation_history.append(user_item)

        # 3. Build LLM messages from conversation history
        llm_messages = []
        for msg in self.conversation_history:
            llm_messages.append(LLMMessage(role=msg.role, content=msg.content))
            
        # logger.info(f"Agent {self.name} starting interaction with user message: {user_message}, llm_messages: {llm_messages}")

        # 4. Call the LLM in streaming mode
        llm_tools = self.tools if force_tools else None
        llm_result = self.llm_client.predict(
            messages=llm_messages,
            tools=llm_tools,
            stream=True,  # always streaming
            response_format=None  # or pass from kwargs if needed
        )
        
        if not stateless:
            self.conversation_history.append(assistant_item)
        
        # We expect a StreamingResult
        if not isinstance(llm_result, StreamingResult):
            logging.warning("LLM client did not return StreamingResult! Behavior may be undefined.")
            yield "[ERROR: Non-streaming result received]"
            self._end_run()
            return

        # 5. We collect partial text and any tool calls while streaming
        partial_text = []
        all_tool_calls: List[LLMToolCall] = []

        # We'll combine text_iter and tool_calls_iter
        text_iter = llm_result.text or iter([])
        calls_iter = llm_result.tool_calls or iter([])

        for text_chunk, tool_chunk in zip_longest(text_iter, calls_iter, fillvalue=None):
            # logger.info(f"Agent {self.name} streaming text chunk: {text_chunk}, tool chunk: {tool_chunk}")
            with self.lock:
                if self._cancel_requested:
                    # Mark assistant item as CANCELLED
                    assistant_item.status = MessageStatus.CANCELLED
                    self._end_run()
                    yield None
                    return

            # If there's partial text:
            if text_chunk is not None:
                partial_text.append(text_chunk)
                # Append to the assistant item content
                assistant_item.content += text_chunk
                # Yield to the user (streaming)
                yield text_chunk

            # If there's a tool chunk:
            if tool_chunk is not None and len(tool_chunk) > 0:
                all_tool_calls.extend(tool_chunk)

        # 6. Streaming ended normally (not canceled)
        # Finalize the assistant message content
        assistant_item.status = MessageStatus.COMPLETED
        assistant_item.content += f" Tool calls: {[{"id": i.id, "name": i.name, "arguments": json.dumps(i.arguments)} for i in all_tool_calls]}"

        # 7. If is_task=True and we have tool calls => auto execute them in parallel
        tool_results = []
        with self.lock:
            canceled_now = self._cancel_requested

        logger.info(f"tool_calls: {all_tool_calls}, canceled_now: {canceled_now}")
        if all_tool_calls and not canceled_now:
            tool_results = self._auto_execute_tool_calls(all_tool_calls)
            # Optionally yield the tool results
            yield tool_results

        # 8. End the run
        self._end_run()
        
    def _start_run(self):
        """
        Set agent state to PROCESSING when starting a new run
        """
        with self.lock:
            # print(f"Agent {self.name} starting run.", flush=True)
            logger.info(f"Agent {self.name} starting run.")
            self.state = AgentState.PROCESSING
            self._cancel_requested = False

    def _end_run(self):
        """
        Set agent state to IDLE after finishing or canceling
        """
        with self.lock:
            # print(f"Agent {self.name} ending run.", flush=True)
            logger.info(f"Agent {self.name} ending run.")
            self.state = AgentState.IDLE

    def _auto_execute_tool_calls(self, tool_calls: List[LLMToolCall]) -> List[Any]:
        """
        Runs each tool call in parallel, returns a list of the results in the same order.
        """
        futures: List[Future] = []
        for tc in tool_calls:
            future = self.executor.submit(tc.function, **tc.arguments)
            futures.append(future)
        results = [f.result() for f in futures]
        return results
