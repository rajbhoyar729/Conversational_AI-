import logging
from typing import List, Dict, Any, AsyncGenerator, Optional, Protocol, runtime_checkable

from app.models.schemas import Message, StreamToken

logger = logging.getLogger(__name__)

@runtime_checkable
class BaseLLMClient(Protocol):
    async def chat(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any] | AsyncGenerator[Dict[str, Any], None]:
        ...

class ConversationalAgent:
    def __init__(self, llm_client: BaseLLMClient):
        if not isinstance(llm_client, BaseLLMClient):
            raise TypeError(
                f"llm_client of type {type(llm_client).__name__} does not conform to BaseLLMClient"
            )
        self.llm_client = llm_client
        logger.info(f"ConversationalAgent initialized with {type(llm_client).__name__}")

    async def process_message(
        self,
        messages: List[Message],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, Any] | AsyncGenerator[Dict[str, Any], None]:
        if not messages:
            logger.warning("process_message called with empty messages list.")
            raise ValueError("Input messages list cannot be empty.")

        llm_params = {**kwargs}
        if temperature is not None:
            llm_params["temperature"] = temperature
        if max_tokens is not None:
            llm_params["max_tokens"] = max_tokens

        logger.debug(
            f"Processing {len(messages)} messages, stream={stream}, params={llm_params}"
        )

        try:
            response = await self.llm_client.chat(
                messages=messages,
                stream=stream,
                **llm_params
            )
            logger.debug(f"LLM client call successful (stream={stream}).")
            return response
        except Exception:
            logger.error(
                "Error during llm_client.chat call", exc_info=True
            )
            raise
