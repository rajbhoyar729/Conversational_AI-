import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, cast

from app.models import Message, StreamToken
from app.services.llm_client import BaseLLMClient, ChatReturnType

logger = logging.getLogger(__name__)

class AgentProcessingError(Exception):
    """Custom exception for errors originating within the agent's logic."""
    pass

class ConversationalAgent:
    """
    A stateless service acting as an intermediary between the API layer and
    the underlying LLM client implementations.

    Responsibilities:
    - Receiving conversation context (messages) and processing parameters.
    - Validating basic inputs specific to the agent's role.
    - Preparing parameters for the LLM client call.
    - Delegating the actual LLM interaction to an injected `BaseLLMClient` instance.
    - Propagating results (responses or stream generators) and errors back to the caller.

    It explicitly **does not**:
    - Manage conversation history state.
    - Directly handle API request/response formatting (e.g., HTTP errors).
    - Implement LLM provider-specific logic (this belongs in `llm_client` adapters).
    """

    def __init__(self, llm_client: BaseLLMClient):
        """
        Initializes the ConversationalAgent.

        Args:
            llm_client: A concrete implementation of the BaseLLMClient interface,
                        responsible for actual LLM communication.

        Raises:
            TypeError: If the provided llm_client is not an instance of BaseLLMClient
                        (or does not conform to its protocol if using runtime_checkable).
        """
        if not isinstance(llm_client, BaseLLMClient):
            err_msg = f"Invalid llm_client provided. Expected BaseLLMClient, got {type(llm_client).__name__}."
            logger.critical(err_msg)
            raise TypeError(err_msg)

        self.llm_client = llm_client
        logger.info(f"ConversationalAgent initialized with LLM client: {type(llm_client).__name__}")

    async def process_message(
        self,
        messages: List[Message],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **llm_specific_kwargs: Any
    ) -> ChatReturnType:
        """
        Processes a list of messages using the injected LLM client.

        Args:
            messages: The list of Message objects. Must not be empty.
            stream: Flag indicating if streaming output is required.
            temperature: Optional temperature setting for the LLM.
            max_tokens: Optional maximum token limit for the LLM response.
            **llm_specific_kwargs: Any additional keyword arguments to be passed
                                    directly to the underlying `llm_client.chat` method
                                    (e.g., `top_p`, `stop_sequences`).

        Returns:
            The result from the `llm_client.chat` call, which is either a dictionary
            (non-streaming) or an async generator yielding dictionaries (streaming).

        Raises:
            ValueError: If the `messages` list is empty.
            AgentProcessingError: For errors originating within the agent's logic (rare).
            Exception: Propagates exceptions raised by the `llm_client` (e.g.,
                        ConnectionError, API errors, SDK-specific errors, RuntimeError).
                        The caller (API layer) is responsible for handling these.
        """
        if not messages:
            logger.error("process_message called with empty 'messages' list.")
            raise ValueError("Cannot process chat: 'messages' list must not be empty.")

        core_params: Dict[str, Any] = {}
        if temperature is not None:
            core_params["temperature"] = temperature
        if max_tokens is not None:
            if max_tokens <= 0:
                logger.warning(f"Received non-positive max_tokens ({max_tokens}). Ignoring.")
            else:
                core_params["max_tokens"] = max_tokens


        final_llm_params = {**core_params, **llm_specific_kwargs}

        log_params_str = ", ".join(f"{k}={v}" for k, v in final_llm_params.items())
        logger.info(
            f"Processing chat request: {len(messages)} messages, stream={stream}. "
            f"LLM Client: {type(self.llm_client).__name__}. Params: {log_params_str}"
        )

        try:
            response_or_generator = await self.llm_client.chat(
                messages=messages,
                stream=stream,
                **final_llm_params
            )
            logger.info(f"LLM client call successful for stream={stream}.")
            return response_or_generator

        except Exception as e:
            logger.error(
                f"Exception raised by LLM client ({type(self.llm_client).__name__}) during chat processing: {type(e).__name__} - {e}",
                exc_info=True
            )
            raise
