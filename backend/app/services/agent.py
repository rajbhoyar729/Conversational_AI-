"""
Conversational Agent Module for Conversational AI Application

This module defines the ConversationalAgent class, which acts as a stateless intermediary
between the API layer and the underlying LLM client implementations.

Responsibilities:
- Validate basic input structure
- Sanitize user input (e.g., escape HTML)
- Prepare LLM parameters
- Call LLM client and return result
- Handle provider-specific errors consistently

It explicitly does *not*:
- Manage conversation history
- Handle HTTP formatting
- Implement provider-specific logic
"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
from html import escape  # Input sanitization

from app.models.schemas import Message, StreamToken, ChatResponse
from app.services.llm_client import BaseLLMClient, ChatReturnType

logger = logging.getLogger(__name__)


# --------------------------
# Custom Exceptions
# --------------------------

class LLMConnectionError(Exception):
    """Raised when connection to LLM provider fails."""
    pass


class LLMRequestError(Exception):
    """Raised when an LLM request fails due to invalid input or internal error."""
    pass


# --------------------------
# Conversational Agent
# --------------------------

class ConversationalAgent:
    """
    A stateless service acting as an intermediary between the API layer and
    the underlying LLM client implementations.

    Responsibilities:
    - Receiving conversation context (messages) and processing parameters.
    - Validating basic inputs specific to the agent's role.
    - Preparing LLM parameters (temperature, max_tokens).
    - Delegating to the LLM client for actual model interaction.
    - Returning structured responses or stream generators.

    Raises:
    - LLMConnectionError: If LLM provider is unreachable.
    - LLMRequestError: If input is invalid or internal error occurs.
    """

    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize the ConversationalAgent.

        Args:
            llm_client: An instance of BaseLLMClient for LLM communication.

        Raises:
            TypeError: If the provided client doesn't conform to BaseLLMClient interface.
        """
        if not isinstance(llm_client, BaseLLMClient):
            err_msg = (
                f"Invalid llm_client provided. Expected BaseLLMClient, "
                f"got {type(llm_client).__name__}."
            )
            logger.critical(err_msg)
            raise TypeError(err_msg)

        self.llm_client = llm_client
        logger.info(f"Initialized ConversationalAgent with LLM client: {type(self.llm_client).__name__}")

    async def process_message(
        self,
        messages: List[Message],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **llm_specific_kwargs: Any
    ) -> ChatReturnType:
        """
        Process a list of messages using the injected LLM client.

        Args:
            messages: List of Message objects representing the conversation.
            stream: Whether to return a token stream.
            temperature: Controls randomness in output (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            **llm_specific_kwargs: Provider-specific parameters (e.g., stop_sequences).

        Returns:
            - Dict for non-streaming responses.
            - AsyncGenerator for streaming responses.

        Raises:
            LLMRequestError: On invalid input or unexpected internal errors.
            LLMConnectionError: On failed connection to LLM provider.
        """
        if not messages:
            logger.error("Received empty messages list.")
            raise LLMRequestError("Cannot process chat: 'messages' list must not be empty.")

        # Ensure at least one user message exists
        if not any(msg.role == "user" for msg in messages):
            logger.warning("No user message found in chat history.")
            raise LLMRequestError("At least one user message is required in the chat history.")

        # Sanitize all message content
        sanitized_messages = [
            Message(role=msg.role, content=escape(msg.content))  # Prevents XSS
            for msg in messages
        ]

        # Build core LLM parameters
        core_params: Dict[str, Any] = {
            k: v for k, v in {
                "temperature": temperature,
                "max_tokens": max_tokens
            }.items() if v is not None
        }

        # Warn if max_tokens is non-positive
        if max_tokens is not None and max_tokens <= 0:
            logger.warning(f"Received non-positive max_tokens ({max_tokens}). Ignoring parameter.")
            core_params.pop("max_tokens", None)

        # Merge with provider-specific parameters
        final_params = {**core_params, **llm_specific_kwargs}

        log_params_str = ", ".join(f"{k}={v}" for k, v in final_params.items())
        logger.debug(
            f"Processing chat: {len(sanitized_messages)} messages, stream={stream}. "
            f"LLM Client: {type(self.llm_client).__name__}. Params: {log_params_str}"
        )

        try:
            # Delegate to LLM client
            response = await self.llm_client.chat(
                messages=sanitized_messages,
                stream=stream,
                **final_params
            )
            return response

        except ConnectionError as ce:
            logger.error(f"Connection to LLM provider failed: {ce}", exc_info=True)
            raise LLMConnectionError(f"Failed to connect to LLM provider: {ce}") from ce

        except (ValueError, RuntimeError) as ve:
            logger.error(f"LLM request failed due to invalid input or runtime error: {ve}", exc_info=True)
            raise LLMRequestError(f"LLM request failed: {ve}") from ve

        except Exception as e:
            logger.error(f"Unexpected error during LLM client interaction: {e}", exc_info=True)
            raise LLMRequestError(f"Unexpected error during chat processing: {e}") from e


# --------------------------
# Conversation Buffer
# --------------------------

class ConversationBuffer:
    def __init__(self):
        self.history = []

    def add(self, user_input: str, response: str):
        """Add a user input and corresponding response to the history."""
        self.history.append({"user": user_input, "bot": response})

    def get_history(self) -> str:
        """Retrieve the conversation history as a formatted string."""
        return "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in self.history])


# --------------------------
# AI Agent
# --------------------------

class AIAgent:
    def __init__(self, llm):
        """Initialize the AI agent with an LLM interface and a conversation buffer."""
        self.memory = ConversationBuffer()
        self.llm = llm

    async def respond(self, user_input: str) -> str:
        """Generate a response based on user input and conversation history."""
        context = self.memory.get_history()
        prompt = f"{context}\nUser: {user_input}"
        response = await self.llm.generate(prompt)
        self.memory.add(user_input, response)
        return response