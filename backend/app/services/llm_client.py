"""
LLM Client Adapters for Conversational AI Application

This module defines adapter classes for supported LLM providers:
- Google Gemini
- Groq (Llama3)

Each adapter implements the BaseLLMClient interface for consistent API routing.
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

# Conditional imports (no failure on missing SDKs)
try:
    from groq import AsyncGroq, GroqError
except ImportError:
    AsyncGroq = None
    GroqError = None

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold, StopCandidateException
    from google.api_core import exceptions as GoogleAPIErrors
except ImportError:
    genai = None
    HarmCategory = HarmBlockThreshold = StopCandidateException = GoogleAPIErrors = None

from app.config import Settings
from app.models.schemas import Message, StreamToken, UsageStats, FinishReason

logger = logging.getLogger(__name__)

# Type aliases for return types
ChatReturnType = Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]

# --------------------------
# Custom Exceptions
# --------------------------

class LLMClientError(Exception):
    """Base class for LLM client errors."""
    pass

class LLMConnectionError(LLMClientError):
    """Raised when connection to LLM provider fails."""
    pass

class LLMRequestError(LLMClientError):
    """Raised when a request to the LLM fails."""
    pass

# --------------------------
# LLM Interface
# --------------------------

class LLMInterface(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        pass

class OpenAIAdapter(LLMInterface):
    def __init__(self, model="gpt-4"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")

    async def generate(self, prompt: str) -> str:
        # Simulate API call to OpenAI
        return f"[OpenAI {self.model}] Response to: {prompt}"

class GeminiAdapter(LLMInterface):
    def __init__(self, model="gemini"):
        self.model = model
        self.api_key = os.getenv("GEMINI_API_KEY")

    async def generate(self, prompt: str) -> str:
        # Simulate API call to Gemini
        return f"[Gemini {self.model}] Response to: {prompt}"

class ClaudeAdapter(LLMInterface):
    def __init__(self, model="claude"):
        self.model = model
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    async def generate(self, prompt: str) -> str:
        # Simulate API call to Claude
        return f"[Claude {self.model}] Response to: {prompt}"

def get_active_model() -> LLMInterface:
    active_model = os.getenv("ACTIVE_LLM", "gpt-4")
    if active_model == "gpt-4":
        return OpenAIAdapter()
    elif active_model == "gemini":
        return GeminiAdapter()
    elif active_model == "claude":
        return ClaudeAdapter()
    else:
        raise ValueError(f"Unsupported LLM model: {active_model}")

# --------------------------
# Base LLM Client Interface
# --------------------------

class BaseLLMClient(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs: Any
    ) -> ChatReturnType:
        pass

# --------------------------
# Groq Adapter
# --------------------------

class GroqAdapter(BaseLLMClient):
    def __init__(self, api_key: str, model: str):
        if AsyncGroq is None:
            raise ImportError("Groq SDK ('groq') is required but not installed.")
        try:
            self.client = AsyncGroq(api_key=api_key)
            self.model = model
            logger.info(f"GroqAdapter initialized for {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}", exc_info=True)
            raise LLMConnectionError(f"Groq client initialization failed: {e}") from e

    async def _process_stream(self, response_stream: AsyncGenerator) -> AsyncGenerator:
        collected_content = ""
        final_finish_reason: Optional[FinishReason] = "stop"
        final_usage = None

        try:
            async for chunk in response_stream:
                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.finish_reason:
                        final_finish_reason = choice.finish_reason
                    if choice.delta and choice.delta.content:
                        token = choice.delta.content
                        collected_content += token
                        yield StreamToken(token=token, is_finished=False).model_dump()
                if chunk.usage:
                    final_usage = UsageStats(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens
                    ).model_dump()
        except GroqError as e:
            logger.error(f"Error during Groq stream processing: {e}", exc_info=True)
            yield StreamToken(token=f"Groq Error: {e}", is_finished=True, finish_reason="error").model_dump()
            return
        except Exception as e:
            logger.error(f"Unexpected error during Groq stream: {e}", exc_info=True)
            yield StreamToken(token=f"Error: {e}", is_finished=True, finish_reason="error").model_dump()
            return

        yield StreamToken(
            token="",
            is_finished=True,
            finish_reason=final_finish_reason,
            content=collected_content,
            usage=final_usage
        ).model_dump()

    async def chat(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs: Any
    ) -> ChatReturnType:
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", None)

        groq_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        try:
            if stream:
                logger.debug(f"Requesting Groq stream: {self.model}, temp={temperature}, max_tokens={max_tokens}")
                response_stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=groq_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True
                )
                return self._process_stream(response_stream)

            logger.debug(f"Requesting Groq non-stream: {self.model}, temp={temperature}, max_tokens={max_tokens}")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=groq_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )

            assistant_content = response.choices[0].message.content or ""
            usage_stats = UsageStats(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            ).model_dump() if response.usage else None

            return {
                "message": Message(role="assistant", content=assistant_content).model_dump(),
                "usage": usage_stats
            }

        except GroqError as e:
            logger.error(f"Groq API call failed: {e}", exc_info=True)
            raise LLMConnectionError(f"Groq API error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected Groq error: {e}", exc_info=True)
            raise LLMRequestError(f"Groq adapter error: {e}") from e

# --------------------------
# Gemini Adapter
# --------------------------

class GeminiAdapter(BaseLLMClient):
    DEFAULT_SAFETY_SETTINGS = {}

    def __init__(self, api_key: str, model: str):
        if genai is None:
            raise ImportError("Google Generative AI SDK ('google-generativeai') is required but not installed.")
        try:
            genai.configure(api_key=api_key)
            self.model_name = model
            self.safety_settings = self.DEFAULT_SAFETY_SETTINGS
            logger.info(f"GeminiAdapter initialized for {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to configure Gemini client: {e}", exc_info=True)
            raise LLMConnectionError(f"Gemini client configuration failed: {e}") from e

    @staticmethod
    def _convert_messages(messages: List[Message]) -> tuple[str, list]:
        system_prompt_parts = []
        history = []

        for msg in messages:
            if msg.role == "system":
                system_prompt_parts.append(msg.content)
            else:
                role = "user" if msg.role == "user" else "model"
                history.append({"role": role, "parts": [msg.content]})

        system_prompt = "\n\n".join(system_prompt_parts) if system_prompt_parts else None
        return system_prompt, history

    async def _process_stream(self, response_stream: Any) -> AsyncGenerator:
        collected_content = ""
        final_finish_reason: Optional[FinishReason] = "stop"

        try:
            loop = asyncio.get_running_loop()
            def get_next_chunk():
                try:
                    return next(response_stream)
                except StopIteration:
                    return None

            while True:
                chunk = await loop.run_in_executor(None, get_next_chunk)
                if chunk is None:
                    break

                try:
                    if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback.block_reason:
                        reason = chunk.prompt_feedback.block_reason.name
                        logger.warning(f"Gemini stream blocked: {reason}")
                        final_finish_reason = "error"
                        yield StreamToken(token=f"[Blocked: {reason}]", is_finished=True, finish_reason=final_finish_reason).model_dump()
                        return

                    if chunk.candidates:
                        cand = chunk.candidates[0]
                        if cand.finish_reason:
                            final_finish_reason = cand.finish_reason.name.lower()

                        if cand.content and cand.content.parts:
                            token = cand.content.parts[0].text
                            collected_content += token
                            yield StreamToken(token=token, is_finished=False).model_dump()
                except StopCandidateException as sce:
                    logger.warning(f"Stream stopped by StopCandidateException: {sce.finish_reason}")
                    final_finish_reason = "error"
                    yield StreamToken(token=f"[Blocked: {sce.finish_reason}]", is_finished=True, finish_reason=final_finish_reason).model_dump()
                    return
                except Exception as chunk_exc:
                    logger.error(f"Error processing Gemini stream chunk: {chunk_exc}", exc_info=True)
                    yield StreamToken(token=f"Error: {chunk_exc}", is_finished=True, finish_reason="error").model_dump()
                    return

        except Exception as e:
            logger.error(f"Unexpected error in Gemini stream wrapper: {e}", exc_info=True)
            yield StreamToken(token=f"Error: {e}", is_finished=True, finish_reason="error").model_dump()
            return

        yield StreamToken(
            token="",
            is_finished=True,
            finish_reason=final_finish_reason,
            content=collected_content,
            usage=None
        ).model_dump()

    async def chat(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs: Any
    ) -> ChatReturnType:
        system_prompt, history = self._convert_messages(messages)
        temperature = kwargs.get("temperature", 0.9)
        max_tokens = kwargs.get("max_tokens", None)

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                },
                safety_settings=self.safety_settings
            )
            chat_session = model.start_chat(history=history)

            # Get last user message
            last_user_message = [msg.content for msg in messages if msg.role == "user"][-1]

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: chat_session.send_message(last_user_message, stream=stream)
            )

            if stream:
                return self._process_stream(response)

            # Non-streaming response
            assistant_content = ""
            if response.candidates and response.candidates[0].content:
                assistant_content = response.candidates[0].content.parts[0].text or ""

            return {
                "message": Message(role="assistant", content=assistant_content).model_dump(),
                "usage": None  # Gemini doesn't return usage in non-streaming
            }

        except StopCandidateException as e:
            logger.warning(f"Gemini stream stopped by StopCandidateException: {e.finish_reason}")
            return {
                "message": Message(role="assistant", content=f"[Blocked due to: {e.finish_reason}]").model_dump(),
                "usage": None
            }
        except GoogleAPIErrors.GoogleAPIError as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            raise LLMConnectionError(f"Gemini API failure: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected Gemini error: {e}", exc_info=True)
            raise LLMRequestError(f"Gemini adapter error: {e}") from e

# --------------------------
# LLM Client Factory
# --------------------------

def get_llm_client(settings: Settings) -> BaseLLMClient:
    """
    Factory function to return the appropriate LLM client based on settings.

    Args:
        settings: Application settings object

    Returns:
        BaseLLMClient: An instance of the selected LLM provider adapter

    Raises:
        ImportError: If the selected provider's SDK is missing
        ValueError: If the selected provider has no API key
    """
    provider = settings.llm_provider
    logger.info(f"Creating LLM client for provider: {provider.value}")

    try:
        if provider == LLMProvider.GROQ:
            if AsyncGroq is None:
                logger.warning("Groq SDK not installed. Falling back to Gemini.")
                return get_llm_client(GeminiAdapter(...))
            if not settings.groq_api_key:
                logger.warning(f"Groq provider selected but {ENV_GROQ_API_KEY} is missing.")
                logger.info("Falling back to Gemini provider.")
                return get_llm_client(GeminiAdapter(...))
            return GroqAdapter(
                api_key=settings.groq_api_key.get_secret_value(),
                model=settings.groq_model
            )
        elif provider == LLMProvider.GEMINI:
            if genai is None:
                logger.warning("Gemini SDK not installed. Falling back to Groq.")
                return get_llm_client(GroqAdapter(...))
            if not settings.gemini_api_key:
                logger.warning(f"Gemini provider selected but {ENV_GEMINI_API_KEY} is missing.")
                logger.info("Falling back to Groq provider.")
                return get_llm_client(GroqAdapter(...))
            return GeminiAdapter(
                api_key=settings.gemini_api_key.get_secret_value(),
                model=settings.gemini_model
            )
        else:
            raise ValueError(f"Unsupported LLM provider configured: {provider.value}")
    except Exception as e:
        logger.critical(f"Failed to instantiate LLM client: {e}", exc_info=True)
        raise