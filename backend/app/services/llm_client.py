# backend/app/services/llm_client.py

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, TypeAlias

try:
    from groq import AsyncGroq, GroqError
except ImportError:
    AsyncGroq = None # type: ignore
    GroqError = None # type: ignore
    logger.warning("Groq SDK not installed. Groq provider will not be available.")

try:
    import google.generativeai as genai
    from google.generativeai.types import (HarmCategory, HarmBlockThreshold,
                                            StopCandidateException)
    from google.api_core import exceptions as GoogleAPIErrors
except ImportError:
    genai = None # type: ignore
    HarmCategory = None # type: ignore
    HarmBlockThreshold = None # type: ignore
    StopCandidateException = None # type: ignore
    GoogleAPIErrors = None # type: ignore
    logger.warning("Google Generative AI SDK not installed. Gemini provider will not be available.")


from app.config import LLMProvider, Settings, get_settings
from app.models import Message, StreamToken, UsageStats, FinishReason

logger = logging.getLogger(__name__)

ChatReturnType: TypeAlias = Dict[str, Any] | AsyncGenerator[Dict[str, Any], None]

class BaseLLMClient(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs: Any
    ) -> ChatReturnType:
        pass

class GroqAdapter(BaseLLMClient):
    def __init__(self, api_key: str, model: str):
        if AsyncGroq is None:
            raise ImportError("Groq SDK ('groq') is required but not installed.")
        try:
            self.client = AsyncGroq(api_key=api_key)
            self.model = model
            logger.info(f"GroqAdapter initialized successfully for model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}", exc_info=True)
            raise RuntimeError(f"Groq client initialization failed: {e}") from e

    async def _process_stream(self, response_stream: AsyncGenerator) -> AsyncGenerator[Dict[str, Any], None]:
        collected_content = ""
        final_usage = None
        final_finish_reason: Optional[FinishReason] = None

        try:
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].finish_reason:
                    final_finish_reason = chunk.choices[0].finish_reason # type: ignore

                token = ""
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    collected_content += token

                if chunk.usage:
                    final_usage = UsageStats(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    ).model_dump()

                if token:
                    yield StreamToken(token=token, is_finished=False).model_dump()

        except GroqError as e:
            logger.error(f"Error during Groq stream processing: {e}", exc_info=True)
            yield StreamToken(token=f"Groq Error: {e}", is_finished=True, finish_reason="error").model_dump()
            return
        except Exception as e:
            logger.error(f"Unexpected error processing Groq stream: {e}", exc_info=True)
            yield StreamToken(token=f"Error: {e}", is_finished=True, finish_reason="error").model_dump()
            return

        yield StreamToken(
            token="",
            is_finished=True,
            finish_reason=final_finish_reason or "stop",
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

        groq_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        try:
            if stream:
                logger.debug(f"Requesting Groq stream: model={self.model}, temp={temperature}, max_tokens={max_tokens}")
                response_stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=groq_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                return self._process_stream(response_stream)
            else:
                logger.debug(f"Requesting Groq non-stream: model={self.model}, temp={temperature}, max_tokens={max_tokens}")
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=groq_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )

                assistant_content = ""
                if response.choices:
                    assistant_content = response.choices[0].message.content or ""

                usage_stats = None
                if response.usage:
                    usage_stats = UsageStats(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )

                return {
                    "message": Message(role="assistant", content=assistant_content).model_dump(),
                    "usage": usage_stats.model_dump() if usage_stats else None,
                }

        except GroqError as e:
            logger.error(f"Groq API call failed: {e}", exc_info=True)
            raise ConnectionError(f"Groq API error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during Groq chat: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected Groq adapter error: {e}") from e


class GeminiAdapter(BaseLLMClient):
    DEFAULT_SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    } if HarmCategory and HarmBlockThreshold else {}

    def __init__(self, api_key: str, model: str):
        if genai is None or HarmCategory is None:
            raise ImportError("Google Generative AI SDK ('google-generativeai') is required but not installed.")
        try:
            genai.configure(api_key=api_key)
            self.model_name = model
            self.safety_settings = self.DEFAULT_SAFETY_SETTINGS
            logger.info(f"GeminiAdapter initialized successfully for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to configure or initialize Gemini client: {e}", exc_info=True)
            raise RuntimeError(f"Gemini client configuration/initialization failed: {e}") from e

    @staticmethod
    def _convert_messages(messages: List[Message]) -> List[Dict[str, Any]]:
        gemini_history = []
        system_prompt = None
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
                continue

            role = "user" if msg.role == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg.content]})

        return gemini_history

    async def _process_stream(self, response_stream_sync: Any) -> AsyncGenerator[Dict[str, Any], None]:
        collected_content = ""
        final_finish_reason: Optional[FinishReason] = "stop"

        try:
            loop = asyncio.get_running_loop()
            def get_next_chunk():
                try:
                    return next(response_stream_sync)
                except StopIteration:
                    return None

            while True:
                chunk = await loop.run_in_executor(None, get_next_chunk)
                if chunk is None:
                    break

                token = ""
                try:
                    if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback.block_reason:
                        reason = chunk.prompt_feedback.block_reason.name
                        logger.warning(f"Gemini stream blocked due to prompt feedback: {reason}")
                        final_finish_reason = "error"
                        yield StreamToken(token=f"Error: Prompt blocked ({reason})", is_finished=True, finish_reason=final_finish_reason).model_dump()
                        return

                    if chunk.candidates:
                        cand = chunk.candidates[0]
                        if cand.finish_reason:
                            reason_str = cand.finish_reason.name.lower()
                            if reason_str == "stop": final_finish_reason = "stop"
                            elif reason_str == "max_tokens": final_finish_reason = "length"
                            elif reason_str in ["safety", "recitation", "other"]: final_finish_reason = "error"
                            else: final_finish_reason = "stop"

                    if hasattr(chunk, 'text'):
                        token = chunk.text
                        collected_content += token

                except StopCandidateException as sce:
                    logger.warning(f"Gemini stream stopped by StopCandidateException: {sce.finish_reason}")
                    final_finish_reason = "error"
                    yield StreamToken(token=f"Error: Stream stopped ({sce.finish_reason})", is_finished=True, finish_reason=final_finish_reason).model_dump()
                    return
                except ValueError as ve:
                    logger.warning(f"Skipping potentially problematic Gemini stream chunk: {ve}")
                    continue
                except Exception as chunk_error:
                    logger.error(f"Error processing Gemini stream chunk content: {chunk_error}", exc_info=True)
                    final_finish_reason = "error"
                    yield StreamToken(token=f"Error processing chunk: {chunk_error}", is_finished=True, finish_reason=final_finish_reason).model_dump()
                    return

                if token:
                    yield StreamToken(token=token, is_finished=False).model_dump()

        except Exception as e:
            logger.error(f"Unexpected error processing Gemini stream wrapper: {e}", exc_info=True)
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
        temperature = kwargs.get("temperature", 0.9)
        max_tokens = kwargs.get("max_tokens", None)

        generation_config: Dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens

        system_prompt_content = None
        gemini_formatted_messages = []
        for msg in messages:
            if msg.role == "system":
                system_prompt_content = msg.content
            else:
                role = "user" if msg.role == "user" else "model"
                gemini_formatted_messages.append({"role": role, "parts": [msg.content]})

        history_for_chat_start = []
        last_message_parts = [" "]
        if gemini_formatted_messages:
            if gemini_formatted_messages[-1]["role"] == "user":
                last_message_parts = gemini_formatted_messages[-1]["parts"]
                history_for_chat_start = gemini_formatted_messages[:-1]
            else:
                history_for_chat_start = gemini_formatted_messages
                logger.warning("Last message in history is 'assistant'. Sending empty prompt to Gemini.")
                last_message_parts = [" "]

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config, # type: ignore
                safety_settings=self.safety_settings,
                system_instruction=system_prompt_content if system_prompt_content else None
            )

            chat_session = model.start_chat(history=history_for_chat_start)

            loop = asyncio.get_running_loop()
            def send_message_sync():
                return chat_session.send_message(
                    content=last_message_parts,
                    stream=stream
                )

            logger.debug(f"Sending request to Gemini: model={self.model_name}, stream={stream}, temp={temperature}, max_tokens={max_tokens}")
            response_or_stream = await loop.run_in_executor(None, send_message_sync)

            if stream:
                return self._process_stream(response_or_stream)
            else:
                assistant_content = ""
                usage_stats = None
                finish_reason_str = "stop"

                if hasattr(response_or_stream, 'text'):
                    assistant_content = response_or_stream.text
                elif response_or_stream.candidates and response_or_stream.candidates[0].content:
                    assistant_content = response_or_stream.candidates[0].content.parts[0].text

                if hasattr(response_or_stream, 'prompt_feedback') and response_or_stream.prompt_feedback.block_reason:
                    reason = response_or_stream.prompt_feedback.block_reason.name
                    logger.warning(f"Gemini non-stream response blocked due to prompt: {reason}")
                    assistant_content = f"[Blocked due to Prompt Feedback: {reason}]"
                    finish_reason_str = "error"

                if response_or_stream.candidates and response_or_stream.candidates[0].finish_reason:
                    finish_reason_str = response_or_stream.candidates[0].finish_reason.name.lower()
                    if finish_reason_str not in ["stop", "max_tokens"]:
                        logger.warning(f"Gemini non-stream response finished due to: {finish_reason_str}")
                        assistant_content += f" [Response stopped due to: {finish_reason_str}]"

                return {
                    "message": Message(role="assistant", content=assistant_content).model_dump(),
                    "usage": usage_stats.model_dump() if usage_stats else None,
                }

        except StopCandidateException as e:
            logger.warning(f"Gemini request stopped by StopCandidateException: {e.finish_reason}")
            return {
                "message": Message(role="assistant", content=f"[Blocked due to: {e.finish_reason}]").model_dump(),
                "usage": None
            }
        except (GoogleAPIErrors.GoogleAPIError, Exception) as e:
            logger.error(f"Gemini API call failed: {e}", exc_info=True)
            raise ConnectionError(f"Gemini API error: {e}") from e


from functools import lru_cache

@lru_cache()
def get_llm_client(settings: Settings) -> BaseLLMClient:
    provider = settings.llm_provider
    logger.info(f"Creating LLM client for provider: {provider.value}")

    try:
        if provider == LLMProvider.GROQ:
            if not settings.groq_api_key:
                raise ValueError(f"Groq API key is required but not found in settings.")
            return GroqAdapter(
                api_key=settings.groq_api_key.get_secret_value(),
                model=settings.groq_model
            )
        elif provider == LLMProvider.GEMINI:
            if not settings.gemini_api_key:
                raise ValueError(f"Gemini API key is required but not found in settings.")
            return GeminiAdapter(
                api_key=settings.gemini_api_key.get_secret_value(),
                model=settings.gemini_model
            )
        else:
            raise ValueError(f"Unsupported LLM provider configured: {provider.value}")

    except ImportError as e:
        logger.critical(f"Missing required SDK for provider '{provider.value}'. Please install it.", exc_info=True)
        raise ImportError(f"SDK required for '{provider.value}' is
