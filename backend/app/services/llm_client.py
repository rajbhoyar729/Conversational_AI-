import asyncio
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Any, Optional

from app.config import LLMProvider, Settings
from app.models.schemas import Message, StreamToken

logger = logging.getLogger(__name__)

class BaseLLMClient(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any] | AsyncGenerator[Dict[str, Any], None]:
        pass

class GroqAdapter(BaseLLMClient):
    def __init__(self, api_key: str, model: str):
        from groq import AsyncGroq
        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        logger.info(f"GroqAdapter initialized with model: {self.model}")

    async def _stream_response_generator(
        self, response_stream: AsyncGenerator
    ) -> AsyncGenerator[Dict[str, Any], None]:
        collected_content = ""
        finish_reason = None
        try:
            async for chunk in response_stream:
                finish_reason = chunk.choices[0].finish_reason
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    token_text = chunk.choices[0].delta.content
                    collected_content += token_text
                    yield StreamToken(
                        token=token_text,
                        is_finished=False,
                        finish_reason=None
                    ).model_dump()
            final_finish_reason = finish_reason if finish_reason else "stop"
            yield StreamToken(
                token="",
                is_finished=True,
                finish_reason=final_finish_reason
            ).model_dump()
        except Exception as e:
            logger.error(f"Error processing Groq stream chunk: {e}", exc_info=True)
            yield StreamToken(
                token=f"Error processing stream: {e}",
                is_finished=True,
                finish_reason="error"
            ).model_dump()

    async def chat(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any] | AsyncGenerator[Dict[str, Any], None]:
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", None)
        groq_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        from groq import GroqError
        try:
            if stream:
                response_stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=groq_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True
                )
                return self._stream_response_generator(response_stream)
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=groq_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
                usage = response.usage
                return {
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                        "total_tokens": usage.total_tokens if usage else 0,
                    } if usage else None
                }
        except GroqError:
            raise
        except Exception:
            raise

class GeminiAdapter(BaseLLMClient):
    DEFAULT_SAFETY_SETTINGS = {}

    def __init__(
        self, api_key: str, model: str
    ):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model_name = model
        self.safety_settings = self.DEFAULT_SAFETY_SETTINGS
        logger.info(f"GeminiAdapter initialized with model: {self.model_name}")

    @staticmethod
    def _convert_messages_to_gemini(messages: List[Message]) -> List[Dict[str, Any]]:
        gemini_history = []
        system_prompt = None
        for msg in messages:
            role = "user"
            if msg.role == "assistant":
                role = "model"
            elif msg.role == "system":
                system_prompt = msg.content
                continue
            gemini_history.append({"role": role, "parts": [msg.content]})
        if system_prompt and gemini_history:
            gemini_history.insert(0, {"role": "user", "parts": [system_prompt]})
            gemini_history.insert(1, {"role": "model", "parts": ["Okay, I understand the instructions."]})
        elif system_prompt:
            gemini_history.append({"role": "user", "parts": [system_prompt]})
            gemini_history.append({"role": "model", "parts": ["Okay."]})
        return gemini_history

    async def _stream_response_generator(
        self, response_stream: AsyncGenerator
    ) -> AsyncGenerator[Dict[str, Any], None]:
        collected_content = ""
        finish_reason = "stop"
        from google.generativeai.types import StopCandidateException
        try:
            async for chunk in response_stream:
                if chunk.text:
                    token_text = chunk.text
                    collected_content += token_text
                    yield StreamToken(
                        token=token_text,
                        is_finished=False,
                        finish_reason=None
                    ).model_dump()
        except StopCandidateException:
            finish_reason = "length"
        except Exception:
            finish_reason = "error"
            yield StreamToken(token="", is_finished=True, finish_reason=finish_reason).model_dump()
        yield StreamToken(token="", is_finished=True, finish_reason=finish_reason).model_dump()

    async def chat(
        self,
        messages: List[Message],
        stream: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any] | AsyncGenerator[Dict[str, Any], None]:
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", None)
        gemini_history = self._convert_messages_to_gemini(messages)
        last_message_content = gemini_history[-1]["parts"][0] if gemini_history else " "
        history_for_chat_start = gemini_history[:-1] if gemini_history else []
        import google.generativeai as genai
        loop = asyncio.get_running_loop()
        if stream:
            response_stream = await loop.run_in_executor(
                None,
                lambda: genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                    safety_settings=self.safety_settings
                ).start_chat(history=history_for_chat_start).send_message(
                    last_message_content, stream=True
                )
            )
            async def async_gen_wrapper(sync_iterator):
                for item in sync_iterator:
                    yield item
            return self._stream_response_generator(async_gen_wrapper(response_stream))
        else:
            response = await loop.run_in_executor(
                None,
                lambda: genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                    safety_settings=self.safety_settings
                ).start_chat(history=history_for_chat_start).send_message(
                    last_message_content, stream=False
                )
            )
            return {"content": response.text, "usage": None}


def get_llm_client(settings: Settings) -> BaseLLMClient:
    provider = settings.llm_provider
    if provider == LLMProvider.GROQ:
        return GroqAdapter(api_key=settings.groq_api_key.get_secret_value(), model=settings.groq_model)
    if provider == LLMProvider.GEMINI:
        return GeminiAdapter(api_key=settings.gemini_api_key.get_secret_value(), model=settings.gemini_model)
    raise ValueError(f"Unsupported provider: {provider}")
