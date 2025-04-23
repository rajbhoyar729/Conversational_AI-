from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional, Union
import os

from app.models.schemas import Message
from app.config import Settings, LLMProvider

logger = logging.getLogger(__name__)


class LLMClientAdapter(ABC):
    """Abstract base class for LLM client adapters"""
    
    @abstractmethod
    async def chat(
        self, 
        messages: List[Message], 
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Send a chat request to the LLM provider
        
        Args:
            messages: List of conversation messages
            stream: Whether to stream the response
            temperature: Temperature for response generation
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            If stream=False: Dict with response content and usage stats
            If stream=True: AsyncGenerator yielding response chunks
        """
        pass


class GroqAdapter(LLMClientAdapter):
    """Adapter for Groq API with Llama models"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-instant"):
        self.api_key = api_key
        self.model = model
        
    async def chat(
        self, 
        messages: List[Message], 
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        try:
            from groq import AsyncGroq
            
            client = AsyncGroq(api_key=self.api_key)
            
            # Convert our Message objects to Groq format
            groq_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            if stream:
                async def stream_response():
                    response_stream = await client.chat.completions.create(
                        model=self.model,
                        messages=groq_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True
                    )
                    
                    collected_content = ""
                    
                    async for chunk in response_stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            token = chunk.choices[0].delta.content
                            collected_content += token
                            yield {
                                "token": token,
                                "is_finished": False,
                                "finish_reason": chunk.choices[0].finish_reason
                            }
                    
                    # Final chunk with complete content
                    yield {
                        "token": "",
                        "content": collected_content,
                        "is_finished": True,
                        "finish_reason": "stop"
                    }
                
                return stream_response()
            else:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=groq_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return {
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
                
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise


class GeminiAdapter(LLMClientAdapter):
    """Adapter for Google Gemini API"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        self.api_key = api_key
        self.model = model
        
    async def chat(
        self, 
        messages: List[Message], 
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        try:
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            import asyncio
            
            genai.configure(api_key=self.api_key)
            
            # Convert our Message objects to Gemini format
            gemini_messages = []
            for msg in messages:
                if msg.role == "system":
                    # Add system message as user message with special prefix
                    gemini_messages.append({"role": "user", "parts": [f"<system>\n{msg.content}\n</system>"]})
                elif msg.role == "user":
                    gemini_messages.append({"role": "user", "parts": [msg.content]})
                elif msg.role == "assistant":
                    gemini_messages.append({"role": "model", "parts": [msg.content]})
            
            # Configure safety settings
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            # Initialize the model
            model = genai.GenerativeModel(
                model_name=self.model,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
                safety_settings=safety_settings
            )
            
            # Create a chat session
            chat = model.start_chat(history=gemini_messages[:-1] if gemini_messages else [])
            
            if stream:
                async def stream_response():
                    # Get the last message (current query)
                    last_message = gemini_messages[-1] if gemini_messages else {"role": "user", "parts": ["Hello"]}
                    
                    # Use the event loop to run the blocking operation
                    loop = asyncio.get_event_loop()
                    response_stream = await loop.run_in_executor(
                        None, 
                        lambda: chat.send_message(
                            last_message["parts"][0], 
                            stream=True
                        )
                    )
                    
                    collected_content = ""
                    
                    for chunk in response_stream:
                        if chunk.text:
                            token = chunk.text
                            collected_content += token
                            yield {
                                "token": token,
                                "is_finished": False
                            }
                    
                    # Final chunk with complete content
                    yield {
                        "token": "",
                        "content": collected_content,
                        "is_finished": True,
                        "finish_reason": "stop"
                    }
                
                return stream_response()
            else:
                # Get the last message (current query)
                last_message = gemini_messages[-1] if gemini_messages else {"role": "user", "parts": ["Hello"]}
                
                # Use the event loop to run the blocking operation
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: chat.send_message(last_message["parts"][0])
                )
                
                return {
                    "content": response.text,
                    # Gemini doesn't provide token usage info in the same way
                    "usage": {}
                }
                
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise


def get_llm_client(settings: Settings) -> LLMClientAdapter:
    """
    Factory function to get the appropriate LLM client based on settings
    """
    if settings.llm_provider == LLMProvider.GROQ:
        if not settings.groq_api_key:
            raise ValueError("Groq API key is required when using Groq provider")
        return GroqAdapter(api_key=settings.groq_api_key, model=settings.groq_model)
    
    elif settings.llm_provider == LLMProvider.GEMINI:
        if not settings.gemini_api_key:
            raise ValueError("Gemini API key is required when using Gemini provider")
        return GeminiAdapter(api_key=settings.gemini_api_key, model=settings.gemini_model)
    
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
