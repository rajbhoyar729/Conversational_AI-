import logging
from typing import AsyncGenerator, List, Optional

from ..config import settings
from ..llm_integrations.groq_llm import generate_groq_stream
from ..llm_integrations.gemini_llm import generate_gemini_stream
from ..schemas import Message

logger = logging.getLogger(__name__)

async def generate_response_stream(
    messages: List[Message],
    provider: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Generate a streaming response from the selected LLM provider.
    
    Args:
        messages: List of message objects
        provider: LLM provider to use (defaults to settings.DEFAULT_LLM_PROVIDER)
        
    Yields:
        Text chunks from the LLM response
    """
    # Use default provider if none specified
    if not provider:
        provider = settings.DEFAULT_LLM_PROVIDER
    
    # Validate provider
    if provider not in settings.SUPPORTED_LLM_PROVIDERS:
        error_msg = f"Unsupported provider: {provider}. Supported providers: {settings.SUPPORTED_LLM_PROVIDERS}"
        logger.error(error_msg)
        yield error_msg
        return
    
    # Convert messages to dict format
    messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]
    
    # Select the appropriate LLM integration
    if provider == "groq":
        if not settings.GROQ_API_KEY:
            error_msg = "Groq API key not configured"
            logger.error(error_msg)
            yield error_msg
            return
            
        async for chunk in generate_groq_stream(messages_dict, settings.GROQ_API_KEY):
            yield chunk
            
    elif provider == "gemini":
        if not settings.GOOGLE_API_KEY:
            error_msg = "Google API key not configured"
            logger.error(error_msg)
            yield error_msg
            return
            
        async for chunk in generate_gemini_stream(messages_dict, settings.GOOGLE_API_KEY):
            yield chunk
