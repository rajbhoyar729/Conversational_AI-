import logging
import asyncio
from typing import AsyncGenerator, List, Dict
import groq

logger = logging.getLogger(__name__)

async def generate_groq_stream(
    messages: List[Dict],
    api_key: str,
) -> AsyncGenerator[str, None]:
    """
    Generate a streaming response from Groq LLM.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        api_key: Groq API key
        
    Yields:
        Text chunks from the LLM response
    """
    try:
        client = groq.AsyncClient(api_key=api_key)
        
        # Convert messages to the format expected by Groq
        formatted_messages = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in messages
        ]
        
        logger.debug(f"Sending request to Groq with {len(formatted_messages)} messages")
        
        # Create a streaming completion
        stream = await client.chat.completions.create(
            model="llama3-8b-8192",  # or another model
            messages=formatted_messages,
            stream=True,
        )
        
        # Yield chunks as they arrive
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                logger.debug(f"Received chunk: {content}")
                yield content
                
    except Exception as e:
        logger.error(f"Error in Groq stream generation: {str(e)}")
        yield f"Error: {str(e)}"
