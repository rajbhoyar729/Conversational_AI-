import logging
import asyncio
from typing import AsyncGenerator, List, Dict
import google.generativeai as genai

logger = logging.getLogger(__name__)

async def generate_gemini_stream(
    messages: List[Dict],
    api_key: str,
) -> AsyncGenerator[str, None]:
    """
    Generate a streaming response from Google Gemini.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        api_key: Google API key
        
    Yields:
        Text chunks from the LLM response
    """
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Convert messages to the format expected by Gemini
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                formatted_messages.append({"role": "model", "parts": [msg["content"]]})
            elif msg["role"] == "system":
                # Prepend system message to the first user message
                if formatted_messages and formatted_messages[0]["role"] == "user":
                    formatted_messages[0]["parts"][0] = f"{msg['content']}\n\n{formatted_messages[0]['parts'][0]}"
                else:
                    # Add as a user message if it's the first message
                    formatted_messages.append({"role": "user", "parts": [msg["content"]]})
        
        logger.debug(f"Sending request to Gemini with {len(formatted_messages)} messages")
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-pro')
        
        # Start the chat
        chat = model.start_chat(history=formatted_messages[:-1] if formatted_messages else [])
        
        # Get the response stream
        response = chat.send_message(
            formatted_messages[-1]["parts"][0] if formatted_messages else "",
            stream=True
        )
        
        # Yield chunks as they arrive
        for chunk in response:
            if chunk.text:
                logger.debug(f"Received chunk: {chunk.text}")
                yield chunk.text
                # Small delay to simulate streaming
                await asyncio.sleep(0.01)
                
    except Exception as e:
        logger.error(f"Error in Gemini stream generation: {str(e)}")
        yield f"Error: {str(e)}"
