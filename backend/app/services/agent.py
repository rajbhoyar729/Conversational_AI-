import logging
from typing import List, Dict, Any, AsyncGenerator, Union, Optional
import asyncio

from app.models.schemas import Message
from app.services.llm_client import LLMClientAdapter

logger = logging.getLogger(__name__)


class ConversationalAgent:
    """
    Manages conversation state and interacts with LLM providers
    """
    
    def __init__(self, llm_client: LLMClientAdapter):
        self.llm_client = llm_client
        self.conversation_history = []
        
    async def process_message(
        self, 
        messages: List[Message], 
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Process a message and get a response from the LLM
        
        Args:
            messages: List of conversation messages
            stream: Whether to stream the response
            temperature: Temperature for response generation
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            If stream=False: Dict with response content and usage stats
            If stream=True: AsyncGenerator yielding response chunks
        """
        try:
            # Update conversation history
            self.conversation_history = messages
            
            # Get response from LLM
            response = await self.llm_client.chat(
                messages=messages,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if stream:
                # For streaming, we need to update the history as we go
                collected_content = ""
                
                async for chunk in response:
                    if "token" in chunk:
                        collected_content += chunk.get("token", "")
                    
                    # If this is the final chunk with the complete content
                    if chunk.get("is_finished", False) and "content" in chunk:
                        collected_content = chunk["content"]
                    
                    yield chunk
                
                # Update conversation history with the complete response
                self.conversation_history.append(
                    Message(role="assistant", content=collected_content)
                )
                
                return
            else:
                # For non-streaming, update history with the complete response
                self.conversation_history.append(
                    Message(role="assistant", content=response["content"])
                )
                
                return response
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            error_response = {
                "content": f"I'm sorry, but I encountered an error: {str(e)}",
                "error": str(e)
            }
            
            if stream:
                async def error_generator():
                    yield {
                        "token": error_response["content"],
                        "content": error_response["content"],
                        "is_finished": True,
                        "error": str(e)
                    }
                return error_generator()
            else:
                return error_response
    
    def get_conversation_history(self) -> List[Message]:
        """Get the current conversation history"""
        return self.conversation_history
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history"""
        self.conversation_history = []
