from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal


class Message(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of conversation messages")
    stream: bool = Field(False, description="Whether to stream the response")
    temperature: Optional[float] = Field(0.7, description="Temperature for response generation", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    
    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Tell me about AI."}
                ],
                "stream": False,
                "temperature": 0.7
            }
        }


class ChatResponse(BaseModel):
    message: Message
    usage: Optional[Dict[str, Any]] = None


class StreamToken(BaseModel):
    token: str
    finish_reason: Optional[str] = None
    is_finished: bool = False
