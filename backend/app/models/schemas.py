from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_items=1, description="List of conversation messages, must contain at least one message.")
    stream: bool = Field(False, description="Whether to stream the response.")
    temperature: Optional[float] = Field(0.7, description="Temperature for response generation.", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate.", gt=0)

    class Config:
        extra = 'forbid' 
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Tell me about AI."}
                ],
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 100
            }
        }

class ChatResponse(BaseModel):
    message: Message
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage statistics (e.g., prompt_tokens, completion_tokens, total_tokens).")

    class Config:
        extra = 'forbid'

class StreamToken(BaseModel):
    token: str = Field(description="The generated token text.")
    finish_reason: Optional[Literal["stop", "length", "error"]] = Field(None, description="Reason the stream finished (if applicable).")
    is_finished: bool = Field(False, description="Indicates if this is the final token in the stream.")

    class Config:
        extra = 'forbid'

    @validator('is_finished')
    def check_finish_reason(cls, v, values):
        if v and 'finish_reason' not in values:
            pass 
        if not v and values.get('finish_reason') is not None:
            raise ValueError('finish_reason should only be present when is_finished is True')
        return v
