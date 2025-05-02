from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

# Chat message schemas
class Message(BaseModel):
    """Chat message model."""
    role: Literal["user", "assistant", "system"] = "user"
    content: str

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str
    history: List[Message] = []
    provider: Optional[str] = None
    session_id: str

class ChatResponse(BaseModel):
    """Chat response model."""
    status: str = "processing"
    message: str = "Request received. Updates will be sent via WebSocket."

# WebSocket message schemas
class LLMChunkPayload(BaseModel):
    """Payload for LLM text chunks."""
    chunk: str
    
class InfoPayload(BaseModel):
    """Payload for informational messages."""
    message: str
    
class ErrorPayload(BaseModel):
    """Payload for error messages."""
    message: str
    code: Optional[str] = None
    
class LLMFinalPayload(BaseModel):
    """Payload for final LLM response."""
    complete_response: str

class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: Literal["llm_chunk", "info", "error", "llm_final"]
    payload: Union[Dict, LLMChunkPayload, InfoPayload, ErrorPayload, LLMFinalPayload]
    session_id: str
