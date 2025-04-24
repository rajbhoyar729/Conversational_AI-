import logging
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ConfigDict, model_validator, PositiveInt

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "error", "tool_calls"]

class Message(BaseModel):
    role: Role = Field(...)
    content: str = Field(..., min_length=1)

    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        json_schema_extra={
            "examples": [
                {"role": "user", "content": "What is the weather like?"},
                {"role": "assistant", "content": "The weather is sunny."},
                {"role": "system", "content": "You are a helpful weather bot."},
            ]
        }
    )

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)
    stream: bool = Field(default=False)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[PositiveInt] = Field(default=None)

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Explain the concept of asynchronous programming."}
                ],
                "stream": False,
                "temperature": 0.6,
                "max_tokens": 1000
            }
        }
    )

class UsageStats(BaseModel):
    prompt_tokens: Optional[int] = Field(None, ge=0)
    completion_tokens: Optional[int] = Field(None, ge=0)
    total_tokens: Optional[int] = Field(None, ge=0)

    model_config = ConfigDict(extra='ignore')

class ChatResponse(BaseModel):
    message: Message = Field(...)
    usage: Optional[UsageStats] = Field(default=None)

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "example": {
                "message": {
                    "role": "assistant",
                    "content": "Asynchronous programming allows tasks to run independently..."
                },
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 210,
                    "total_tokens": 240
                }
            }
        }
    )

class StreamToken(BaseModel):
    token: str = Field(default="")
    is_finished: bool = Field(default=False)
    finish_reason: Optional[FinishReason] = Field(default=None)
    content: Optional[str] = Field(default=None)
    usage: Optional[UsageStats] = Field(default=None)

    model_config = ConfigDict(extra='ignore')

    @model_validator(mode='after')
    def check_final_chunk_fields(self) -> 'StreamToken':
        is_finished = self.is_finished
        finish_reason = self.finish_reason
        content = self.content
        usage = self.usage

        if finish_reason is not None and not is_finished:
            logger.warning(
                f"StreamToken received finish_reason='{finish_reason}' but is_finished=False. "
                f"Setting finish_reason to None for consistency."
            )
            self.finish_reason = None

        return self

class ServerInfo(BaseModel):
    status: Literal["ok", "error"] = Field(...)
    message: str = Field(...)
    llm_provider: Optional[str] = Field(None)
    debug_mode: Optional[bool] = Field(None)

    model_config = ConfigDict(extra='forbid')
