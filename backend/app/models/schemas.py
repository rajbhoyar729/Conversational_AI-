from typing import Any, Dict, List, Literal, Optional
from enum import Enum

from pydantic import BaseModel, Field, model_validator, ConfigDict
from pydantic_core.core_schema import FieldValidationInfo

# ======================
# Enums and Type Aliases
# ======================

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "error", "tool_calls"]

# ================
# Base Data Models
# ================

class Message(BaseModel):
    """
    Represents a single message in a conversation.

    - **role**: The speaker, e.g., `system`, `user`, or `assistant`.
    - **content**: The actual message text.

    Example:
    ```json
    {
      "role": "user",
      "content": "What is the weather like?"
    }
    ```
    """
    role: Role = Field(..., description="The speaker role in the conversation.")
    content: str = Field(..., min_length=1, description="Content of the message.")

    model_config = ConfigDict(
        extra='forbid',
        frozen=True,
        json_schema_extra={
            "example": {
                "role": "user",
                "content": "What is the weather like today?"
            }
        }
    )


class UsageStats(BaseModel):
    """
    Token usage statistics for a single LLM request.

    - **prompt_tokens**: Number of tokens in the input prompt.
    - **completion_tokens**: Tokens generated in the output.
    - **total_tokens**: Total tokens processed.

    Example:
    ```json
    {
      "prompt_tokens": 100,
      "completion_tokens": 200,
      "total_tokens": 300
    }
    ```
    """
    prompt_tokens: Optional[int] = Field(None, ge=0)
    completion_tokens: Optional[int] = Field(None, ge=0)
    total_tokens: Optional[int] = Field(None, ge=0)

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "example": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300
            }
        }
    )


class ChatRequest(BaseModel):
    """
    Request body for chat endpoints.

    - **messages**: Ordered list of conversation history.
    - **stream**: If `True`, response will stream tokens via SSE or WebSocket.
    - **temperature**: Controls randomness in output (0.0 to 2.0).
    - **max_tokens**: Maximum number of tokens to generate.

    Example:
    ```json
    {
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."}
      ],
      "stream": false,
      "temperature": 0.7,
      "max_tokens": 200
    }
    ```
    """
    messages: List[Message] = Field(..., min_length=1)
    stream: bool = Field(default=False)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Explain quantum computing."}
                ],
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 200
            }
        }
    )

    @model_validator(mode='after')
    def validate_at_least_one_user_message(self):
        """
        Ensures at least one message is from the user.
        """
        if not any(msg.role == "user" for msg in self.messages):
            raise ValueError("At least one user message is required.")
        return self


class ChatResponse(BaseModel):
    """
    Response from non-streaming chat endpoints.

    - **message**: Assistant's reply.
    - **usage**: Optional token usage statistics.

    Example:
    ```json
    {
      "message": {
        "role": "assistant",
        "content": "Quantum computing uses qubits..."
      },
      "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 200,
        "total_tokens": 300
      }
    }
    ```
    """
    message: Message
    usage: Optional[UsageStats] = None

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "example": {
                "message": {
                    "role": "assistant",
                    "content": "Quantum computing leverages quantum bits..."
                },
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "total_tokens": 300
                }
            }
        }
    )


class StreamToken(BaseModel):
    """
    A single token returned during streaming responses.

    - **token**: The piece of text being streamed.
    - **is_finished**: Whether the stream has completed.
    - **finish_reason**: Why the stream ended (`stop`, `length`, `error`).
    - **content**: Full assistant response (only present in final token).
    - **usage**: Token stats (only present in final token).

    Example:
    ```json
    {
      "token": "Quantum",
      "is_finished": false
    }
    ```
    """
    token: str = Field(default="")
    is_finished: bool = Field(default=False)
    finish_reason: Optional[FinishReason] = Field(default=None)
    content: Optional[str] = Field(default=None)
    usage: Optional[UsageStats] = Field(default=None)

    model_config = ConfigDict(extra='ignore')

    @model_validator(mode='after')
    def validate_final_token(self):
        """
        Ensures that if `finish_reason` is present, `is_finished` must be True.
        """
        if self.finish_reason is not None and not self.is_finished:
            raise ValueError("finish_reason must not be set unless is_finished is True.")
        return self


class ServerInfo(BaseModel):
    """
    Health check response.

    - **status**: `"ok"` or `"error"`.
    - **message**: Descriptive message (e.g., server running status).
    - **llm_provider**: The current LLM provider (e.g., gemini, groq).
    - **debug_mode**: Whether debug mode is enabled.

    Example:
    ```json
    {
      "status": "ok",
      "message": "Server is running",
      "llm_provider": "gemini",
      "debug_mode": false
    }
    ```
    """
    status: Literal["ok", "error"] = Field(...)
    message: str = Field(...)
    llm_provider: Optional[str] = Field(None)
    debug_mode: Optional[bool] = Field(None)

    model_config = ConfigDict(extra='forbid')