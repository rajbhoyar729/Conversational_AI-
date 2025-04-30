"""
Chat API Router for Conversational AI Application

This module implements the FastAPI routes for chat functionality with support for:
- Non-streaming chat responses
- Server-Sent Events (SSE) streaming
- WebSocket communication
"""

import asyncio
import json
import logging
from typing import Annotated, AsyncGenerator, Dict, List, Optional, Union

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.models.schemas import ChatRequest, ChatResponse, Message, ServerInfo, StreamToken, UsageStats
from app.services.llm_client import BaseLLMClient, get_llm_client
from app.services.agent import ConversationalAgent
from app.services.llm_client import get_active_model

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

# Type aliases for dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings)]
LLMClientDep = Annotated[
    BaseLLMClient,
    Depends(lambda settings: get_llm_client(settings), use_cache=True)
]

async def get_conversational_agent(llm_client: LLMClientDep) -> ConversationalAgent:
    """Factory dependency for creating a conversational agent"""
    return ConversationalAgent(llm_client=llm_client)

AgentDep = Annotated[ConversationalAgent, Depends(get_conversational_agent)]

class ChatError(Exception):
    """Base class for chat-related errors"""
    def __init__(self, message: str, status_code: int):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

class ValidationError(ChatError):
    """Raised when input validation fails"""
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_400_BAD_REQUEST)

class StreamingError(ChatError):
    """Raised during streaming operations"""
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)

@router.post(
    "",
    response_model=ChatResponse,
    summary="Process a Non-Streaming Chat Request",
    description="Sends the conversation history to the LLM and receives a single, complete response.",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input data"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error"},
        status.HTTP_502_BAD_GATEWAY: {"description": "Error communicating with LLM provider"},
    },
)
async def process_chat_non_streaming(
    request: ChatRequest,
    agent: AgentDep,
) -> ChatResponse:
    """
    Process a non-streaming chat request.
    
    Args:
        request: Validated chat request containing message history and parameters
        agent: Conversational agent for processing the request
        
    Returns:
        ChatResponse: Complete response from the LLM
    """
    if request.stream:
        logger.warning("Received request with stream=True on non-streaming endpoint. Processing as non-streaming.")
        request.stream = False

    try:
        logger.info(f"Processing non-streaming request with {len(request.messages)} messages.")
        response_data = await agent.process_message(
            messages=request.messages,
            stream=False,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        
        if isinstance(response_data, dict):
            return ChatResponse(**response_data)
        raise ValidationError("Unexpected response format from agent")
            
    except ValidationError as ve:
        logger.error(f"Validation error processing agent response: {ve}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error: Failed to format response.",
        )
    except Exception as e:
        logger.error(f"Error processing non-streaming request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred."
        )

@router.post(
    "/stream",
    summary="Process a Streaming Chat Request (SSE)",
    description="Sends the conversation history and streams the LLM response back chunk by chunk using Server-Sent Events.",
    responses={
        status.HTTP_200_OK: {
            "content": {"text/event-stream": {}},
            "description": "Successful streaming response using Server-Sent Events. Each event data payload is a JSON object matching the StreamToken schema.",
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input data"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error during streaming"},
        status.HTTP_502_BAD_GATEWAY: {"description": "Error communicating with LLM provider"},
    },
)
async def process_chat_streaming_sse(
    request: ChatRequest,
    agent: AgentDep,
) -> StreamingResponse:
    """
    Process a streaming chat request using Server-Sent Events (SSE).
    
    Args:
        request: Validated chat request containing message history and parameters
        agent: Conversational agent for processing the request
        
    Returns:
        StreamingResponse: Event stream of tokenized responses
    """
    if not request.stream:
        logger.warning("Received request with stream=False on streaming endpoint. Processing as streaming.")
        request.stream = True

    async def stream_generator() -> AsyncGenerator[str, None]:
        try:
            logger.info(f"Processing streaming SSE request with {len(request.messages)} messages.")
            response_stream = await agent.process_message(
                messages=request.messages,
                stream=True,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            
            if not isinstance(response_stream, AsyncGenerator):
                logger.error("Agent did not return an async generator for streaming request.")
                error_token = StreamToken(token="Internal Server Error: Invalid stream response.", is_finished=True, finish_reason="error")
                yield f"data: {error_token.model_dump_json()}\n\n"
                return

            async for chunk_dict in response_stream:
                try:
                    token_data = StreamToken(**chunk_dict)
                    yield f"data: {token_data.model_dump_json()}\n\n"
                except ValidationError as ve:
                    logger.warning(f"Skipping invalid chunk received from agent stream: {ve}. Chunk: {chunk_dict}")
                    continue
                except Exception as chunk_exc:
                    logger.error(f"Error processing/yielding stream chunk: {chunk_exc}", exc_info=True)
                    error_token = StreamToken(token=f"Error during streaming: {chunk_exc}", is_finished=True, finish_reason="error")
                    yield f"data: {error_token.model_dump_json()}\n\n"
                    return
                    
        except Exception as e:
            logger.error(f"Error initiating or processing SSE stream: {e}", exc_info=True)
            detail = "An internal error occurred during streaming."
            if isinstance(e, ConnectionError):
                detail = f"Could not connect to the upstream LLM service: {e}"
            elif isinstance(e, ValueError):
                detail = f"Invalid input: {e}"
            error_token = StreamToken(token=detail, is_finished=True, finish_reason="error")
            yield f"data: {error_token.model_dump_json()}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@router.websocket("/ws")
async def process_chat_websocket(
    websocket: WebSocket,
    settings: SettingsDep,
    agent: AgentDep,
):
    """
    Handle WebSocket connections for real-time chat interactions.
    
    Args:
        websocket: WebSocket connection object
        settings: Application settings
        agent: Conversational agent for processing requests
    """
    await websocket.accept()
    logger.info(f"WebSocket connection accepted from: {websocket.client}")
    
    try:
        while True:
            try:
                raw_data = await websocket.receive_text()
                data = json.loads(raw_data)
                request = ChatRequest(**data)
                logger.debug(f"WebSocket received request: stream={request.stream}, messages={len(request.messages)}")
            except WebSocketDisconnect:
                logger.info(f"WebSocket client {websocket.client} disconnected.")
                break
            except json.JSONDecodeError:
                logger.warning(f"WebSocket received invalid JSON from {websocket.client}.")
                await websocket.send_json({"error": "Invalid JSON format received.", "is_finished": True, "finish_reason": "error"})
                continue
            except ValidationError as ve:
                logger.warning(f"WebSocket received invalid ChatRequest data: {ve}")
                await websocket.send_json({
                    "error": "Invalid request data.",
                    "details": ve.errors(),
                    "is_finished": True,
                    "finish_reason": "error"
                })
                continue
            except Exception as e:
                logger.error(f"Error receiving WebSocket message: {e}", exc_info=True)
                await websocket.send_json({"error": f"Server error receiving message: {e}", "is_finished": True, "finish_reason": "error"})
                break

            try:
                response_or_stream = await agent.process_message(
                    messages=request.messages,
                    stream=request.stream,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
                
                if request.stream:
                    if not isinstance(response_or_stream, AsyncGenerator):
                        logger.error("Agent did not return async generator for WS stream.")
                        await websocket.send_json(StreamToken(token="Internal Error: Invalid stream response", is_finished=True, finish_reason="error").model_dump())
                        continue
                        
                    async for chunk_dict in response_or_stream:
                        try:
                            token_data = StreamToken(**chunk_dict)
                            await websocket.send_json(token_data.model_dump())
                        except ValidationError as ve:
                            logger.warning(f"Skipping invalid chunk for WebSocket: {ve}. Chunk: {chunk_dict}")
                            continue
                        except Exception as chunk_exc:
                            logger.error(f"Error processing/sending WS chunk: {chunk_exc}", exc_info=True)
                            await websocket.send_json(StreamToken(token=f"Error during streaming: {chunk_exc}", is_finished=True, finish_reason="error").model_dump())
                            raise
                else:
                    if not isinstance(response_or_stream, dict):
                        logger.error("Agent did not return dict for non-streaming WS request.")
                        await websocket.send_json({"error": "Internal Server Error: Invalid response format.", "is_finished": True})
                        continue
                        
                    try:
                        validated_response = ChatResponse(**response_or_stream)
                        await websocket.send_json(validated_response.model_dump())
                    except ValidationError as ve:
                        logger.error(f"Failed to validate non-streaming agent response for WS: {ve}")
                        await websocket.send_json({"error": "Internal Server Error: Failed to format response.", "is_finished": True})
                        
            except Exception as e:
                logger.error(f"Error processing WebSocket request via agent: {e}", exc_info=True)
                detail = "An internal error occurred while processing your request."
                if isinstance(e, ConnectionError): 
                    detail = f"Could not connect to the upstream LLM service: {e}"
                elif isinstance(e, ValueError): 
                    detail = f"Invalid input: {e}"
                await websocket.send_json({"token": detail, "error": detail, "is_finished": True, "finish_reason": "error"})
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket client {websocket.client} disconnected during processing/sending.")
    except Exception as e:
        logger.error(f"Unhandled exception in WebSocket handler for {websocket.client}: {e}", exc_info=True)
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason=f"Unhandled server error: {type(e).__name__}")
        except Exception:
            logger.warning("Failed to send WebSocket close frame, connection might already be closed.")
    finally:
        logger.info(f"WebSocket connection closing for {websocket.client}.")

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        model = get_active_model()
        response = await model.generate(request.prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@router.post("/switch-model")
async def switch_model(new_model: str):
    try:
        # Logic to update the active model in the configuration
        # For example, update the .env file or in-memory config
        return {"message": f"Active model switched to {new_model}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error switching model: {str(e)}")