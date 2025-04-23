import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import ValidationError, TypeAdapter

from app.config import Settings, get_settings
from app.models.schemas import ChatRequest, ChatResponse, Message, StreamToken
from app.services.agent import ConversationalAgent
from app.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

MessageListAdapter = TypeAdapter(list[Message])

async def _process_chat_request(request: ChatRequest, agent: ConversationalAgent) -> dict:
    return await agent.process_message(
        request.messages,
        stream=False,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

async def _generate_stream_response(request: ChatRequest, agent: ConversationalAgent) -> AsyncGenerator[str, None]:
    try:
        finish_reason = None
        async for chunk in agent.process_message(
            request.messages,
            stream=True,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ):
            try:
                token_data = StreamToken(**chunk)
                if token_data.is_finished:
                    finish_reason = token_data.finish_reason
                yield f"data: {token_data.model_dump_json()}\n\n"
            except ValidationError:
                continue
    except Exception as e:
        logger.error(f"Error during stream generation: {e}", exc_info=True)
        error_token = StreamToken(token="", is_finished=True, finish_reason="error")
        yield f"data: {error_token.model_dump_json()}\n\n"

async def _handle_websocket_session(websocket: WebSocket, agent: ConversationalAgent):
    while True:
        try:
            data = await websocket.receive_json()
            request = ChatRequest(**data)

            if request.stream:
                async for chunk in agent.process_message(
                    request.messages,
                    stream=True,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ):
                    try:
                        token_data = StreamToken(**chunk)
                        await websocket.send_json(token_data.model_dump())
                    except ValidationError:
                        await websocket.send_json(StreamToken(token="", is_finished=True, finish_reason="error").model_dump())
                        break
            else:
                response_data = await _process_chat_request(request, agent)
                chat_response = ChatResponse(
                    message=Message(role="assistant", content=response_data["content"]),
                    usage=response_data.get("usage"),
                )
                await websocket.send_json(chat_response.model_dump())

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected.")
            break
        except ValidationError as e:
            logger.warning(f"Invalid WebSocket request received: {e}")
            await websocket.send_json({
                "error": "Invalid request format",
                "details": e.errors()
            })
        except json.JSONDecodeError:
            logger.warning("Invalid JSON received over WebSocket.")
            await websocket.send_json({"error": "Invalid JSON format"})
        except Exception as e:
            logger.error(f"Unhandled error in WebSocket chat: {e}", exc_info=True)
            await websocket.send_json({"error": "An internal server error occurred."})
            break

@router.post("", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
):
    try:
        llm_client = get_llm_client(settings)
        agent = ConversationalAgent(llm_client)
        response_data = await _process_chat_request(request, agent)
        return ChatResponse(
            message=Message(role="assistant", content=response_data["content"]),
            usage=response_data.get("usage"),
        )
    except Exception as e:
        logger.error(f"Error processing non-streaming chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the chat request."
        )

@router.post("/stream")
async def stream_chat_endpoint(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
):
    if not request.stream:
        logger.warning("/stream endpoint called with stream=False in request body.")
    try:
        llm_client = get_llm_client(settings)
        agent = ConversationalAgent(llm_client)
        return StreamingResponse(
            _generate_stream_response(request, agent),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.error(f"Error processing streaming chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the streaming chat request."
        )

@router.websocket("/ws")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    settings: Settings = Depends(get_settings),
):
    await websocket.accept()
    try:
        llm_client = get_llm_client(settings)
        agent = ConversationalAgent(llm_client)
        await _handle_websocket_session(websocket, agent)
    except Exception as e:
        logger.error(f"Failed to initialize WebSocket session: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason="Internal server error during setup")
        except RuntimeError:
            pass
    finally:
        logger.info("WebSocket connection closing.")

