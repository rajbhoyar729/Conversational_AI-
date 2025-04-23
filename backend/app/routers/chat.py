from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
import logging
import json
from typing import List

from app.models.schemas import ChatRequest, ChatResponse, Message, StreamToken
from app.services.agent import ConversationalAgent
from app.services.llm_client import get_llm_client
from app.config import get_settings, Settings

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Process a chat request and return a complete response
    """
    try:
        llm_client = get_llm_client(settings)
        agent = ConversationalAgent(llm_client)
        
        # If stream is requested, we still process as non-streaming for this endpoint
        response = await agent.process_message(request.messages, stream=False, 
                                              temperature=request.temperature,
                                              max_tokens=request.max_tokens)
        
        return ChatResponse(
            message=Message(role="assistant", content=response["content"]),
            usage=response.get("usage")
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@router.get("/stream")
async def stream_chat(
    messages: str,
    temperature: float = 0.7,
    max_tokens: int = None,
    settings: Settings = Depends(get_settings)
):
    """
    Stream chat responses using server-sent events
    """
    try:
        # Parse messages from JSON string
        parsed_messages = [Message(**msg) for msg in json.loads(messages)]
        
        llm_client = get_llm_client(settings)
        agent = ConversationalAgent(llm_client)
        
        async def event_generator():
            async for chunk in agent.process_message(
                parsed_messages, 
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                if isinstance(chunk, dict):
                    # Format as SSE
                    yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send end of stream marker
            yield f"data: {json.dumps({'is_finished': True})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error processing streaming chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@router.websocket("/ws")
async def websocket_chat(
    websocket: WebSocket,
    settings: Settings = Depends(get_settings)
):
    """
    Stream chat responses over WebSocket
    """
    await websocket.accept()
    
    try:
        llm_client = get_llm_client(settings)
        agent = ConversationalAgent(llm_client)
        
        while True:
            # Receive and parse the request
            data = await websocket.receive_json()
            request = ChatRequest(**data)
            
            if request.stream:
                # Stream the response
                async for chunk in agent.process_message(
                    request.messages,
                    stream=True,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ):
                    await websocket.send_json(chunk)
                
                # Send end of stream marker
                await websocket.send_json({"is_finished": True})
            else:
                # Send complete response
                response = await agent.process_message(
                    request.messages,
                    stream=False,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                
                await websocket.send_json(
                    ChatResponse(
                        message=Message(role="assistant", content=response["content"]),
                        usage=response.get("usage")
                    ).dict()
                )
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket chat: {str(e)}")
        await websocket.send_json({"error": str(e)})
