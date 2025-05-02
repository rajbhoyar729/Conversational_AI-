import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from ..schemas import ChatRequest, ChatResponse, WebSocketMessage, LLMChunkPayload, LLMFinalPayload, ErrorPayload
from ..services.chat_service import generate_response_stream
from ..utils.websocket_manager import manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

async def process_llm_stream(session_id: str, messages, provider: str):
    """Background task to process LLM stream and send updates via WebSocket"""
    try:
        # Check if WebSocket connection exists
        if not manager.get_connection(session_id):
            logger.error(f"No active WebSocket connection for session {session_id}")
            return
            
        # Accumulate the complete response
        complete_response = ""
        
        # Stream response from LLM
        async for chunk in generate_response_stream(messages, provider):
            if chunk.startswith("Error:"):
                # Send error message
                error_msg = WebSocketMessage(
                    type="error",
                    payload=ErrorPayload(message=chunk),
                    session_id=session_id
                )
                await manager.send_json(session_id, error_msg.dict())
                return
                
            # Accumulate the response
            complete_response += chunk
            
            # Send chunk update
            chunk_msg = WebSocketMessage(
                type="llm_chunk",
                payload=LLMChunkPayload(chunk=chunk),
                session_id=session_id
            )
            await manager.send_json(session_id, chunk_msg.dict())
            
        # Send final complete response
        final_msg = WebSocketMessage(
            type="llm_final",
            payload=LLMFinalPayload(complete_response=complete_response),
            session_id=session_id
        )
        await manager.send_json(session_id, final_msg.dict())
        
    except Exception as e:
        logger.error(f"Error in process_llm_stream: {str(e)}")
        # Try to send error message if WebSocket is still connected
        try:
            error_msg = WebSocketMessage(
                type="error",
                payload=ErrorPayload(message=f"Server error: {str(e)}"),
                session_id=session_id
            )
            await manager.send_json(session_id, error_msg.dict())
        except:
            pass

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to initiate a chat request.
    The actual response will be streamed via WebSocket.
    """
    logger.info(f"Received chat request for session {request.session_id}")
    
    # Check if there's an active WebSocket connection for this session
    if not manager.get_connection(request.session_id):
        raise HTTPException(status_code=400, detail="No active WebSocket connection for this session")
    
    # Add the user's message to the history
    messages = request.history + [{"role": "user", "content": request.message}]
    
    # Process the LLM stream in the background
    background_tasks.add_task(
        process_llm_stream,
        request.session_id,
        messages,
        request.provider
    )
    
    return ChatResponse()

@router.get("/providers")
async def get_providers():
    """Get available LLM providers."""
    from ..config import settings
    return {
        "providers": settings.SUPPORTED_LLM_PROVIDERS,
        "default": settings.DEFAULT_LLM_PROVIDER
    }
