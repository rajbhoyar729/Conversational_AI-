import logging
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..utils.websocket_manager import manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

@router.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming updates to clients"""
    session_id = None
    
    try:
        await websocket.accept()
        
        # Get initial message with session ID
        data = await websocket.receive_json()
        session_id = data.get("session_id")
        
        if not session_id:
            # Generate a session ID if not provided
            session_id = str(uuid.uuid4())
            await websocket.send_json({
                "type": "info", 
                "payload": {"message": f"Assigned session ID: {session_id}"}
            })
        
        # Store the WebSocket connection
        await manager.connect(session_id, websocket)
        
        # Keep the connection alive
        while True:
            # This will raise WebSocketDisconnect when the client disconnects
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
        if session_id:
            manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if session_id:
            manager.disconnect(session_id)
