import logging
from typing import Dict, Optional
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages active WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        """Connect a new WebSocket and store it."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session {session_id}")
    
    def disconnect(self, session_id: str):
        """Remove a WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected for session {session_id}")
    
    async def send_json(self, session_id: str, message: dict):
        """Send a JSON message to a specific WebSocket."""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(message)
                return True
            except Exception as e:
                logger.error(f"Error sending message to session {session_id}: {str(e)}")
                self.disconnect(session_id)
                return False
        return False
    
    def get_connection(self, session_id: str) -> Optional[WebSocket]:
        """Get a WebSocket connection by session ID."""
        return self.active_connections.get(session_id)
    
    @property
    def connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)

# Create a global instance
manager = WebSocketManager()
