import logging
import json
import threading
import queue
import websocket
from typing import Dict, Optional, Callable

logger = logging.getLogger(__name__)

class WebSocketClient:
    """Client for WebSocket connections to the backend."""
    
    def __init__(self, base_url: str = "ws://localhost:8000"):
        self.base_url = base_url
        self.connections: Dict[str, websocket.WebSocketApp] = {}
        self.message_queues: Dict[str, queue.Queue] = {}
        
    def connect(self, session_id: str) -> bool:
        """
        Connect to the WebSocket for the given session ID.
        
        Args:
            session_id: Session ID for the WebSocket connection
            
        Returns:
            True if connection was successful or already exists, False otherwise
        """
        # Check if connection already exists
        if session_id in self.connections and self.connections[session_id].sock and self.connections[session_id].sock.connected:
            logger.info(f"WebSocket connection already exists for session {session_id}")
            return True
            
        # Create a message queue for this session
        if session_id not in self.message_queues:
            self.message_queues[session_id] = queue.Queue()
        
        # Create WebSocket connection
        url = f"{self.base_url}/ws/updates"
        
        try:
            ws = websocket.WebSocketApp(
                url,
                on_open=lambda ws: self._on_open(ws, session_id),
                on_message=lambda ws, msg: self._on_message(ws, msg, session_id),
                on_error=lambda ws, err: self._on_error(ws, err, session_id),
                on_close=lambda ws, close_status_code, close_msg: self._on_close(ws, close_status_code, close_msg, session_id),
            )
            
            # Store the connection
            self.connections[session_id] = ws
            
            # Start the WebSocket connection in a background thread
            threading.Thread(target=ws.run_forever, daemon=True).start()
            
            logger.info(f"Started WebSocket connection for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {str(e)}")
            return False
    
    def disconnect(self, session_id: str) -> None:
        """
        Disconnect the WebSocket for the given session ID.
        
        Args:
            session_id: Session ID for the WebSocket connection
        """
        if session_id in self.connections:
            try:
                self.connections[session_id].close()
            except:
                pass
            del self.connections[session_id]
            logger.info(f"Disconnected WebSocket for session {session_id}")
    
    def get_messages(self, session_id: str, max_messages: int = 10) -> list:
        """
        Get messages from the queue for the given session ID.
        
        Args:
            session_id: Session ID for the WebSocket connection
            max_messages: Maximum number of messages to retrieve
            
        Returns:
            List of messages
        """
        if session_id not in self.message_queues:
            return []
            
        messages = []
        try:
            for _ in range(max_messages):
                if self.message_queues[session_id].empty():
                    break
                messages.append(self.message_queues[session_id].get_nowait())
        except queue.Empty:
            pass
            
        return messages
    
    def _on_open(self, ws, session_id):
        """Handle WebSocket connection open."""
        logger.info(f"WebSocket opened for session {session_id}")
        # Send session ID to the server
        ws.send(json.dumps({"session_id": session_id}))
    
    def _on_message(self, ws, message, session_id):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            # Put the message in the queue for the appropriate session
            if session_id in self.message_queues:
                self.message_queues[session_id].put(data)
                logger.debug(f"Queued message for session {session_id}: {data['type']}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {str(e)}")
    
    def _on_error(self, ws, error, session_id):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error for session {session_id}: {str(error)}")
        # Put an error message in the queue
        if session_id in self.message_queues:
            self.message_queues[session_id].put({
                "type": "error",
                "payload": {"message": f"WebSocket error: {str(error)}"}
            })
    
    def _on_close(self, ws, close_status_code, close_msg, session_id):
        """Handle WebSocket connection close."""
        logger.info(f"WebSocket closed for session {session_id}: {close_status_code} - {close_msg}")
        # Clean up
        if session_id in self.connections:
            del self.connections[session_id]

# Create a global instance
ws_client = WebSocketClient()
