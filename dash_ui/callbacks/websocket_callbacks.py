import logging
import json
import queue
import threading
import websocket
import dash
from dash import Input, Output, State, callback

logger = logging.getLogger(__name__)

# Global variables
message_queues = {}
ws_connections = {}

def on_message(ws, message, session_id):
    """Handle incoming WebSocket messages."""
    try:
        data = json.loads(message)
        # Put the message in the queue for the appropriate session
        if session_id in message_queues:
            message_queues[session_id].put(data)
            logger.debug(f"Queued message for session {session_id}: {data['type']}")
    except Exception as e:
        logger.error(f"Error processing WebSocket message: {str(e)}")

def on_error(ws, error, session_id):
    """Handle WebSocket errors."""
    logger.error(f"WebSocket error for session {session_id}: {str(error)}")
    # Put an error message in the queue
    if session_id in message_queues:
        message_queues[session_id].put({
            "type": "error",
            "payload": {"message": f"WebSocket error: {str(error)}"}
        })

def on_close(ws, close_status_code, close_msg, session_id):
    """Handle WebSocket connection close."""
    logger.info(f"WebSocket closed for session {session_id}: {close_status_code} - {close_msg}")
    # Clean up
    if session_id in ws_connections:
        del ws_connections[session_id]

def on_open(ws, session_id):
    """Handle WebSocket connection open."""
    logger.info(f"WebSocket opened for session {session_id}")
    # Send session ID to the server
    ws.send(json.dumps({"session_id": session_id}))

def create_websocket_connection(session_id, websocket_url):
    """Create a WebSocket connection for the given session ID."""
    if session_id in ws_connections and ws_connections[session_id].sock and ws_connections[session_id].sock.connected:
        logger.info(f"WebSocket connection already exists for session {session_id}")
        return
        
    # Create a message queue for this session
    if session_id not in message_queues:
        message_queues[session_id] = queue.Queue()
    
    # Create WebSocket connection
    ws = websocket.WebSocketApp(
        websocket_url,
        on_open=lambda ws: on_open(ws, session_id),
        on_message=lambda ws, msg: on_message(ws, msg, session_id),
        on_error=lambda ws, err: on_error(ws, err, session_id),
        on_close=lambda ws, close_status_code, close_msg: on_close(ws, close_status_code, close_msg, session_id),
    )
    
    # Store the connection
    ws_connections[session_id] = ws
    
    # Start the WebSocket connection in a background thread
    threading.Thread(target=ws.run_forever, daemon=True).start()
    
    logger.info(f"Started WebSocket connection for session {session_id}")

def register_websocket_callbacks(app):
    """Register WebSocket-related callbacks."""
    
    @callback(
        Output('ws-messages', 'data'),
        Input('message-poller', 'n_intervals'),
        State('session-id', 'data'),
        State('ws-messages', 'data'),
        prevent_initial_call=True
    )
    def poll_messages(n_intervals, session_id, current_messages):
        """Poll for new messages from the WebSocket queue."""
        if session_id not in message_queues:
            return dash.no_update
            
        # Get all available messages from the queue
        new_messages = []
        try:
            while not message_queues[session_id].empty():
                new_messages.append(message_queues[session_id].get_nowait())
        except queue.Empty:
            pass
            
        if not new_messages:
            return dash.no_update
            
        # Add new messages to the current messages
        return current_messages + new_messages
