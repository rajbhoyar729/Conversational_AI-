import logging
import json
import dash
from dash import Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify

from utils.api_client import send_chat_request
from components.chat_interface import render_chat_message

logger = logging.getLogger(__name__)

def register_chat_callbacks(app):
    """Register chat-related callbacks."""
    
    @callback(
        Output('send-status', 'children'),
        Output('message-input', 'value'),
        Output('chat-history', 'data'),
        Output('current-response', 'data', allow_duplicate=True),
        Input('send-button', 'n_clicks'),
        Input('message-input', 'n_submit'),
        State('message-input', 'value'),
        State('provider-dropdown', 'value'),
        State('session-id', 'data'),
        State('chat-history', 'data'),
        State('use-langchain-switch', 'value'),
        prevent_initial_call=True
    )
    def send_message(n_clicks, n_submit, message, provider, session_id, chat_history, use_langchain):
        """Send a message to the backend when the send button is clicked or Enter is pressed."""
        if (not n_clicks and not n_submit) or not message:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        try:
            # Add user message to chat history
            updated_history = chat_history + [{"role": "user", "content": message}]
            
            # Send request to backend
            response = send_chat_request(
                message=message,
                history=updated_history,
                provider=provider,
                session_id=session_id,
                use_langchain=use_langchain
            )
            
            if response.status_code == 200:
                return [
                    DashIconify(icon="mdi:check-circle", width=16, className="me-1"),
                    "Message sent"
                ], "", updated_history, ""
            else:
                return [
                    DashIconify(icon="mdi:alert-circle", width=16, className="me-1"),
                    f"Error: {response.text}"
                ], dash.no_update, updated_history, dash.no_update
                
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            return [
                DashIconify(icon="mdi:alert-circle", width=16, className="me-1"),
                f"Error: {str(e)}"
            ], dash.no_update, dash.no_update, dash.no_update
    
    @callback(
        Output('chat-messages', 'children'),
        Output('current-response', 'data'),
        Input('ws-messages', 'data'),
        State('chat-history', 'data'),
        State('current-response', 'data'),
        prevent_initial_call=True
    )
    def process_ws_messages(messages, chat_history, current_response):
        """Process WebSocket messages and update the chat display."""
        if not messages:
            return dash.no_update, dash.no_update
            
        # Process only new messages
        new_messages = messages[-10:]  # Limit to last 10 messages for performance
        
        for msg in new_messages:
            msg_type = msg.get('type')
            payload = msg.get('payload', {})
            
            if msg_type == 'llm_chunk':
                # Append chunk to current response
                current_response += payload.get('chunk', '')
            elif msg_type == 'llm_final':
                # Final response received
                current_response = payload.get('complete_response', current_response)
            elif msg_type == 'error':
                # Handle error
                current_response += f"\n\nError: {payload.get('message', 'Unknown error')}"
        
        # Render chat messages
        chat_elements = []
        
        # Add user messages from history
        for msg in chat_history:
            if msg['role'] == 'user':
                chat_elements.append(render_chat_message(msg['content'], is_user=True))
            else:
                chat_elements.append(render_chat_message(msg['content'], is_user=False))
        
        # Add current response if it exists
        if current_response:
            chat_elements.append(render_chat_message(current_response, is_user=False))
        
        return chat_elements, current_response
    
    @callback(
        Output('connection-status', 'children'),
        Output('connection-status', 'className'),
        Output('connection-icon', 'icon'),
        Input('connect-button', 'n_clicks'),
        State('session-id', 'data'),
        prevent_initial_call=True
    )
    def connect_websocket(n_clicks, session_id):
        """Connect to the WebSocket when the connect button is clicked."""
        if not n_clicks:
            return "Disconnected", "status-disconnected", "mdi:connection-off"
            
        try:
            # Logic to connect to WebSocket would go here
            # For now, we'll just simulate a successful connection
            return "Connected", "status-connected", "mdi:connection"
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {str(e)}")
            return f"Error: {str(e)}", "status-disconnected", "mdi:connection-off"
