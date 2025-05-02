"""Callbacks for the Dash application."""
from .chat_callbacks import register_chat_callbacks
from .websocket_callbacks import register_websocket_callbacks

def register_callbacks(app):
    """Register all callbacks for the application."""
    register_chat_callbacks(app)
    register_websocket_callbacks(app)
