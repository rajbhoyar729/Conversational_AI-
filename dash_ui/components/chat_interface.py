import dash_bootstrap_components as dbc
from dash import html, dcc
from dash_iconify import DashIconify

def create_chat_interface():
    """Create the chat interface component."""
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                DashIconify(icon="mdi:chat-processing", width=24, className="me-2"),
                html.H3("Chat", className="mb-0 d-inline")
            ], className="d-flex align-items-center")
        ], className="card-header-gradient gradient-primary"),
        
        dbc.CardBody([
            html.Div(
                id='chat-messages',
                className="d-flex flex-column",
                style={
                    'height': '60vh',
                    'overflowY': 'auto',
                    'padding': '10px',
                }
            ),
        ], style={"background": "linear-gradient(135deg, #ffffff, #f8f9fa)"}),
        
        dbc.CardFooter([
            dbc.InputGroup([
                dbc.Input(
                    id='message-input',
                    placeholder='Type your message...',
                    type='text',
                    className="input-custom",
                    style={"border-radius": "8px 0 0 8px"}
                ),
                dbc.Button([
                    DashIconify(icon="mdi:send", width=20),
                    html.Span("Send", className="ms-1")
                ], 
                id='send-button', 
                className="btn-gradient",
                style={"border-radius": "0 8px 8px 0"}
                ),
            ]),
            html.Div(id='send-status', className="mt-2 fade-in")
        ], className="gradient-light")
    ], className="card-gradient")

def render_chat_message(message, is_user=False):
    """Render a chat message with appropriate styling."""
    if is_user:
        return html.Div([
            html.Div([
                html.Div(message, className="message-content")
            ], className="message-user fade-in")
        ], className="d-flex justify-content-end mb-3")
    else:
        return html.Div([
            html.Div([
                html.Div(message, className="message-content")
            ], className="message-assistant fade-in")
        ], className="d-flex justify-content-start mb-3")
