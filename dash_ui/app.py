import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify

from components.chat_interface import create_chat_interface, render_chat_message
from components.settings_panel import create_settings_panel
from callbacks import register_callbacks
from utils.logging import setup_logging

# Setup logging
logger = setup_logging()

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ],
    suppress_callback_exceptions=True,
    title="Conversational AI Chat",
    update_title="Updating...",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

# App layout
app.layout = html.Div([
    # Stores for app state
    dcc.Store(id='session-id', data=''),
    dcc.Store(id='chat-history', data=[]),
    dcc.Store(id='current-response', data=""),
    dcc.Store(id='ws-messages', data=[]),
    dcc.Interval(id='message-poller', interval=100),  # Poll for new messages every 100ms
    
    # Main container
    dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.Div([
                    DashIconify(icon="mdi:robot-excited", width=40, className="me-3"),
                    html.H1("Conversational AI", className="mb-0")
                ], className="d-flex align-items-center justify-content-center my-4")
            ])
        ]),
        
        # Main content
        dbc.Row([
            # Sidebar with settings panel
            dbc.Col([
                create_settings_panel()
            ], xs=12, md=4, lg=3, className="mb-4"),
            
            # Main chat area
            dbc.Col([
                create_chat_interface()
            ], xs=12, md=8, lg=9)
        ]),
        
        # Footer
        dbc.Row([
            dbc.Col([
                html.Footer([
                    html.P([
                        "Built with ",
                        DashIconify(icon="mdi:heart", width=16, color="#ff6b6b"),
                        " using Dash and FastAPI"
                    ], className="text-center text-muted my-4")
                ])
            ])
        ])
    ], fluid=True, className="main-container")
])

# Register all callbacks
register_callbacks(app)

# Run the app
if __name__ == '__main__':
    # Generate a session ID on startup
    import uuid
    session_id = str(uuid.uuid4())
    logger.info(f"Starting Dash app with session ID: {session_id}")
    
    # Run the app
    app.run(debug=True, port=8050)  # Updated to app.run