import dash_bootstrap_components as dbc
from dash import html, dcc
from dash_iconify import DashIconify

def create_settings_panel():
    """Create the settings panel component."""
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    DashIconify(icon="mdi:cog", width=24, className="me-2"),
                    html.H4("Settings", className="mb-0 d-inline")
                ], className="d-flex align-items-center")
            ], className="card-header-gradient gradient-secondary"),
            
            dbc.CardBody([
                html.Div([
                    html.Label([
                        DashIconify(icon="mdi:robot", width=20, className="me-2"),
                        "LLM Provider"
                    ], className="d-flex align-items-center mb-2"),
                    dcc.Dropdown(
                        id='provider-dropdown',
                        options=[
                            {'label': 'Groq (Llama)', 'value': 'groq'},
                            {'label': 'Google Gemini', 'value': 'gemini'}
                        ],
                        value='groq',
                        clearable=False,
                        className="mb-3"
                    ),
                    html.Div(id='provider-info', className="mt-2 mb-3 fade-in"),
                ], className="mb-4"),
                
                html.Hr(className="my-4"),
                
                html.Div([
                    html.Label([
                        DashIconify(icon="mdi:tune", width=20, className="me-2"),
                        "Advanced Settings"
                    ], className="d-flex align-items-center mb-3"),
                    
                    dbc.Switch(
                        id="use-langchain-switch",
                        label="Use Langchain",
                        value=False,
                        className="mb-2"
                    ),
                    
                    html.Small(
                        "Enable Langchain for enhanced capabilities like agents and tools.",
                        className="text-muted"
                    )
                ])
            ])
        ], className="card-gradient mb-4"),
        
        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    DashIconify(icon="mdi:connection", width=24, className="me-2"),
                    html.H4("Connection", className="mb-0 d-inline")
                ], className="d-flex align-items-center")
            ], className="card-header-gradient gradient-dark"),
            
            dbc.CardBody([
                html.Div([
                    DashIconify(
                        id="connection-icon",
                        icon="mdi:connection-off",
                        width=24,
                        className="me-2"
                    ),
                    html.Div(
                        id='connection-status',
                        children="Disconnected",
                        className="status-disconnected"
                    )
                ], className="d-flex align-items-center mb-3"),
                
                dbc.Button([
                    DashIconify(icon="mdi:connection", width=20, className="me-1"),
                    "Connect"
                ],
                id='connect-button',
                className="btn-gradient w-100"
                ),
            ])
        ], className="card-gradient")
    ])
