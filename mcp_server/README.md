# Conversational AI - Model Control Plane (MCP)

This is the backend service for the Conversational AI application. It provides a FastAPI server that handles LLM interactions and WebSocket connections for real-time updates.

## Features

- FastAPI backend with WebSocket support
- Support for multiple LLM providers (Groq, Gemini)
- Optional Langchain integration
- Real-time streaming of LLM responses

## Setup

### Prerequisites

- Python 3.9+
- Virtual environment (recommended)

### Installation

1. Clone the repository
2. Create and activate a virtual environment:
   \`\`\`
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   \`\`\`
3. Install dependencies:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`
4. Copy `.env.example` to `.env` and fill in your API keys:
   \`\`\`
   cp .env.example .env
   \`\`\`

### Running the Server

\`\`\`
uvicorn app.main:app --reload
\`\`\`

The server will be available at http://localhost:8000.

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| GROQ_API_KEY | Groq API key | None |
| GOOGLE_API_KEY | Google API key | None |
| DEFAULT_LLM_PROVIDER | Default LLM provider | groq |
| USE_LANGCHAIN | Use Langchain for LLM interactions | false |
| HOST | Server host | 0.0.0.0 |
| PORT | Server port | 8000 |
| LOG_LEVEL | Logging level | INFO |
| CORS_ORIGINS | Allowed CORS origins | ["*"] |

## Docker

Build and run the Docker container:

\`\`\`
docker build -t mcp-server .
docker run -p 8000:8000 --env-file .env mcp-server
