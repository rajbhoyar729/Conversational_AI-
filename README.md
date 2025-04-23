# Conversational AI Application

A flexible conversational AI application that supports multiple LLM providers (Google Gemini and Groq with Llama models) with a FastAPI backend and Gradio frontend.

## Features

- **Multiple LLM Providers**: Switch between Google Gemini and Groq (Llama models)
- **Streaming Support**: Real-time token streaming for a better user experience
- **Flexible API**: Both REST and WebSocket endpoints
- **Gradio UI**: User-friendly chat interface
- **Docker Support**: Easy deployment with Docker

## Project Structure

\`\`\`
conversational_ai_app/
├── backend/
│   ├── app/
│   │   ├── main.py         # FastAPI server entrypoint
│   │   ├── config.py       # env-based LLM selection config
│   │   ├── routers/
│   │   │   └── chat.py     # chat endpoints
│   │   ├── services/
│   │   │   ├── llm_client.py   # wrapper for LLM APIs
│   │   │   └── agent.py        # conversational agent logic
│   │   └── models/
│   │       └── schemas.py  # Pydantic schemas
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py            # Gradio interface
│   └── requirements.txt
├── .env.example
└── README.md
\`\`\`

## Setup

### Environment Variables

Create a `.env` file in the project root with the following variables:

\`\`\`
# LLM Provider (gemini or groq)
LLM_PROVIDER=gemini

# Groq settings (for Llama models)
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-70b-instant

# Google Gemini settings
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-pro

# Debug mode
DEBUG=false
\`\`\`

### Running Locally

#### Backend

1. Navigate to the backend directory:
   \`\`\`
   cd conversational_ai_app/backend
   \`\`\`

2. Install dependencies:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`

3. Run the FastAPI server:
   \`\`\`
   uvicorn app.main:app --reload
   \`\`\`

The API will be available at http://localhost:8000

#### Frontend

1. Navigate to the frontend directory:
   \`\`\`
   cd conversational_ai_app/frontend
   \`\`\`

2. Install dependencies:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`

3. Run the Gradio app:
   \`\`\`
   python app.py
   \`\`\`

The UI will be available at http://localhost:7860

### Running with Docker

1. Build the Docker image for the backend:
   \`\`\`
   docker build -t conversational-ai-backend .
   \`\`\`

2. Run the container:
   \`\`\`
   docker run -p 8000:8000 --env-file .env conversational-ai-backend
   \`\`\`

## Switching LLM Providers

You can switch between different LLM providers by changing the `LLM_PROVIDER` environment variable:

- `gemini`: Uses Google Gemini models
- `groq`: Uses Groq API with Llama models

Make sure you have the appropriate API keys set in your environment variables.

## Available Models

### Groq (Llama Models)
- `llama-3.1-70b-instant`: Llama 3.1 70B model (fastest)
- `llama-3.1-70b`: Llama 3.1 70B model
- `llama-3.1-8b`: Llama 3.1 8B model
- `llama-3-70b-instructional`: Llama 3 70B model
- `llama-3-8b-instructional`: Llama 3 8B model

### Gemini
- `gemini-1.5-pro`: Gemini 1.5 Pro model
- `gemini-1.5-flash`: Gemini 1.5 Flash model (faster)

## API Endpoints

### REST API

- `POST /chat`: Send a chat message and receive a complete response
- `GET /chat/stream`: Stream a chat response using server-sent events

### WebSocket API

- `WebSocket /chat/ws`: Bidirectional chat communication with streaming support

## Development

### Adding a New LLM Provider

To add support for a new LLM provider:

1. Create a new adapter class in `app/services/llm_client.py` that implements the `LLMClientAdapter` interface
2. Add the new provider to the `LLMProvider` enum in `app/config.py`
3. Update the `get_llm_client` factory function to handle the new provider

## License

MIT
