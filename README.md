# Conversational AI Application  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A flexible conversational AI application featuring a FastAPI backend and a Gradio frontend, supporting multiple LLM providers (Google Gemini, Groq Llama) with streaming capabilities.

## Table of Contents
- [Features](#features)  
- [Technology Stack](#technology-stack)  
- [Project Structure](#project-structure)  
- [Prerequisites](#prerequisites)  
- [Configuration](#configuration)  
- [Installation](#installation)  
- [Setup Instructions](#setup-instructions)  
  - [Backend Setup](#backend-setup)  
  - [Frontend Setup](#frontend-setup)  
- [Running the Application](#running-the-application)  
  - [Locally](#locally)  
  - [Using Docker Compose](#using-docker-compose)  
- [Usage](#usage)  
- [API Endpoints](#api-endpoints)  
- [Supported LLMs & Switching Providers](#supported-llms--switching-providers)  
- [Development](#development)  
  - [Adding a New LLM Provider](#adding-a-new-llm-provider)  
- [License](#license)  

## Features
- **Multiple LLM Provider Support**: Easily switch between Google Gemini and Groq (Llama).  
- **Real-time Streaming**: Token streaming via SSE and WebSockets.  
- **Asynchronous API**: FastAPI + Uvicorn for high performance.  
- **Flexible Endpoints**: REST (POST), streaming REST (SSE), WebSocket.  
- **User-Friendly UI**: Interactive Gradio chat interface.  
- **Configuration-Driven**: Environment variables + `.env` files.  
- **Docker Support**: `Dockerfile` + sample `docker-compose.yml`.

## Technology Stack
- **Backend**: Python, FastAPI, Uvicorn, Pydantic  
- **Frontend**: Python, Gradio  
- **LLM Integration**: `google-generativeai`, `groq`, `websockets`  
- **Config**: `pydantic-settings`  
- **Containerization**: Docker, Docker Compose  

## Project Structure
```
conversational-ai-app/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI entrypoint & setup
│   │   ├── config.py          # Pydantic Settings + get_settings()
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   └── chat.py         # /chat, /chat/stream, /chat/ws
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── llm_client.py   # BaseLLMClient, adapters, factory
│   │   │   └── agent.py        # Stateless ConversationalAgent
│   │   └── models/
│   │       ├── __init__.py
│   │       └── schemas.py      # Pydantic schemas (Message, ChatRequest, etc.)
│   ├── Dockerfile             # Backend containerization
│   └── requirements.txt       # Backend deps
├── frontend/
│   ├── app.py                 # Gradio UI
│   └── requirements.txt       # Frontend deps
├── .env.example               # Env template
├── docker-compose.yml         # Sample Docker Compose
└── README.md                  # This file
```

## Prerequisites
- Python 3.9+  
- pip  
- Docker & Docker Compose (for containerized deployment)  
- API Keys for your chosen LLM providers  
- Microsoft Visual Studio Build Tools (ensure vswhere.exe is installed)  

## Configuration
1. Copy the example:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env`:
   ```dotenv
   LLM_PROVIDER=gemini
   DEBUG=false

   # For Groq:
   GROQ_API_KEY="your_groq_api_key"
   GROQ_MODEL="llama-3.1-70b-instant"

   # For Gemini:
   GEMINI_API_KEY="your_gemini_api_key"
   GEMINI_MODEL="gemini-1.5-pro"

   # Frontend (if needed):
   API_BASE_URL=http://backend:8000
   WS_BASE_URL=ws://backend:8000
   ```

## Installation
```bash
git clone <your-repo-url>
cd conversational-ai-app

# Backend
cd backend
pip install -r requirements.txt
cd ..

# Frontend
cd frontend
pip install -r requirements.txt
cd ..
```

## Setup Instructions

### Backend Setup
1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in `backend/app/` with the following content:
   ```env
   ACTIVE_LLM=gpt-4
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

### Frontend Setup
1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Locally
1. **Backend**  
   ```bash
   cd backend
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   → API at `http://localhost:8000`, docs at `/docs`.

2. **Frontend**  
   ```bash
   cd frontend
   python app.py
   ```
   → UI at `http://localhost:7860`.

### Using Docker Compose
```yaml
version: '3.8'
services:
  backend:
    build:
      context: ./backend
    env_file: .env
    ports:
      - "8000:8000"
  frontend:
    build:
      context: ./frontend
    environment:
      - API_BASE_URL=http://backend:8000
      - WS_BASE_URL=ws://backend:8000
    ports:
      - "7860:7860"
    depends_on:
      - backend
```
```bash
docker-compose up --build
```
Access UI at `localhost:7860`, API at `localhost:8000`.

### Switching LLMs
To switch the active LLM, update the `ACTIVE_LLM` variable in the `.env` file to one of the following:
- `gpt-4`
- `gemini`
- `claude`

Restart the backend server after making changes.

## Usage
- **Gradio UI**: Browse to `localhost:7860`.  
- **API**: Use Swagger at `localhost:8000/docs` or call endpoints directly.

## API Endpoints
- **GET /** — health check: returns status & provider.  
- **POST /chat** — non-streaming chat (JSON response).  
- **POST /chat/stream** — SSE streaming (each `data:` line is a `StreamToken`).  
- **WebSocket /chat/ws** — bidirectional chat (JSON messages).

## Supported LLMs & Switching Providers
Set `LLM_PROVIDER` in `.env`:
- **gemini**: requires `GEMINI_API_KEY`, default `gemini-1.5-pro`.  
- **groq**: requires `GROQ_API_KEY`, default `llama-3.1-70b-instant`.  

## Development

### Adding a New LLM Provider
1. Implement adapter in `backend/app/services/llm_client.py` subclassing `BaseLLMClient`.  
2. Add enum to `LLMProvider` in `config.py`.  
3. Update `get_llm_client()` factory.  
4. Add fields to `Settings` and update `.env.example`.

## License
This project is licensed under MIT — see [LICENSE](LICENSE).
