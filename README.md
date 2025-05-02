
## Prerequisites
- Python 3.9+ (Python 3.10 or higher recommended)
- `pip` for package installation
- `git` for cloning the repository
- API keys for both Google Gemini and Groq.
    *   Get a Google Gemini API key [here](https://aistudio.google.com/app/apikey).
    *   Get a Groq API key [here](https://console.groq.com/keys).

## Configuration

The application uses environment variables, primarily managed via a `.env` file in the `mcp_server` directory.

1.  Navigate to the `mcp_server` directory:
    ```bash
    cd mcp_server
    ```
2.  Copy the example environment file:
    ```bash
    # On Windows:
    copy .env.example .env
    # On macOS/Linux:
    cp .env.example .env
    ```
3.  Edit the newly created `.env` file (`mcp_server/.env`) and replace the placeholder values with your actual API keys:
    ```dotenv
    # .env file inside mcp_server/
    # Choose the LLM provider: 'groq' or 'gemini'
    LLM_PROVIDER=groq

    # API Keys - Replace with your actual keys
    GROQ_API_KEY="YOUR_ACTUAL_GROQ_KEY"
    GOOGLE_API_KEY="YOUR_ACTUAL_GEMINI_KEY"

    # Optional: WebSocket host and port (defaults are usually fine)
    # WS_HOST=0.0.0.0
    # WS_PORT=8000
    ```
    **Important:** Do not commit your `.env` file containing sensitive keys to Git. The `.gitignore` file is configured to ignore it.
4.  Return to the project root:
    ```bash
    cd ..
    ```

## Setup

### Backend (MCP Server)

1.  Navigate to the `mcp_server` directory:
    ```bash
    cd mcp_server
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```
3.  Install backend dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Ensure your `.env` file is configured as described in the [Configuration](#configuration) section.
5.  Return to the project root:
    ```bash
    cd ..
    ```

### Frontend (Streamlit UI)

1.  Navigate to the `streamlit_ui` directory:
    ```bash
    cd streamlit_ui
    ```
2.  Create and activate a **separate** virtual environment:
    ```bash
    python -m venv venv
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```
3.  Install frontend dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Return to the project root:
    ```bash
    cd ..
    ```

## Running the Application

The backend and frontend run as separate processes.

1.  **Start the MCP Server (Backend):**
    *   Open a **new** terminal or command prompt.
    *   Navigate to the `mcp_server` directory.
    *   Activate the backend virtual environment.
    *   Run the server:
        ```bash
        uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --ws-ping-interval 20 --ws-ping-timeout 20
        ```
    *   Keep this terminal open and running.
2.  **Start the Streamlit UI (Frontend):**
    *   Open another **new** terminal or command prompt.
    *   Navigate to the `streamlit_ui` directory.
    *   Activate the frontend virtual environment.
    *   Run the Streamlit app:
        ```bash
        streamlit run app_ui.py
        ```
    *   This will open the application in your web browser (usually at `http://localhost:8501` by default, but check the terminal output).

## LLM Switching

To switch the active LLM provider:

1.  Stop the MCP Server (Ctrl+C in the server terminal).
2.  Edit the `mcp_server/.env` file.
3.  Change the line `LLM_PROVIDER=groq` to `LLM_PROVIDER=gemini` (or vice versa).
4.  Save the `.env` file.
5.  Restart the MCP Server using the command from the "Running the Application" section.
6.  In the Streamlit app in your browser, click the "Disconnect" button (if connected), then click the "Connect" button to establish a new WebSocket connection with the reconfigured backend. The chat will now use the newly selected LLM.

## API Endpoints

The FastAPI backend exposes the following endpoints:

-   **GET `/`**: A basic health check endpoint. Returns JSON indicating the server is running and the configured LLM provider.
-   **WebSocket `/ws/chat`**: The primary endpoint for chat interactions. Handles bidirectional communication for sending user messages and receiving streamed AI responses.

*(Note: Unlike the template, this implementation focuses on WebSocket for chat streaming and does not include separate non-streaming or SSE `/chat` endpoints).*

## Development

### Adding a New LLM Provider

To integrate an additional LLM provider:

1.  **Implement Integration Module:** Create a new Python file (e.g., `new_llm.py`) in `mcp_server/app/llm_integrations/`. Implement an asynchronous function (similar to `get_groq_response_stream` or `get_gemini_response_stream`) that takes messages and necessary API keys/configs, and yields string chunks from the provider's streaming API. Add relevant imports and error handling.
2.  **Update Configuration:**
    *   Add a field for the new provider's API key (and potentially model name) to the `Settings` class in `mcp_server/app/config.py`.
    *   Add the provider's name to the `Literal` type hint for `llm_provider` in `config.py`.
    *   Update `mcp_server/.env.example` with placeholders for the new provider's settings.
    *   Update your `mcp_server/.env` file with the actual API key if needed.
3.  **Update Chat Service:**
    *   Import the new integration module in `mcp_server/app/services/chat_service.py`.
    *   Add an `elif` block in the `generate_response_stream` function to check for the new provider's name and call its streaming function, passing the required configuration.
4.  **Update Requirements:** Add the new provider's SDK to `mcp_server/requirements.txt` and install it (`pip install -r requirements.txt`).
5.  **Update Documentation:** Add notes about the new provider to this README.

## Deliverables Checklist

Based on the assignment requirements, ensure you have the following:

*   [ ] Source code for the FastAPI MCP server (`mcp_server/` directory).
*   [ ] Source code for the Streamlit frontend (`streamlit_ui/` directory).
*   [ ] AI agent logic & LLM integration modules (`mcp_server/app/services/`, `mcp_server/app/llm_integrations/`).
*   [ ] `README.md` file (this file) with setup, LLM switching guide, and running steps.
*   [ ] Brief demo video (screen-capture) showing the app in action (placed in `demo_video/` or linked).

## How to Submit

Please follow the specific submission instructions provided in the original assignment document. This typically involves:

*   Packaging your project folder (`conversational-ai-app`) into a ZIP archive or sharing a link to your Git repository.
*   Emailing the ZIP or link to `info@yun.buzz`.
*   Using the specific subject line: `"LinkedIn - Software Developer Assignment"`.
*   Including your full name and LinkedIn profile URL in the email body.
*   Adding any setup instructions not covered in this README to the email body (though this README aims to be comprehensive).

---

*Looking forward to seeing your creativity in action!*
*- Team Yuñ*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.