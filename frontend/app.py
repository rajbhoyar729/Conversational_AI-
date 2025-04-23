import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import gradio as gr
import httpx
import websockets
from pydantic import BaseModel, Field

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
WS_BASE_URL = os.getenv("WS_BASE_URL", "ws://localhost:8000").replace("http", "ws")
HTTP_TIMEOUT = 60.0

class Message(BaseModel):
    role: str
    content: str

class AsyncChatClient:
    def __init__(self, base_url: str, ws_base_url: str, timeout: float = HTTP_TIMEOUT):
        self.base_url = base_url
        self.ws_base_url = ws_base_url
        self.timeout = timeout
        self._http_client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def close(self):
        await self._http_client.aclose()

    async def get_server_info(self) -> Dict[str, Any]:
        try:
            response = await self._http_client.get("/")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to API at {self.base_url}/: {e}")
            return {
                "status": "error",
                "message": f"Connection Error: Could not reach backend at {self.base_url}",
                "llm_provider": "Unknown",
                "debug_mode": False
            }
        except Exception as e:
            logger.error(f"Error fetching server info: {e}", exc_info=True)
            return {"status": "error", "message": f"Error fetching info: {e}"}

    async def get_non_streaming_response(
        self, messages: List[Message], temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        payload = {
            "messages": [msg.model_dump() for msg in messages],
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        try:
            response = await self._http_client.post("/chat", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_detail = e.response.json().get("detail", error_detail)
            except json.JSONDecodeError:
                pass
            logger.error(f"HTTP error from /chat: {e.response.status_code} - {error_detail}")
            return {"message": {"role": "assistant", "content": f"API Error ({e.response.status_code}): {error_detail}"}}
        except httpx.RequestError as e:
            logger.error(f"Request error contacting /chat: {e}")
            return {"message": {"role": "assistant", "content": "Connection Error: Failed to reach API."}}
        except Exception as e:
            logger.error(f"Unexpected error in get_non_streaming_response: {e}", exc_info=True)
            return {"message": {"role": "assistant", "content": f"Client Error: {e}"}}

    async def get_streaming_response(
        self, messages: List[Message], temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        payload = {
            "messages": [msg.model_dump() for msg in messages],
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        try:
            async with self._http_client.stream("POST", "/chat/stream", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        try:
                            data_str = line[len("data:"):].strip()
                            if data_str:
                                data = json.loads(data_str)
                                yield data
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode SSE data: {data_str}")
                            continue
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_detail = e.response.json().get("detail", error_detail)
            except json.JSONDecodeError:
                pass
            logger.error(f"HTTP error from /chat/stream: {e.response.status_code} - {error_detail}")
            yield {"token": f"API Error ({e.response.status_code}): {error_detail}", "is_finished": True, "finish_reason": "error"}
        except httpx.RequestError as e:
            logger.error(f"Request error contacting /chat/stream: {e}")
            yield {"token": "Connection Error: Failed to reach API.", "is_finished": True, "finish_reason": "error"}
        except Exception as e:
            logger.error(f"Unexpected error in get_streaming_response: {e}", exc_info=True)
            yield {"token": f"Client Error: {e}", "is_finished": True, "finish_reason": "error"}

    async def get_websocket_response(
        self, messages: List[Message], temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        ws_url = f"{self.ws_base_url}/chat/ws"
        payload = {
            "messages": [msg.model_dump() for msg in messages],
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        try:
            async with websockets.connect(ws_url) as websocket:
                await websocket.send(json.dumps(payload))
                while True:
                    response_str = await websocket.recv()
                    data = json.loads(response_str)
                    yield data
                    if data.get("is_finished", False):
                        break
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK) as e:
            logger.info(f"WebSocket connection closed: {e}")
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error connecting to {ws_url}: {e}", exc_info=True)
            yield {"token": f"WebSocket Error: {e}", "is_finished": True, "finish_reason": "error"}
        except Exception as e:
            logger.error(f"Unexpected error in get_websocket_response: {e}", exc_info=True)
            yield {"token": f"Client Error: {e}", "is_finished": True, "finish_reason": "error"}

chat_client = AsyncChatClient(API_BASE_URL, WS_BASE_URL)

async def process_chat_interaction(
    message_text: str,
    history: List[List[str | None]],
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
    use_streaming: bool,
    use_websockets: bool
) -> AsyncGenerator[List[List[str | None]], None]:
    if not message_text:
        yield history
        return

    messages: List[Message] = []
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append(Message(role="user", content=user_msg))
        if assistant_msg:
            messages.append(Message(role="assistant", content=assistant_msg))
    messages.append(Message(role="user", content=message_text))

    history.append([message_text, None])
    yield history

    effective_max_tokens = max_tokens if max_tokens and max_tokens > 0 else None

    assistant_response = ""
    error_occurred = False

    try:
        if use_streaming:
            client_method = (
                chat_client.get_websocket_response if use_websockets else chat_client.get_streaming_response
            )
            async for chunk_data in client_method(
                messages, temperature=temperature, max_tokens=effective_max_tokens
            ):
                token = chunk_data.get("token", "")
                assistant_response += token
                history[-1][1] = assistant_response
                yield history
                if chunk_data.get("is_finished", False):
                    if chunk_data.get("finish_reason") == "error":
                        logger.error(f"Stream finished with error: {token}")
                        error_occurred = True
                    break
        else:
            response_data = await chat_client.get_non_streaming_response(
                messages, temperature=temperature, max_tokens=effective_max_tokens
            )
            if "message" in response_data and response_data["message"]["role"] == "assistant" and "Error:" in response_data["message"]["content"]:
                assistant_response = response_data["message"]["content"]
                error_occurred = True
            elif "message" in response_data:
                assistant_response = response_data["message"]["content"]
            else:
                assistant_response = "Error: Received unexpected response format from API."
                error_occurred = True
            history[-1][1] = assistant_response
            yield history
    except Exception as e:
        logger.error(f"Error during chat processing: {e}", exc_info=True)
        history[-1][1] = f"An unexpected error occurred in the Gradio client: {e}"
        error_occurred = True
        yield history

    if not error_occurred:
        logger.info("Assistant response generated successfully.")

async def create_ui() -> gr.Blocks:
    server_info = await chat_client.get_server_info()
    provider = server_info.get('llm_provider', 'Unknown')
    status_message = server_info.get('message', 'Could not determine server status.')
    server_status_color = "red" if server_info.get('status') == 'error' else "green"

    with gr.Blocks(title="Conversational AI Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Conversational AI Chat Interface")
        gr.Markdown(f"Backend Status: <span style='color:{server_status_color};'>{status_message}</span> (Provider:
