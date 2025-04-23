import gradio as gr
import requests
import json
import os
import asyncio
import websockets
import logging
from typing import List, Dict, Any, Generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
WS_BASE_URL = os.getenv("WS_BASE_URL", "ws://localhost:8000")

class ChatClient:
    """Client for interacting with the FastAPI backend"""
    
    @staticmethod
    def get_non_streaming_response(messages, temperature=0.7, max_tokens=None):
        """Get a complete response from the chat API"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={
                    "messages": messages,
                    "stream": False,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting chat response: {str(e)}")
            return {"message": {"role": "assistant", "content": f"Error: {str(e)}"}}
    
    @staticmethod
    def get_streaming_response(messages, temperature=0.7, max_tokens=None) -> Generator[str, None, None]:
        """Get a streaming response from the chat API"""
        try:
            # Convert messages to JSON string for URL parameter
            messages_json = json.dumps([msg.dict() for msg in messages])
            
            # Build URL with query parameters
            url = f"{API_BASE_URL}/chat/stream?messages={messages_json}"
            if temperature is not None:
                url += f"&temperature={temperature}"
            if max_tokens is not None:
                url += f"&max_tokens={max_tokens}"
            
            # Make request with streaming enabled
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                
                # Process server-sent events
                buffer = ""
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            if "token" in data:
                                yield data["token"]
                            if data.get("is_finished", False):
                                break
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield f"Error: {str(e)}"
    
    @staticmethod
    async def get_websocket_response(messages, temperature=0.7, max_tokens=None) -> Generator[str, None, None]:
        """Get a response via WebSocket"""
        try:
            async with websockets.connect(f"{WS_BASE_URL}/chat/ws") as websocket:
                # Send the request
                await websocket.send(json.dumps({
                    "messages": [msg.dict() for msg in messages],
                    "stream": True,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }))
                
                # Receive and yield chunks
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    if "token" in data:
                        yield data["token"]
                    
                    if data.get("is_finished", False):
                        break
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            yield f"Error: {str(e)}"


# Define message class for compatibility with the API
class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content
    
    def dict(self):
        return {"role": self.role, "content": self.content}


def format_message(role, content):
    """Format a message for display in the Gradio interface"""
    if role == "user":
        return f"👤 User: {content}"
    elif role == "assistant":
        return f"🤖 Assistant: {content}"
    elif role == "system":
        return f"⚙️ System: {content}"
    return f"{role.capitalize()}: {content}"


def chat_interface(message, history, system_prompt, temperature, max_tokens, use_streaming, model_info):
    """Handle chat interactions"""
    # Convert history to the format expected by the API
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append(Message("system", system_prompt))
    
    # Add conversation history
    for user_msg, assistant_msg in history:
        messages.append(Message("user", user_msg))
        if assistant_msg:  # Skip None responses
            messages.append(Message("assistant", assistant_msg))
    
    # Add the current message
    messages.append(Message("user", message))
    
    # Get response based on streaming preference
    if use_streaming:
        # For streaming, we need to yield partial responses
        partial_response = ""
        for token in ChatClient.get_streaming_response(
            messages, 
            temperature=temperature, 
            max_tokens=max_tokens if max_tokens > 0 else None
        ):
            partial_response += token
            yield partial_response
    else:
        # For non-streaming, we get the complete response at once
        response = ChatClient.get_non_streaming_response(
            messages,
            temperature=temperature,
            max_tokens=max_tokens if max_tokens > 0 else None
        )
        yield response["message"]["content"]


def create_ui():
    """Create the Gradio UI"""
    with gr.Blocks(title="Conversational AI Chat") as demo:
        gr.Markdown("# 🤖 Conversational AI Chat Interface")
        gr.Markdown("Connect to Gemini and Llama models through a unified interface")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=600,
                    show_copy_button=True,
                    bubble_full_width=False,
                    avatar_images=("👤", "🤖")
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your message here...",
                        container=False,
                        scale=8,
                        show_label=False
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Column(scale=1):
                system_prompt = gr.Textbox(
                    placeholder="Optional system prompt...",
                    label="System Prompt",
                    info="Instructions for the AI assistant"
                )
                
                model_info = gr.JSON(
                    value={
                        "provider": "Unknown",
                        "model": "Unknown"
                    },
                    label="Model Information",
                    visible=False
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Higher values make output more random, lower values more deterministic"
                    )
                    
                    max_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=0,
                        step=1,
                        label="Max Tokens",
                        info="Maximum length of response (0 = no limit)"
                    )
                    
                    use_streaming = gr.Checkbox(
                        value=True,
                        label="Enable Streaming",
                        info="Show response as it's being generated"
                    )
                
                clear_btn = gr.Button("Clear Conversation")
                
                with gr.Accordion("Model Information", open=True):
                    gr.Markdown("""
                    ### Available Models
                    
                    #### Groq (Llama Models)
                    - `llama-3.1-70b-instant`: Llama 3.1 70B model (fastest)
                    - `llama-3.1-70b`: Llama 3.1 70B model
                    - `llama-3.1-8b`: Llama 3.1 8B model
                    
                    #### Gemini
                    - `gemini-1.5-pro`: Gemini 1.5 Pro model
                    - `gemini-1.5-flash`: Gemini 1.5 Flash model (faster)
                    
                    To change the LLM provider, set the `LLM_PROVIDER` environment variable on the backend.
                    """)
        
        # Set up event handlers
        submit_btn.click(
            chat_interface,
            inputs=[msg, chatbot, system_prompt, temperature, max_tokens, use_streaming, model_info],
            outputs=[chatbot],
            queue=True
        )
        
        msg.submit(
            chat_interface,
            inputs=[msg, chatbot, system_prompt, temperature, max_tokens, use_streaming, model_info],
            outputs=[chatbot],
            queue=True
        )
        
        clear_btn.click(lambda: None, None, chatbot, queue=False)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
