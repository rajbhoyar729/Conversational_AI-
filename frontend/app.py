import gradio as gr
import requests

BACKEND_URL = "http://127.0.0.1:8000"  # updated backend URL

def send_message(user_input, chat_history):
    try:
        response = requests.post(BACKEND_URL + "/chat", json={"prompt": user_input})
        response_data = response.json()
        bot_response = response_data.get("response", "Error: No response from server.")
        chat_history.append((user_input, bot_response))
        return "", chat_history
    except Exception as e:
        chat_history.append((user_input, f"Error: {str(e)}"))
        return "", chat_history

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Conversational AI Chatbot")
        chatbot = gr.Chatbot()
        user_input = gr.Textbox(placeholder="Type your message here...")
        submit_button = gr.Button("Send")

        chat_history = gr.State([])

        submit_button.click(
            send_message,
            inputs=[user_input, chat_history],
            outputs=[user_input, chatbot]
        )

    demo.launch()

if __name__ == "__main__":
    main()
