import logging
import requests
from typing import List, Dict, Optional, Any
import json

logger = logging.getLogger(__name__)

class ApiClient:
    """Client for interacting with the backend API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def send_chat_request(
        self,
        message: str,
        history: List[Dict[str, str]],
        session_id: str,
        provider: Optional[str] = None,
        use_langchain: bool = False
    ) -> requests.Response:
        """
        Send a chat request to the backend API.
        
        Args:
            message: The user's message
            history: Chat history
            session_id: Session ID for WebSocket connection
            provider: LLM provider to use
            use_langchain: Whether to use Langchain
            
        Returns:
            Response from the API
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "message": message,
            "history": history,
            "session_id": session_id,
            "provider": provider
        }
        
        if use_langchain:
            payload["use_langchain"] = use_langchain
            
        logger.debug(f"Sending chat request to {url}: {json.dumps(payload)}")
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            logger.debug(f"Received response: {response.status_code} - {response.text}")
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending chat request: {str(e)}")
            raise

    def get_providers(self) -> Dict[str, Any]:
        """
        Get available LLM providers from the backend API.
        
        Returns:
            Dictionary with providers information
        """
        url = f"{self.base_url}/api/providers"
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error getting providers: {response.status_code} - {response.text}")
                return {"providers": [], "default": None}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting providers: {str(e)}")
            return {"providers": [], "default": None}

# Create a global instance
api_client = ApiClient()

# Export functions for easier imports
def send_chat_request(
    message: str,
    history: List[Dict[str, str]],
    session_id: str,
    provider: Optional[str] = None,
    use_langchain: bool = False
) -> requests.Response:
    """Send a chat request to the backend API."""
    return api_client.send_chat_request(
        message=message,
        history=history,
        session_id=session_id,
        provider=provider,
        use_langchain=use_langchain
    )

def get_providers() -> Dict[str, Any]:
    """Get available LLM providers from the backend API."""
    return api_client.get_providers()
