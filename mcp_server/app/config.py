import os
from typing import Literal, Optional, List
from pydantic_settings import BaseSettings  # Updated import

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    GROQ_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    
    # Application settings
    DEFAULT_LLM_PROVIDER: Literal["groq", "gemini"] = "groq"
    SUPPORTED_LLM_PROVIDERS: List[str] = ["groq", "gemini"]
    USE_LANGCHAIN: bool = False
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    
    # Security
    CORS_ORIGINS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()