from enum import Enum
from typing import Optional
from functools import lru_cache
import os

from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"


class Settings(BaseSettings):
    # LLM Provider configuration
    llm_provider: LLMProvider = LLMProvider.GEMINI
    
    # Groq settings
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.1-70b-instant"
    
    # Google Gemini settings
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-pro"
    
    # Application settings
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings from environment variables with caching
    """
    # Override provider from environment if set
    provider_env = os.getenv("LLM_PROVIDER", "").lower()
    settings = Settings()
    
    if provider_env:
        try:
            settings.llm_provider = LLMProvider(provider_env)
        except ValueError:
            valid_providers = ", ".join([p.value for p in LLMProvider])
            print(f"Warning: Invalid LLM_PROVIDER '{provider_env}'. Using default. Valid options: {valid_providers}")
    
    return settings
