"""
Application Configuration Module for Conversational AI Backend

This module defines the Settings class for managing environment-based configuration,
provider-specific API keys, and validation rules.
"""

import logging
import os
from pathlib import Path
from enum import Enum
from functools import lru_cache
from typing import ClassVar, List, Optional
from pydantic import BaseModel, Field, SecretStr, model_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Environment variable prefix
ENV_PREFIX = "APP_"
ENV_DEBUG = f"{ENV_PREFIX}DEBUG"
ENV_LLM_PROVIDER = f"{ENV_PREFIX}LLM_PROVIDER"

# Provider-specific environment variables
ENV_GROQ_API_KEY = f"{ENV_PREFIX}GROQ_API_KEY"
ENV_GROQ_MODEL = f"{ENV_PREFIX}GROQ_MODEL"
ENV_GEMINI_API_KEY = f"{ENV_PREFIX}GEMINI_API_KEY"
ENV_GEMINI_MODEL = f"{ENV_PREFIX}GEMINI_MODEL"

# ======================
# LLMProvider Enum
# ======================

class LLMProvider(str, Enum):
    """
    Supported LLM providers.

    Extend this class to support additional providers like Anthropic or OpenAI.
    Ensure corresponding adapters are implemented in `llm_client.py`.
    """
    GEMINI = "gemini"
    GROQ = "groq"

    @classmethod
    def list_values(cls) -> List[str]:
        return [item.value for item in cls]

# ======================
# Helper Function to Load .env File
# ======================

def load_env_file(env_path: str) -> dict:
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    return env_vars

# ======================
# Settings Model
# ======================

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.

    Fields:
    - debug: Enables debug mode in FastAPI
    - llm_provider: Active LLM provider (e.g., 'gemini', 'groq')
    - groq_api_key: API key for Groq
    - groq_model: Model name for Groq (e.g., 'llama-3.1-70b-instant')
    - gemini_api_key: API key for Google Gemini
    - gemini_model: Model name for Gemini (e.g., 'gemini-1.5-pro')
    """
    APP_GEMINI_API_KEY: str = ""
    APP_GROQ_API_KEY: str = ""
    llm_provider: str = ""
    groq_api_key: str = ""
    groq_model: str = ""
    gemini_api_key: str = ""
    gemini_model: str = ""
    debug: bool = False

    model_config = ConfigDict(extra="allow")  # Allow extra inputs from the env file

# ======================
# Get Settings Function
# ======================

@lru_cache()
def get_settings() -> Settings:
    """
    Loads and caches application settings.

    Returns:
        Settings: The validated application configuration.

    Raises:
        ValidationError: If the configuration fails validation.
        RuntimeError: If an unexpected error occurs during loading.
    """
    logger.info("Attempting to load application settings...")
    try:
        base_dir = Path(__file__).parent.parent  # Adjust if .env is at the backend root
        env_path = os.path.join(base_dir, ".env")
        env_vars = load_env_file(env_path)
        settings = Settings(**env_vars)
        logger.info(f"Settings loaded successfully. Provider: '{settings.llm_provider}', Debug: {settings.debug}")
        return settings
    except Exception as e:
        logger.critical(f"Application settings failed to load: {e}", exc_info=True)
        raise