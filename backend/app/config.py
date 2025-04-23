import logging
import os
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

ENV_LLM_PROVIDER = "LLM_PROVIDER"
ENV_GROQ_API_KEY = "GROQ_API_KEY"
ENV_GROQ_MODEL = "GROQ_MODEL"
ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
ENV_GEMINI_MODEL = "GEMINI_MODEL"
ENV_DEBUG = "DEBUG"

class LLMProvider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"

class Settings(BaseSettings):
    debug: bool = Field(default=False, validation_alias=ENV_DEBUG)
    llm_provider: LLMProvider = Field(
        default=LLMProvider.GEMINI, validation_alias=ENV_LLM_PROVIDER
    )
    groq_api_key: Optional[SecretStr] = Field(
        default=None, validation_alias=ENV_GROQ_API_KEY
    )
    groq_model: str = Field(default="llama-3.1-70b-instant", validation_alias=ENV_GROQ_MODEL)
    gemini_api_key: Optional[SecretStr] = Field(
        default=None, validation_alias=ENV_GEMINI_API_KEY
    )
    gemini_model: str = Field(default="gemini-1.5-pro", validation_alias=ENV_GEMINI_MODEL)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra='ignore'
    )

@lru_cache()
def get_settings() -> Settings:
    try:
        settings = Settings()
        logger.debug("Settings loaded successfully.")
        return settings
    except ValidationError as e:
        logger.error(f"Failed to load settings due to validation errors: {e}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading settings: {e}", exc_info=True)
        raise
