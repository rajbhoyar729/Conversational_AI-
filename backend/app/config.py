import logging
import os
from enum import Enum
from functools import lru_cache
from typing import Optional, ClassVar, List

from pydantic import Field, SecretStr, ValidationError, HttpUrl, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

ENV_PREFIX = "APP_"
ENV_DEBUG = f"{ENV_PREFIX}DEBUG"
ENV_LLM_PROVIDER = f"{ENV_PREFIX}LLM_PROVIDER"
ENV_GROQ_API_KEY = f"{ENV_PREFIX}GROQ_API_KEY"
ENV_GROQ_MODEL = f"{ENV_PREFIX}GROQ_MODEL"
ENV_GEMINI_API_KEY = f"{ENV_PREFIX}GEMINI_API_KEY"
ENV_GEMINI_MODEL = f"{ENV_PREFIX}GEMINI_MODEL"

class LLMProvider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"

    @classmethod
    def list_values(cls) -> List[str]:
        return [item.value for item in cls]

class Settings(BaseSettings):
    debug: bool = Field(
        default=False,
        validation_alias=ENV_DEBUG
    )
    llm_provider: LLMProvider = Field(
        default=LLMProvider.GEMINI,
        validation_alias=ENV_LLM_PROVIDER
    )
    groq_api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias=ENV_GROQ_API_KEY
    )
    groq_model: str = Field(
        default="llama-3.1-70b-instant",
        min_length=1,
        validation_alias=ENV_GROQ_MODEL
    )
    gemini_api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias=ENV_GEMINI_API_KEY
    )
    gemini_model: str = Field(
        default="gemini-1.5-pro",
        min_length=1,
        validation_alias=ENV_GEMINI_MODEL
    )

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra='ignore',
    )

    @field_validator('groq_api_key', 'gemini_api_key')
    @classmethod
    def check_api_key_present_for_provider(cls, v: Optional[SecretStr], info: ValidationInfo) -> Optional[SecretStr]:
        if not info.data:
            return v

        selected_provider = info.data.get('llm_provider')
        field_name = info.field_name

        if field_name == 'groq_api_key' and selected_provider == LLMProvider.GROQ and v is None:
            raise ValueError(f"'{ENV_GROQ_API_KEY}' must be set when LLM provider is '{LLMProvider.GROQ.value}'.")

        if field_name == 'gemini_api_key' and selected_provider == LLMProvider.GEMINI and v is None:
            raise ValueError(f"'{ENV_GEMINI_API_KEY}' must be set when LLM provider is '{LLMProvider.GEMINI.value}'.")

        return v

@lru_cache()
def get_settings() -> Settings:
    logger.info("Attempting to load application settings...")
    try:
        settings = Settings()
        logger.info(f"Settings loaded successfully. Provider: '{settings.llm_provider.value}', Debug: {settings.debug}")
        return settings
    except ValidationError as e:
        logger.critical("FATAL: Settings validation failed. Please check your .env file and environment variables.")
        logger.critical(f"Validation Errors:\n{e}")
        raise
    except Exception as e:
        logger.critical(f"FATAL: An unexpected error occurred while loading settings: {e}", exc_info=True)
        raise
