import logging
import logging.config
import sys
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.config import Settings, get_settings
from app.routers import chat

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": sys.stdout,
        },
    },
    "loggers": {
        "": {
            "handlers": ["console"],
            "level": "INFO",
        },
        "uvicorn.error": {
            "level": "INFO",
        },
        "uvicorn.access": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "app": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

def create_app(settings: Settings) -> FastAPI:
    log_level = "DEBUG" if settings.debug else "INFO"
    LOGGING_CONFIG["loggers"]["""]["level"] = log_level
    LOGGING_CONFIG["loggers"]["app"]["level"] = log_level
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    app_instance = FastAPI(
        title="Conversational AI API",
        description="API for interacting with LLMs (e.g., Gemini, Groq Llama)",
        version="1.0.0",
    )
    allowed_origins = ["*"]
    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app_instance.include_router(chat.router)
    @app_instance.get("/", tags=["Health Check"])
    async def root(current_settings: Settings = Depends(get_settings)):
        return {
            "status": "ok",
            "llm_provider": current_settings.llm_provider.value,
            "debug_mode": current_settings.debug,
            "message": f"Conversational AI API running with {current_settings.llm_provider.value}",
        }
    @app_instance.on_event("startup")
    async def startup_event():
        get_settings()
    @app_instance.on_event("shutdown")
    async def shutdown_event():
        pass
    return app_instance

try:
    settings = get_settings()
    app = create_app(settings)
except Exception:
    logging.basicConfig(level=logging.INFO)
    sys.exit(1)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=logging.getLevelName(LOGGING_CONFIG["loggers"]["""]["level"]).lower(),
    )
