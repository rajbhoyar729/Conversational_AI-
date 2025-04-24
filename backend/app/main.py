import logging
import logging.config
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.config import Settings, get_settings, LLMProvider
from app.models import ServerInfo
from app.routers import chat as chat_router

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s | %(name)s:%(lineno)d | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(asctime)s | %(client_addr)s | "%(request_line)s" %(status_code)s',
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
        },
    },
    "loggers": {
        "app": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": True,
        },
        "uvicorn.error": {
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["access"],
            "level": "INFO",
            "propagate": False,
        },
        "": {
            "handlers": ["default"],
            "level": "WARNING",
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Application lifespan: Startup sequence initiated...")
    try:
        settings = get_settings()
        log_level = "DEBUG" if settings.debug else "INFO"

        logging.getLogger("app").setLevel(log_level)
        logger.info(f"Logging level set to: {log_level}")
        logger.info(f"Selected LLM Provider: {settings.llm_provider.value}")
        logger.info(f"Debug mode: {settings.debug}")

        _ = get_llm_client(settings)
        logger.info("LLM Client factory check successful.")

        logger.info("Application startup completed successfully.")

    except (ValidationError, ValueError, ImportError, RuntimeError) as startup_err:
        logger.critical(f"FATAL: Application startup failed: {startup_err}", exc_info=True)
        sys.exit(f"Application startup failed: {startup_err}")
    except Exception as e:
        logger.critical(f"FATAL: Unexpected error during application startup: {e}", exc_info=True)
        sys.exit(f"Unexpected startup error: {e}")

    yield

    logger.info("Application lifespan: Shutdown sequence initiated...")
    logger.info("Application shutdown completed.")


app = FastAPI(
    title="Conversational AI Backend",
    description="API backend for the Conversational AI application, supporting multiple LLM providers.",
    version="1.0.0",
    lifespan=lifespan,
)

allowed_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Request validation failed: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation Error", "errors": exc.errors()},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException caught: Status={exc.status_code}, Detail='{exc.detail}'")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.critical(f"Unhandled exception during request processing: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal server error occurred."},
    )

app.include_router(chat_router.router)
logger.debug("Chat router included.")


@app.get(
    "/",
    response_model=ServerInfo,
    tags=["Health Check"],
    summary="Application Health Check",
    description="Provides basic status information about the running application instance, including the configured LLM provider.",
)
async def root(settings: Annotated[Settings, Depends(get_settings)]) -> ServerInfo:
    return ServerInfo(
        status="ok",
        message="Conversational AI Backend is running.",
        llm_provider=settings.llm_provider.value,
        debug_mode=settings.debug,
    )


if __name__ == "__main__":
    logger.info("Starting server directly via __main__ (for debugging/testing).")

    try:
        run_settings = get_settings()
        uvicorn_log_level = "debug" if run_settings.debug else "info"

        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=run_settings.debug,
            log_level=uvicorn_log_level,
        )
    except Exception as main_run_err:
        logger.critical(f"Failed to start Uvicorn server from __main__: {main_run_err}", exc_info=True)
        sys.exit("Failed to start server.")
