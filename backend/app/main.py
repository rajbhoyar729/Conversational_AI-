"""
FastAPI Application Entry Point for Conversational AI App
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Annotated

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import Settings, get_settings, LLMProvider
from app.models.schemas import ServerInfo
from app.routers import chat as chat_router

logger = logging.getLogger(__name__)

# Global state for runtime provider switching
_current_provider: LLMProvider = LLMProvider.GEMINI

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("🚀 Application startup: Initializing LLM client")
    try:
        settings = get_settings()
        logger.info(f"Loaded settings: Provider={settings.llm_provider}, Debug={settings.debug}")
        yield
    except Exception as e:
        logger.critical(f"🚨 Application failed to start: {e}", exc_info=True)
        raise RuntimeError(f"Startup failed: {e}") from e

app = FastAPI(lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=ServerInfo)
async def health_check(settings: Annotated[Settings, Depends(get_settings)]) -> ServerInfo:
    """Health check endpoint showing current provider and status"""
    return ServerInfo(
        status="ok",
        message="Service is operational",
        llm_provider=settings.llm_provider,
        debug_mode=settings.debug
    )

@app.post("/chat/provider", summary="Switch LLM Provider at Runtime")
async def switch_provider(
    request: dict, 
    settings: Annotated[Settings, Depends(get_settings)]
) -> JSONResponse:
    """
    Dynamically change the active LLM provider.  
    Example request body: {"provider": "groq"}
    """
    global _current_provider
    new_provider = request.get("provider", "").lower()
    
    if new_provider not in LLMProvider.list_values():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid provider. Supported: {LLMProvider.list_values()}"
        )
    
    try:
        # Validate provider-specific requirements
        if new_provider == LLMProvider.GROQ and settings.groq_api_key is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{ENV_GROQ_API_KEY} is required for Groq"
            )
        
        if new_provider == LLMProvider.GEMINI and settings.gemini_api_key is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{ENV_GEMINI_API_KEY} is required for Gemini"
            )
        
        # Update provider globally
        _current_provider = LLMProvider(new_provider)
        logger.info(f"🔄 LLM provider changed to: {_current_provider.value}")
        return JSONResponse(content={"status": "success", "provider": _current_provider.value})
    
    except Exception as e:
        logger.warning(f"🔄 Provider switch failed. Fallback to {_current_provider.value}: {e}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

# Include chat router
app.include_router(chat_router.router)

if __name__ == "__main__":
    import uvicorn
    run_settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",  # changed from "0.0.0.0"
        port=8000,
        reload=run_settings.debug,
        log_level="debug" if run_settings.debug else "info"
    )