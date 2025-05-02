import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .utils.logging import setup_logging
from .api import router as api_router

# Setup logging
logger = setup_logging()

# Create FastAPI app
app = FastAPI(
    title="Conversational AI MCP",
    description="Model Control Plane for Conversational AI",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)

@app.get("/", tags=["status"])
async def root():
    """Root endpoint to check if the API is running."""
    return {
        "status": "ok",
        "message": "Conversational AI Model Control Plane",
        "version": app.__version__
    }

@app.get("/health", tags=["status"])
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
