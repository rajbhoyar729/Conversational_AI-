"""API routes for the MCP server."""

from fastapi import APIRouter

router = APIRouter()

# Import and include sub-routers
from .chat import router as chat_router
from .websocket import router as websocket_router

router.include_router(chat_router)
router.include_router(websocket_router)
