from fastapi import APIRouter
from app.core.model_registry import model_registry
from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """System health — liveness and model status."""
    loaded = model_registry.available_models()
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "models_loaded": len(loaded),
        "models": list(loaded.keys()),
    }


@router.get("/ping")
async def ping():
    return {"ping": "pong"}
