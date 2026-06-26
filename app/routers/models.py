from fastapi import APIRouter, HTTPException
from app.core.model_registry import model_registry
from app.schemas import ModelsListResponse, ModelInfo

router = APIRouter()


@router.get("/", response_model=ModelsListResponse)
async def list_models():
    """List all loaded models and their metadata."""
    available = model_registry.available_models()
    models = []
    for name, meta in available.items():
        models.append(ModelInfo(
            name=name,
            file=meta.get("file"),
            description=meta.get("description"),
            features=meta.get("features"),
            accuracy=meta.get("accuracy"),
            trained_on=meta.get("trained_on"),
        ))
    return ModelsListResponse(count=len(models), models=models)


@router.get("/{model_name}", response_model=ModelInfo)
async def get_model(model_name: str):
    """Get metadata for a specific model."""
    if not model_registry.is_loaded(model_name):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    meta = model_registry.get_metadata(model_name)
    return ModelInfo(
        name=model_name,
        file=meta.get("file"),
        description=meta.get("description"),
        features=meta.get("features"),
        accuracy=meta.get("accuracy"),
        trained_on=meta.get("trained_on"),
    )
