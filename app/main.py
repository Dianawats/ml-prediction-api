"""
ML Prediction API
=================
A production-ready REST API serving ML forecasting models.
Designed for Uganda, Africa, and global deployment.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

from app.routers import predictions, health, models as model_router
from app.core.config import settings
from app.core.model_registry import model_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins in dev; lock down in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(round((time.time() - start) * 1000, 2))
    return response


@app.on_event("startup")
async def startup_event():
    logger.info("Starting ML Prediction API...")
    model_registry.load_all()
    logger.info(f"Loaded models: {list(model_registry.available_models().keys())}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down ML Prediction API.")


# Routers
app.include_router(health.router, tags=["Health"])
app.include_router(predictions.router, prefix="/predict", tags=["Predictions"])
app.include_router(model_router.router, prefix="/models", tags=["Models"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
