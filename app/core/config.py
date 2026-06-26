"""
Configuration — reads from environment variables with sensible defaults.
Set these in your .env file or Docker environment.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "ML Prediction API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "A REST API serving machine learning forecasting models. "
        "Built for Uganda, Africa, and global deployment. "
        "Handles crop yield, disease outbreak, fraud detection, and more."
    )

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 2

    # CORS — in production, replace with your actual frontend domain
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Model storage — path where .joblib model files are stored
    MODELS_DIR: str = "models"

    # API key for write operations (optional; set to "" to disable auth)
    API_KEY: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
