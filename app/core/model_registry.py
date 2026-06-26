"""
Model Registry
==============
Loads all .joblib model files from the models/ directory at startup.
Provides a central store for inference across all prediction endpoints.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import joblib

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central store for all trained scikit-learn models.
    Models are loaded once at startup and kept in memory for fast inference.
    """

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, dict] = {}

    def load_all(self):
        """Scan MODELS_DIR and load every .joblib file found."""
        models_path = Path(settings.MODELS_DIR)
        if not models_path.exists():
            logger.warning(f"Models directory '{models_path}' not found. No models loaded.")
            return

        for filepath in sorted(models_path.glob("*.joblib")):
            model_name = filepath.stem
            try:
                artifact = joblib.load(filepath)

                # Support both raw model objects and packaged dicts
                if isinstance(artifact, dict) and "model" in artifact:
                    self._models[model_name] = artifact["model"]
                    self._metadata[model_name] = {
                        k: v for k, v in artifact.items() if k != "model"
                    }
                else:
                    self._models[model_name] = artifact
                    self._metadata[model_name] = {}

                self._metadata[model_name]["file"] = str(filepath)
                logger.info(f"Loaded model: {model_name}")

            except Exception as e:
                logger.error(f"Failed to load model '{filepath}': {e}")

    def get(self, name: str) -> Optional[Any]:
        return self._models.get(name)

    def get_metadata(self, name: str) -> dict:
        return self._metadata.get(name, {})

    def available_models(self) -> Dict[str, dict]:
        return {
            name: self._metadata.get(name, {})
            for name in self._models
        }

    def is_loaded(self, name: str) -> bool:
        return name in self._models

    def register(self, name: str, model: Any, metadata: dict = None):
        """Programmatically register a model (used in tests)."""
        self._models[name] = model
        self._metadata[name] = metadata or {}
        logger.info(f"Registered model: {name}")


# Global singleton used across all routers
model_registry = ModelRegistry()
