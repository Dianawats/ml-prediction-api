"""
Schemas
=======
Pydantic v2 models for request validation and response serialisation.
Every field is validated before it touches an ML model.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional, Union
from enum import Enum


# ---------------------------------------------------------------------------
# Generic prediction schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Generic prediction — send any feature dict, specify which model to use."""
    model_name: str = Field(..., description="Name of the model to query (matches filename in models/)")
    features: Dict[str, Union[float, int, str, bool]] = Field(
        ..., description="Feature key-value pairs that the model expects"
    )
    return_probabilities: bool = Field(
        False, description="If true, return class probabilities alongside the prediction"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "model_name": "crop_yield",
            "features": {"rainfall_mm": 120.5, "temperature_c": 24.0, "region": "Mbale"},
            "return_probabilities": False,
        }
    }}


class PredictResponse(BaseModel):
    model_name: str
    prediction: Any
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    probabilities: Optional[Dict[str, float]] = None
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Uganda-specific domain schemas
# ---------------------------------------------------------------------------

class CropYieldRequest(BaseModel):
    """Crop yield prediction for Ugandan agriculture."""
    region: str = Field(..., description="Ugandan district e.g. Mbale, Mbarara, Gulu")
    crop_type: str = Field(..., description="e.g. maize, coffee, beans, cassava")
    rainfall_mm: float = Field(..., ge=0, le=3000, description="Seasonal rainfall in mm")
    temperature_c: float = Field(..., ge=10.0, le=45.0, description="Average temperature °C")
    soil_type: str = Field(..., description="e.g. clay, loam, sandy")
    planting_month: int = Field(..., ge=1, le=12, description="Month planting begins (1-12)")
    farm_size_acres: float = Field(..., ge=0.1, le=10000.0)

    model_config = {"json_schema_extra": {
        "example": {
            "region": "Mbale",
            "crop_type": "maize",
            "rainfall_mm": 145.0,
            "temperature_c": 22.5,
            "soil_type": "loam",
            "planting_month": 3,
            "farm_size_acres": 2.5,
        }
    }}


class CropYieldResponse(BaseModel):
    region: str
    crop_type: str
    predicted_yield_kg_per_acre: float
    risk_level: str  # low / medium / high
    confidence: float
    recommendation: str
    processing_time_ms: float


class DiseaseRiskRequest(BaseModel):
    """Disease outbreak risk prediction for Uganda districts."""
    district: str = Field(..., description="Ugandan district name")
    disease: str = Field(..., description="e.g. malaria, cholera, typhoid")
    rainfall_mm_last_30d: float = Field(..., ge=0, le=1000)
    clinic_cases_last_week: int = Field(..., ge=0)
    population_density: float = Field(..., ge=0, description="People per sq km")
    has_clean_water_access: bool = Field(..., description="District has treated water access")
    temperature_c: float = Field(..., ge=10.0, le=45.0)

    model_config = {"json_schema_extra": {
        "example": {
            "district": "Gulu",
            "disease": "malaria",
            "rainfall_mm_last_30d": 210.0,
            "clinic_cases_last_week": 87,
            "population_density": 145.0,
            "has_clean_water_access": False,
            "temperature_c": 28.5,
        }
    }}


class DiseaseRiskResponse(BaseModel):
    district: str
    disease: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str  # low / medium / high / critical
    confidence: float
    alert_recommended: bool
    action: str
    processing_time_ms: float


class FraudDetectionRequest(BaseModel):
    """Mobile money fraud detection (MTN MoMo / Airtel Money style)."""
    transaction_amount_ugx: float = Field(..., ge=100, description="Amount in Uganda Shillings")
    sender_account_age_days: int = Field(..., ge=0)
    receiver_account_age_days: int = Field(..., ge=0)
    transactions_last_24h: int = Field(..., ge=0)
    is_new_recipient: bool
    transaction_hour: int = Field(..., ge=0, le=23)
    sender_district: str
    receiver_district: str
    device_changed_recently: bool

    model_config = {"json_schema_extra": {
        "example": {
            "transaction_amount_ugx": 500000,
            "sender_account_age_days": 365,
            "receiver_account_age_days": 3,
            "transactions_last_24h": 12,
            "is_new_recipient": True,
            "transaction_hour": 2,
            "sender_district": "Kampala",
            "receiver_district": "Wakiso",
            "device_changed_recently": True,
        }
    }}


class FraudDetectionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str  # safe / low / medium / high
    recommended_action: str  # allow / review / block
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    name: str
    file: Optional[str] = None
    description: Optional[str] = None
    features: Optional[List[str]] = None
    accuracy: Optional[float] = None
    trained_on: Optional[str] = None


class ModelsListResponse(BaseModel):
    count: int
    models: List[ModelInfo]
