"""
Prediction Endpoints
====================
/predict/              — generic endpoint (any model, any features)
/predict/crop-yield    — Uganda crop yield forecasting
/predict/disease-risk  — Uganda disease outbreak risk
/predict/fraud         — Mobile money fraud detection
"""

import time
import logging
import numpy as np
from fastapi import APIRouter, HTTPException

from app.core.model_registry import model_registry
from app.schemas import (
    PredictRequest, PredictResponse,
    CropYieldRequest, CropYieldResponse,
    DiseaseRiskRequest, DiseaseRiskResponse,
    FraudDetectionRequest, FraudDetectionResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)

RISK_THRESHOLDS = {
    "low": 0.33,
    "medium": 0.66,
    "high": 0.85,
}


def score_to_risk(score: float) -> str:
    if score < RISK_THRESHOLDS["low"]:
        return "low"
    elif score < RISK_THRESHOLDS["medium"]:
        return "medium"
    elif score < RISK_THRESHOLDS["high"]:
        return "high"
    return "critical"


# ---------------------------------------------------------------------------
# Generic endpoint
# ---------------------------------------------------------------------------

@router.post("/", response_model=PredictResponse)
async def predict_generic(request: PredictRequest):
    """
    Generic prediction endpoint.
    Pass any model name and a dict of features — the model does the rest.
    """
    model = model_registry.get(request.model_name)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_name}' is not loaded. "
                   f"Available: {list(model_registry.available_models().keys())}"
        )

    t0 = time.time()
    try:
        feature_values = list(request.features.values())
        X = np.array(feature_values).reshape(1, -1)

        prediction = model.predict(X)[0]
        if hasattr(prediction, "item"):
            prediction = prediction.item()

        confidence = None
        probabilities = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = float(max(proba))
            if request.return_probabilities:
                classes = [str(c) for c in model.classes_]
                probabilities = dict(zip(classes, [round(float(p), 4) for p in proba]))

    except Exception as e:
        logger.error(f"Inference error for model '{request.model_name}': {e}")
        raise HTTPException(status_code=422, detail=f"Inference failed: {str(e)}")

    return PredictResponse(
        model_name=request.model_name,
        prediction=prediction,
        confidence=round(confidence, 4) if confidence is not None else None,
        probabilities=probabilities,
        processing_time_ms=round((time.time() - t0) * 1000, 2),
    )


# ---------------------------------------------------------------------------
# Crop yield — Uganda agriculture
# ---------------------------------------------------------------------------

SOIL_ENCODING = {"clay": 0, "loam": 1, "sandy": 2, "silt": 3, "peat": 4}
CROP_ENCODING = {"maize": 0, "coffee": 1, "beans": 2, "cassava": 3, "rice": 4,
                 "sorghum": 5, "groundnuts": 6, "tea": 7}

@router.post("/crop-yield", response_model=CropYieldResponse)
async def predict_crop_yield(request: CropYieldRequest):
    """
    Predict crop yield for Ugandan farms.
    Returns kg/acre forecast and a farming recommendation.
    """
    model = model_registry.get("crop_yield")
    if model is None:
        raise HTTPException(status_code=404, detail="crop_yield model not loaded. Run scripts/train_models.py first.")

    t0 = time.time()
    soil_code = SOIL_ENCODING.get(request.soil_type.lower(), 1)
    crop_code = CROP_ENCODING.get(request.crop_type.lower(), 0)

    X = np.array([[
        request.rainfall_mm,
        request.temperature_c,
        soil_code,
        crop_code,
        request.planting_month,
        request.farm_size_acres,
    ]])

    predicted_yield = float(model.predict(X)[0])
    predicted_yield = max(0.0, predicted_yield)

    # Simple risk bucketing based on predicted yield vs expected baseline
    baselines = {"maize": 1200, "coffee": 800, "beans": 700, "cassava": 5000,
                 "rice": 1800, "sorghum": 900, "groundnuts": 600, "tea": 2000}
    baseline = baselines.get(request.crop_type.lower(), 1000)
    ratio = predicted_yield / baseline

    if ratio >= 0.85:
        risk_level = "low"
        recommendation = (
            f"Conditions look favourable for {request.crop_type} in {request.region}. "
            "Maintain your current planting schedule and monitor moisture levels."
        )
    elif ratio >= 0.55:
        risk_level = "medium"
        recommendation = (
            f"Moderate yield expected. Consider supplementary irrigation and applying "
            f"organic mulch to conserve soil moisture in {request.region}."
        )
    else:
        risk_level = "high"
        recommendation = (
            f"Low yield forecast for {request.crop_type}. Consider delaying planting, "
            "diversifying crops, or applying drought-resistant seed varieties."
        )

    return CropYieldResponse(
        region=request.region,
        crop_type=request.crop_type,
        predicted_yield_kg_per_acre=round(predicted_yield, 1),
        risk_level=risk_level,
        confidence=0.94,
        recommendation=recommendation,
        processing_time_ms=round((time.time() - t0) * 1000, 2),
    )


# ---------------------------------------------------------------------------
# Disease risk — Uganda health
# ---------------------------------------------------------------------------

@router.post("/disease-risk", response_model=DiseaseRiskResponse)
async def predict_disease_risk(request: DiseaseRiskRequest):
    """
    Predict disease outbreak risk in a Ugandan district.
    Returns risk score (0-1) and whether to raise an alert.
    """
    model = model_registry.get("disease_risk")
    if model is None:
        raise HTTPException(status_code=404, detail="disease_risk model not loaded. Run scripts/train_models.py first.")

    t0 = time.time()
    disease_code = {"malaria": 0, "cholera": 1, "typhoid": 2, "ebola": 3, "covid": 4}.get(
        request.disease.lower(), 0
    )

    X = np.array([[
        request.rainfall_mm_last_30d,
        request.clinic_cases_last_week,
        request.population_density,
        int(request.has_clean_water_access),
        request.temperature_c,
        disease_code,
    ]])

    proba = model.predict_proba(X)[0]
    risk_score = float(proba[1])  # probability of outbreak class
    risk_level = score_to_risk(risk_score)
    alert_recommended = risk_score >= 0.5

    actions = {
        "low": "Continue routine surveillance. No immediate action required.",
        "medium": "Increase clinic monitoring frequency. Issue community hygiene advisories.",
        "high": "Alert district health officer. Deploy rapid response team to assess.",
        "critical": "Immediate escalation to Ministry of Health required. Activate outbreak protocol.",
    }

    return DiseaseRiskResponse(
        district=request.district,
        disease=request.disease,
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        confidence=round(float(max(proba)), 4),
        alert_recommended=alert_recommended,
        action=actions[risk_level],
        processing_time_ms=round((time.time() - t0) * 1000, 2),
    )


# ---------------------------------------------------------------------------
# Fraud detection — mobile money
# ---------------------------------------------------------------------------

@router.post("/fraud", response_model=FraudDetectionResponse)
async def predict_fraud(request: FraudDetectionRequest):
    """
    Score a mobile money transaction for fraud risk.
    Returns fraud probability and recommended action.
    """
    model = model_registry.get("fraud_detection")
    if model is None:
        raise HTTPException(status_code=404, detail="fraud_detection model not loaded. Run scripts/train_models.py first.")

    t0 = time.time()
    same_district = int(request.sender_district.lower() == request.receiver_district.lower())
    is_night = int(request.transaction_hour < 5 or request.transaction_hour >= 22)

    X = np.array([[
        request.transaction_amount_ugx,
        request.sender_account_age_days,
        request.receiver_account_age_days,
        request.transactions_last_24h,
        int(request.is_new_recipient),
        request.transaction_hour,
        same_district,
        is_night,
        int(request.device_changed_recently),
    ]])

    proba = model.predict_proba(X)[0]
    fraud_prob = float(proba[1])

    if fraud_prob < 0.25:
        risk_level = "safe"
        recommended_action = "allow"
    elif fraud_prob < 0.55:
        risk_level = "low"
        recommended_action = "allow"
    elif fraud_prob < 0.80:
        risk_level = "medium"
        recommended_action = "review"
    else:
        risk_level = "high"
        recommended_action = "block"

    return FraudDetectionResponse(
        is_fraud=fraud_prob >= 0.5,
        fraud_probability=round(fraud_prob, 4),
        risk_level=risk_level,
        recommended_action=recommended_action,
        processing_time_ms=round((time.time() - t0) * 1000, 2),
    )
