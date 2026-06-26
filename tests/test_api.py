"""
Test Suite
==========
Run with:
    pytest tests/ -v

Tests cover:
  - Health endpoints
  - Generic prediction endpoint
  - Crop yield endpoint
  - Disease risk endpoint
  - Fraud detection endpoint
  - Invalid input rejection (422)
  - Missing model handling (404)
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app
from app.core.model_registry import model_registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def register_mock_models():
    """
    Register mock scikit-learn-compatible models for all tests.
    This avoids needing real .joblib files during CI.
    """
    def make_classifier(n_classes=2):
        m = MagicMock()
        m.predict.return_value = np.array([0])
        m.predict_proba.return_value = np.array([[0.15, 0.85]])
        m.classes_ = np.array([0, 1])
        return m

    def make_regressor():
        m = MagicMock()
        m.predict.return_value = np.array([1450.7])
        return m

    model_registry.register("generic_test", make_classifier(), {"description": "Test model"})
    model_registry.register("crop_yield", make_regressor(), {"description": "Crop yield mock"})
    model_registry.register("disease_risk", make_classifier(), {"description": "Disease risk mock"})
    model_registry.register("fraud_detection", make_classifier(), {"description": "Fraud detection mock"})


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "models_loaded" in body

    def test_ping(self, client):
        r = client.get("/ping")
        assert r.status_code == 200
        assert r.json() == {"ping": "pong"}

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "docs" in r.json()


# ---------------------------------------------------------------------------
# Generic prediction
# ---------------------------------------------------------------------------

class TestGenericPredict:
    def test_valid_prediction(self, client):
        r = client.post("/predict/", json={
            "model_name": "generic_test",
            "features": {"f1": 1.0, "f2": 2.0, "f3": 0},
            "return_probabilities": False,
        })
        assert r.status_code == 200
        body = r.json()
        assert "prediction" in body
        assert "processing_time_ms" in body

    def test_with_probabilities(self, client):
        r = client.post("/predict/", json={
            "model_name": "generic_test",
            "features": {"f1": 1.0},
            "return_probabilities": True,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["probabilities"] is not None
        assert "0" in body["probabilities"] or "1" in body["probabilities"]

    def test_unknown_model_returns_404(self, client):
        r = client.post("/predict/", json={
            "model_name": "does_not_exist",
            "features": {"f1": 1.0},
        })
        assert r.status_code == 404

    def test_missing_model_name_returns_422(self, client):
        r = client.post("/predict/", json={"features": {"f1": 1.0}})
        assert r.status_code == 422

    def test_missing_features_returns_422(self, client):
        r = client.post("/predict/", json={"model_name": "generic_test"})
        assert r.status_code == 422

    def test_response_has_process_time_header(self, client):
        r = client.post("/predict/", json={
            "model_name": "generic_test",
            "features": {"f1": 1.0},
        })
        assert "x-process-time-ms" in r.headers


# ---------------------------------------------------------------------------
# Crop yield
# ---------------------------------------------------------------------------

class TestCropYield:
    VALID_PAYLOAD = {
        "region": "Mbale",
        "crop_type": "maize",
        "rainfall_mm": 145.0,
        "temperature_c": 22.5,
        "soil_type": "loam",
        "planting_month": 3,
        "farm_size_acres": 2.5,
    }

    def test_valid_request(self, client):
        r = client.post("/predict/crop-yield", json=self.VALID_PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert "predicted_yield_kg_per_acre" in body
        assert "risk_level" in body
        assert "recommendation" in body
        assert body["region"] == "Mbale"

    def test_risk_levels_valid(self, client):
        r = client.post("/predict/crop-yield", json=self.VALID_PAYLOAD)
        body = r.json()
        assert body["risk_level"] in ["low", "medium", "high"]

    def test_temperature_out_of_range(self, client):
        payload = {**self.VALID_PAYLOAD, "temperature_c": 99.0}
        r = client.post("/predict/crop-yield", json=payload)
        assert r.status_code == 422

    def test_negative_rainfall_rejected(self, client):
        payload = {**self.VALID_PAYLOAD, "rainfall_mm": -10.0}
        r = client.post("/predict/crop-yield", json=payload)
        assert r.status_code == 422

    def test_invalid_month_rejected(self, client):
        payload = {**self.VALID_PAYLOAD, "planting_month": 13}
        r = client.post("/predict/crop-yield", json=payload)
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Disease risk
# ---------------------------------------------------------------------------

class TestDiseaseRisk:
    VALID_PAYLOAD = {
        "district": "Gulu",
        "disease": "malaria",
        "rainfall_mm_last_30d": 210.0,
        "clinic_cases_last_week": 87,
        "population_density": 145.0,
        "has_clean_water_access": False,
        "temperature_c": 28.5,
    }

    def test_valid_request(self, client):
        r = client.post("/predict/disease-risk", json=self.VALID_PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert "risk_score" in body
        assert "risk_level" in body
        assert "alert_recommended" in body
        assert "action" in body

    def test_risk_score_in_range(self, client):
        r = client.post("/predict/disease-risk", json=self.VALID_PAYLOAD)
        body = r.json()
        assert 0.0 <= body["risk_score"] <= 1.0

    def test_confidence_in_range(self, client):
        r = client.post("/predict/disease-risk", json=self.VALID_PAYLOAD)
        body = r.json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_alert_recommended_is_bool(self, client):
        r = client.post("/predict/disease-risk", json=self.VALID_PAYLOAD)
        assert isinstance(r.json()["alert_recommended"], bool)

    def test_missing_district_rejected(self, client):
        payload = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "district"}
        r = client.post("/predict/disease-risk", json=payload)
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Fraud detection
# ---------------------------------------------------------------------------

class TestFraudDetection:
    VALID_PAYLOAD = {
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

    def test_valid_request(self, client):
        r = client.post("/predict/fraud", json=self.VALID_PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert "is_fraud" in body
        assert "fraud_probability" in body
        assert "recommended_action" in body

    def test_is_fraud_is_bool(self, client):
        r = client.post("/predict/fraud", json=self.VALID_PAYLOAD)
        assert isinstance(r.json()["is_fraud"], bool)

    def test_recommended_action_valid_values(self, client):
        r = client.post("/predict/fraud", json=self.VALID_PAYLOAD)
        assert r.json()["recommended_action"] in ["allow", "review", "block"]

    def test_fraud_probability_in_range(self, client):
        r = client.post("/predict/fraud", json=self.VALID_PAYLOAD)
        prob = r.json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_tiny_amount_rejected(self, client):
        payload = {**self.VALID_PAYLOAD, "transaction_amount_ugx": 0}
        r = client.post("/predict/fraud", json=payload)
        assert r.status_code == 422

    def test_invalid_hour_rejected(self, client):
        payload = {**self.VALID_PAYLOAD, "transaction_hour": 25}
        r = client.post("/predict/fraud", json=payload)
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Models listing
# ---------------------------------------------------------------------------

class TestModelsEndpoint:
    def test_list_models(self, client):
        r = client.get("/models/")
        assert r.status_code == 200
        body = r.json()
        assert "count" in body
        assert "models" in body
        assert body["count"] >= 1

    def test_get_specific_model(self, client):
        r = client.get("/models/crop_yield")
        assert r.status_code == 200
        assert r.json()["name"] == "crop_yield"

    def test_nonexistent_model_404(self, client):
        r = client.get("/models/does_not_exist")
        assert r.status_code == 404
