"""
Train Models
============
Generates realistic synthetic data for Uganda use cases and trains
three scikit-learn models:

  - crop_yield      → RandomForestRegressor
  - disease_risk    → GradientBoostingClassifier
  - fraud_detection → RandomForestClassifier

Run from the project root:
    python scripts/train_models.py

Trained models are saved to models/ as .joblib files.
In production, replace the synthetic data with real datasets from:
  - NARO (crop data)
  - Uganda Ministry of Health
  - MTN / Airtel transaction logs (anonymised)
"""

import os
import sys
import numpy as np
import joblib
from pathlib import Path
from datetime import date

# Ensure we can import from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)
TODAY = date.today().isoformat()

print("=" * 60)
print("  ML Prediction API — Model Training")
print("  Target: Uganda, Africa, Global")
print("=" * 60)


# ---------------------------------------------------------------------------
# 1. CROP YIELD MODEL
# ---------------------------------------------------------------------------
print("\n[1/3] Training crop yield model (RandomForestRegressor)...")

N = 8000
rainfall    = RNG.uniform(50, 400, N)       # mm per season
temperature = RNG.uniform(14, 35, N)        # °C
soil_type   = RNG.integers(0, 4, N)         # 0=clay 1=loam 2=sandy 3=silt
crop_type   = RNG.integers(0, 8, N)         # 8 crop types
plant_month = RNG.integers(1, 13, N)        # 1–12
farm_size   = RNG.uniform(0.5, 50, N)       # acres

# Simulate realistic yield relationships
base_yield = (
    800
    + rainfall * 3.2
    - np.abs(temperature - 24) * 20
    + (soil_type == 1) * 300          # loam is best
    - (soil_type == 2) * 150          # sandy is worst
    + (plant_month.isin([3, 4, 9, 10]) if hasattr(plant_month, 'isin') else
       np.isin(plant_month, [3, 4, 9, 10])) * 200   # good planting months
    + np.log1p(farm_size) * 80
    + RNG.normal(0, 120, N)           # noise
)
base_yield = np.clip(base_yield, 100, 6000)

X_yield = np.column_stack([rainfall, temperature, soil_type, crop_type, plant_month, farm_size])
y_yield = base_yield

X_train, X_test, y_train, y_test = train_test_split(X_yield, y_yield, test_size=0.2, random_state=42)

crop_model = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
crop_model.fit(X_train, y_train)

preds = crop_model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"  MAE: {mae:.1f} kg/acre")

joblib.dump({
    "model": crop_model,
    "description": "Crop yield forecasting for Ugandan agriculture",
    "features": ["rainfall_mm", "temperature_c", "soil_type", "crop_type", "planting_month", "farm_size_acres"],
    "accuracy": round(1 - mae / y_test.mean(), 4),
    "trained_on": TODAY,
}, MODELS_DIR / "crop_yield.joblib")
print("  Saved: models/crop_yield.joblib")


# ---------------------------------------------------------------------------
# 2. DISEASE RISK MODEL
# ---------------------------------------------------------------------------
print("\n[2/3] Training disease risk model (GradientBoostingClassifier)...")

N = 6000
rainfall_30d    = RNG.uniform(0, 500, N)
clinic_cases    = RNG.integers(0, 300, N)
pop_density     = RNG.uniform(10, 800, N)
clean_water     = RNG.integers(0, 2, N)
temperature     = RNG.uniform(15, 38, N)
disease_code    = RNG.integers(0, 5, N)

# Outbreak probability driven by environmental factors
outbreak_score = (
    rainfall_30d / 500 * 0.30
    + clinic_cases / 300 * 0.35
    + (1 - clean_water) * 0.20
    + (temperature - 20) / 18 * 0.10
    + pop_density / 800 * 0.05
    + RNG.uniform(-0.1, 0.1, N)
)
y_disease = (outbreak_score > 0.45).astype(int)

X_disease = np.column_stack([rainfall_30d, clinic_cases, pop_density, clean_water, temperature, disease_code])
X_train, X_test, y_train, y_test = train_test_split(X_disease, y_disease, test_size=0.2, random_state=42)

disease_model = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
disease_model.fit(X_train, y_train)

acc = accuracy_score(y_test, disease_model.predict(X_test))
print(f"  Accuracy: {acc:.2%}")
print(classification_report(y_test, disease_model.predict(X_test), target_names=["no_outbreak", "outbreak"]))

joblib.dump({
    "model": disease_model,
    "description": "Disease outbreak risk prediction for Ugandan districts",
    "features": ["rainfall_mm_last_30d", "clinic_cases_last_week", "population_density",
                 "has_clean_water_access", "temperature_c", "disease_code"],
    "accuracy": round(acc, 4),
    "trained_on": TODAY,
}, MODELS_DIR / "disease_risk.joblib")
print("  Saved: models/disease_risk.joblib")


# ---------------------------------------------------------------------------
# 3. FRAUD DETECTION MODEL
# ---------------------------------------------------------------------------
print("\n[3/3] Training fraud detection model (RandomForestClassifier)...")

N = 12000
amount           = RNG.exponential(200000, N)       # UGX
sender_age       = RNG.integers(0, 1825, N)          # account days
receiver_age     = RNG.integers(0, 1825, N)
txn_last_24h     = RNG.integers(0, 30, N)
is_new_recipient = RNG.integers(0, 2, N)
txn_hour         = RNG.integers(0, 24, N)
same_district    = RNG.integers(0, 2, N)
is_night         = ((txn_hour < 5) | (txn_hour >= 22)).astype(int)
device_changed   = RNG.integers(0, 2, N)

fraud_score = (
    (amount > 1_000_000) * 0.20
    + (receiver_age < 7) * 0.25
    + is_new_recipient * 0.15
    + (txn_last_24h > 10) * 0.10
    + is_night * 0.10
    + device_changed * 0.10
    + (1 - same_district) * 0.05
    + (sender_age < 30) * 0.05
    + RNG.uniform(-0.05, 0.05, N)
)
y_fraud = (fraud_score > 0.40).astype(int)

X_fraud = np.column_stack([
    amount, sender_age, receiver_age, txn_last_24h,
    is_new_recipient, txn_hour, same_district, is_night, device_changed
])
X_train, X_test, y_train, y_test = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)

fraud_model = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1,
                                     class_weight="balanced", random_state=42)
fraud_model.fit(X_train, y_train)

acc = accuracy_score(y_test, fraud_model.predict(X_test))
print(f"  Accuracy: {acc:.2%}")
print(classification_report(y_test, fraud_model.predict(X_test), target_names=["legit", "fraud"]))

joblib.dump({
    "model": fraud_model,
    "description": "Mobile money fraud detection for Uganda (MTN MoMo / Airtel Money)",
    "features": ["transaction_amount_ugx", "sender_account_age_days", "receiver_account_age_days",
                 "transactions_last_24h", "is_new_recipient", "transaction_hour",
                 "same_district", "is_night", "device_changed_recently"],
    "accuracy": round(acc, 4),
    "trained_on": TODAY,
}, MODELS_DIR / "fraud_detection.joblib")
print("  Saved: models/fraud_detection.joblib")

print("\n" + "=" * 60)
print("  All 3 models trained and saved successfully.")
print("  Start the API with: uvicorn app.main:app --reload")
print("=" * 60)
