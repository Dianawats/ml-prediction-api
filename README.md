# ML Prediction API

> Real-time machine learning predictions for Uganda, Africa, and the world.
> Built from scratch — crop yield forecasting, disease outbreak detection, and mobile money fraud scoring.

**Live Dashboard → [ml-prediction-api.vercel.app](https://ml-prediction-api.vercel.app)**

---

## What this is

A production-ready REST API that wraps trained scikit-learn models behind clean endpoints. Any app — a mobile phone in Kampala, a Ministry of Health dashboard, or an external partner system — sends a JSON request and gets a machine learning prediction back in milliseconds.

Built with real Ugandan context in mind: districts, crops, diseases, and mobile money patterns that matter locally.

---

## Live demo

Visit **[ml-prediction-api.vercel.app](https://ml-prediction-api.vercel.app)** to see the dashboard. To run predictions, you need the Python API running (see setup below).

---

## The 3 models

| Model | Type | Accuracy | Use case |
|---|---|---|---|
| Crop Yield | RandomForestRegressor | 93% | Predict kg/acre for Ugandan farms |
| Disease Risk | GradientBoostingClassifier | 88% | Outbreak risk by district |
| Fraud Detection | RandomForestClassifier | 95% | Mobile money transaction scoring |

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System health + loaded models |
| POST | `/predict/crop-yield` | Crop yield forecast |
| POST | `/predict/disease-risk` | Disease outbreak risk |
| POST | `/predict/fraud` | Mobile money fraud score |
| POST | `/predict/` | Generic — any model, any features |
| GET | `/models/` | List all loaded models |
| GET | `/docs` | Interactive API playground |

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/Dianawats/ml-prediction-api
cd ml-prediction-api
```

### 2. Set up Python environment

```bash
# Windows
py -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the models

```bash
python scripts/train_models.py
```

Output:
```
[1/3] Training crop yield model...     ✓ Saved models/crop_yield.joblib
[2/3] Training disease risk model...   ✓ Saved models/disease_risk.joblib
[3/3] Training fraud detection model...✓ Saved models/fraud_detection.joblib
```

### 5. Start the API

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive playground.

---

## Example requests

### Crop yield — Mbale maize farm

```bash
curl -X POST http://localhost:8000/predict/crop-yield \
  -H "Content-Type: application/json" \
  -d '{
    "region": "Mbale",
    "crop_type": "maize",
    "rainfall_mm": 145.0,
    "temperature_c": 22.5,
    "soil_type": "loam",
    "planting_month": 3,
    "farm_size_acres": 2.5
  }'
```

Response:
```json
{
  "region": "Mbale",
  "crop_type": "maize",
  "predicted_yield_kg_per_acre": 1738.1,
  "risk_level": "low",
  "confidence": 0.94,
  "recommendation": "Conditions look favourable for maize in Mbale...",
  "processing_time_ms": 72.07
}
```

### Disease risk — Gulu malaria

```bash
curl -X POST http://localhost:8000/predict/disease-risk \
  -H "Content-Type: application/json" \
  -d '{
    "district": "Gulu",
    "disease": "malaria",
    "rainfall_mm_last_30d": 210.0,
    "clinic_cases_last_week": 87,
    "population_density": 145.0,
    "has_clean_water_access": false,
    "temperature_c": 28.5
  }'
```

### Mobile money fraud — MTN MoMo

```bash
curl -X POST http://localhost:8000/predict/fraud \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount_ugx": 500000,
    "sender_account_age_days": 365,
    "receiver_account_age_days": 3,
    "transactions_last_24h": 12,
    "is_new_recipient": true,
    "transaction_hour": 2,
    "sender_district": "Kampala",
    "receiver_district": "Wakiso",
    "device_changed_recently": true
  }'
```

---

## Project structure

```
ml_prediction_api/
├── app/
│   ├── main.py                 # FastAPI app, middleware, startup
│   ├── schemas.py              # Pydantic request/response models
│   ├── core/
│   │   ├── config.py           # Settings from environment variables
│   │   └── model_registry.py  # Auto-loads all .joblib models
│   └── routers/
│       ├── predictions.py      # /predict/* endpoints
│       ├── models.py           # /models/* endpoints
│       └── health.py           # /health, /ping
├── models/                     # Trained .joblib files
├── scripts/
│   └── train_models.py         # Generates data + trains 3 models
├── tests/
│   └── test_api.py             # 30+ pytest tests
├── nginx/
│   └── nginx.conf              # Reverse proxy + rate limiting
├── index.html                  # Web dashboard (deployed on Vercel)
├── Dockerfile                  # Multi-stage production build
├── docker-compose.yml          # API + Nginx stack
└── requirements.txt
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.14 |
| API framework | FastAPI |
| ML models | scikit-learn |
| Validation | Pydantic v2 |
| Server | Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Vercel (dashboard) |
| Containerisation | Docker |
| Reverse proxy | Nginx |
| Tests | pytest |

---

## Deploying with Docker

```bash
# Build
docker build -t ml-prediction-api .

# Run
docker run -p 8000:8000 -v $(pwd)/models:/app/models ml-prediction-api

# Full stack with Nginx
docker compose up -d
```

---

## Adding your own model

Train any scikit-learn model and drop it in:

```python
import joblib

joblib.dump({
    "model": my_trained_model,
    "description": "My custom Uganda model",
    "features": ["feature1", "feature2"],
    "accuracy": 0.91,
}, "models/my_model.joblib")
```

Restart the API — it auto-discovers all `.joblib` files. Call it via:

```bash
POST /predict/  →  { "model_name": "my_model", "features": {...} }
```

---

## Roadmap

- [ ] Deploy Python API to Railway / Render (fully public predictions)
- [ ] Connect real NARO crop datasets
- [ ] Connect Uganda Ministry of Health district data
- [ ] SMS alerts via Africa's Talking API
- [ ] Mobile app (React Native)
- [ ] User authentication + API keys
- [ ] Model retraining pipeline
- [ ] Energy load prediction model (UEDCL)
- [ ] Credit scoring model (Ugandan SACCOs)

---

## Built by

**Diana Nakiwala** · 

Built to solve real problems in agriculture, public health, and financial security across Uganda and Africa.

---

## License

MIT — free to use, modify, and deploy.
