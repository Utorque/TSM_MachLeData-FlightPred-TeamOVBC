# TSM MachLeData - Flight Price Prediction

ML pipeline with continuous training, monitoring, and automated deployment.

## Dataset
[Kaggle - Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

## Features
- CI/CD with GitHub Actions
- Data drift simulation (weekly splits)
- Continuous model retraining
- Performance & drift monitoring
- Model optimization (pruning, quantization)
- REST API deployment

## Setup docker - FastAPI Server

```
# build image
docker build -t flight-api -f docker/Dockerfile .

# run container
docker run -p 8000:8000 flight-api

# access API
http://localhost:8000/docs
```
## Plan Proposition
GitHub Repo → GitHub Actions → MLflow (VM) → FastAPI (VM)

### Components

**On VM:**
- MLflow tracking server (port 5000)
- FastAPI app (port 8000)
- Both running as systemd services or Docker containers

**In GitHub:**
- Code + 7 weekly data folders
- GitHub Actions workflows
- Secrets: VM_IP, SSH_KEY, MLFLOW_TRACKING_URI

---

## GitHub Actions Workflows

**1. `weekly_training.yml`** (Manual trigger with week parameter)
- Load data for specified week
- Train model
- Log to MLflow (metrics, params, model)
- Calculate drift vs previous week
- If good → promote model in MLflow registry

**2. `deploy.yml`** (Triggered after model promotion)
- SSH into VM
- Pull latest model from MLflow
- Restart FastAPI service
- Health check

**3. `monitor.yml`** (Optional)
- Query MLflow for last 7 days metrics
- Generate drift report
- Post to GitHub Issues if drift detected

---

## MLflow Usage
```
MLflow Tracking Server (VM)
├── Experiments
│   ├── week_1_training
│   ├── week_2_training
│   └── ...
├── Models Registry
│   └── flight_price_model
│       ├── v1 (week_1)
│       ├── v2 (week_2) [Production]
│       └── ...
└── Artifacts
    ├── models/
    ├── drift_reports/
    └── feature_distributions/
```

**What goes in MLflow:**
- Training: `mlflow.log_metrics()`, `mlflow.sklearn.log_model()`
- Drift: `mlflow.log_artifact(drift_report.json)`
- Production: `mlflow.register_model()` → stage to "Production"

---

## Simple Execution Flow
```
Week 1: Manual trigger workflow → Train → Log to MLflow → Deploy
Week 2: Manual trigger workflow → Train → Detect drift → Log → Compare → Deploy if better
Week 3-7: Repeat
