# Cybersecurity Intrusion Detection — ML Project

**MOD10: Machine Learning — Winter 2026 — Concordia**
**Instructor:** Mohammed A. Shehab

## Overview

Machine learning-based intrusion detection system (IDS) that analyzes network traffic and user behavior patterns to detect cyber intrusions in real-time. Combines classical ML baselines with Deep Learning models, deployed via Docker with a FastAPI REST API and tracked through MLflow.

## Dataset

- **Source:** [Cybersecurity Intrusion Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/)
- **Records:** 9,537 network sessions
- **Target:** `attack_detected` (binary: 0 = normal, 1 = attack)
- **Features:** 10 (network packet size, protocol type, login attempts, session duration, encryption used, IP reputation score, failed logins, browser type, unusual time access)

## Project Structure

```
repo/
├── config/                    # Configuration files
│   └── config.yaml
├── data/                      # Dataset
│   └── cybersecurity_intrusion_data.csv
├── notebooks/                 # Jupyter notebooks (EDA, training, eval)
│   ├── 01_EDA_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training_baselines.ipynb
│   ├── 04_model_training_deep_learning.ipynb
│   ├── 05_evaluation_comparison.ipynb
│   └── 06_interpretability.ipynb
├── src/                       # Source code modules
│   ├── api/                   # FastAPI app
│   │   └── app.py
│   ├── models/                # Model definitions
│   │   ├── baseline.py
│   │   └── deep_learning.py
│   └── utils/                 # Preprocessing, evaluation helpers
│       ├── preprocessing.py
│       ├── evaluation.py
│       └── feature_engineering.py
├── docker/                    # Docker deployment
│   ├── Dockerfile
│   └── docker-compose.yml
├── mlflow/                    # MLflow experiment tracking
├── reports/                   # Report assets
│   └── figures/
├── presentation/              # PPT files
├── tests/                     # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

```bash
# Clone
git clone https://github.com/gregipoux/ML-Concordia-Project.git
cd ML-Concordia-Project

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order (01 → 06)
jupyter notebook notebooks/

# Launch API
uvicorn src.api.app:app --reload --port 8000

# Docker deployment
cd docker
docker-compose up --build
```

## Models Implemented

| Model | Type | Purpose |
|---|---|---|
| Logistic Regression | ML Baseline | Simple linear baseline |
| Random Forest | ML Baseline | Tree-based baseline |
| XGBoost | ML Ensemble | Gradient boosting |
| Dense Neural Network (DNN) | Deep Learning | Main DL model |
| Voting / Stacking Ensemble | Ensemble | Best model combination |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Predict intrusion from features |
| GET | `/model/info` | Model metadata and version |

## Team

| Member                             | Role |
|------------------------------------|---|
| BELLEPERCHE Grégoire               | ML Engineer |
| QUERREC Thomas                     | Backend / DevOps |
| Montenegro Loureiro  Marco-Antonio | Data Analyst |
| Relut-Vainqueur  Xavier            | Research / Presentation |

