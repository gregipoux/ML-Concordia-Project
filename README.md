# Cybersecurity Intrusion Detection (ML Project)

**MOD10 Machine Learning, Winter 2026, Concordia**
**Instructor:** Mohammed A. Shehab &nbsp;·&nbsp; **Group 07** &nbsp;·&nbsp; **Team of 4**

Binary intrusion detector trained on the Kaggle *Cybersecurity Intrusion Detection* dataset (9 537 sessions, 9 features). Random Forest is our deployed champion (F1 = 0.855 on the held-out test set). It is served through a FastAPI service with a Gradio UI that shows live SHAP explanations and a side-by-side comparison of four candidate models. Everything launches with a single `docker compose up -d`.

## Quick start

```bash
cd docker
docker compose up -d
# wait ~7 seconds for the container to become healthy
open http://localhost:8000     # or just visit it in any browser
```

That is the full deployment contract. No `pip install`, no model training, no manual downloads. Pre-fitted model artefacts are committed to `saved_models/` so the container is ready on first boot.

The full report is at [`reports/FINAL_REPORT.pdf`](reports/FINAL_REPORT.pdf) (24 pages, 21 figures). The Markdown source is `reports/FINAL_REPORT.md` and the LaTeX is `reports/FINAL_REPORT.tex`.

## Project structure

```
repo/
├── config/
│   └── config.yaml                 # hyperparameters, paths, seeds
├── data/
│   └── cybersecurity_intrusion_data.csv
├── notebooks/
│   ├── 01_EDA_preprocessing.ipynb            # 8 EDA figures + preprocessing pipeline
│   ├── 02_model_training_baselines.ipynb     # Dummy / LR / RF / XGBoost + MLflow
│   ├── 03_model_training_deep_learning.ipynb # 4 DNN variants + MLflow
│   ├── 04_interpretability_shap.ipynb        # SHAP TreeExplainer + local cases
│   └── 05_evaluation_comparison.ipynb        # 10-model comparison + champion bundle
├── scripts/
│   └── export_models.py            # reproducible training of the 4 comparison models
├── src/
│   ├── api/app.py                  # FastAPI service (/predict, /health, /model/info, /ui mount)
│   ├── ui/gradio_app.py            # Gradio UI: form + SHAP waterfall + 4-model live comparison
│   ├── models/
│   │   ├── baseline.py             # LR / RF / XGBoost builders
│   │   └── deep_learning.py        # parametric DNN builder
│   └── utils/
│       ├── preprocessing.py        # load + engineer + split + fit ColumnTransformer
│       ├── feature_engineering.py  # 4 engineered features (reused at training and inference)
│       └── evaluation.py           # metric helpers
├── docker/
│   ├── Dockerfile                  # python:3.10-slim + deps + app code + saved_models
│   └── docker-compose.yml          # one ids-api service; mlflow under `--profile tools`
├── saved_models/                   # committed — container starts with these pre-fitted artefacts
│   ├── best_model.joblib           # champion bundle (RF + preprocessor + metadata)
│   ├── preprocessor.joblib
│   ├── lr.joblib / rf.joblib / xgb.joblib    # comparison models (classical)
│   ├── dnn.weights.h5              # DNN weights (arch reconstructed at load time)
│   ├── dnn_arch.joblib             # {n_features: 18}
│   └── comparison_metadata.joblib  # per-model test F1/AUC for the UI
├── reports/
│   ├── FINAL_REPORT.md             # report source
│   ├── FINAL_REPORT.tex            # LaTeX generated via pandoc
│   ├── FINAL_REPORT.pdf            # the final report
│   └── figures/                    # 21 figures (EDA + training + SHAP + comparison) + confusion matrices
├── requirements.txt                # Python deps, scikit-learn pinned to 1.7.2
└── README.md                       # this file
```

## What runs inside the container

A single `python:3.10-slim`-based image launches `uvicorn src.api.app:app`. The FastAPI app does three things:

1. **REST API.** `POST /predict` takes a JSON session and returns `{prediction, probability, label, risk_level}`. `GET /health` for probes, `GET /model/info` for metadata.
2. **Gradio UI** mounted at `/ui` via `gr.mount_gradio_app()`. Same process, same loaded model, no network hop between UI and inference.
3. **Root redirect** (`/` to `/ui`) so a user opening `http://localhost:8000` lands directly on the interactive page.

The Gradio UI ships with three one-click presets (**Normal traffic**, **Obvious attack**, **Edge case**) that showcase the champion's behaviour, a live SHAP waterfall for every prediction, and a side-by-side comparison of four models (LR, RF ★ champion, XGBoost, DNN v2). Each card shows the current-sample probability *and* the model's global F1 on the test set, which makes the champion choice self-explanatory even when the per-sample probabilities are close.

## API endpoints

| Method | Path              | Description                                                           |
|-------:|-------------------|-----------------------------------------------------------------------|
|    GET | `/`               | Redirect to `/ui`                                                     |
|    GET | `/ui`             | Gradio interactive interface                                          |
|    GET | `/health`         | Liveness/readiness probe                                              |
|    GET | `/model/info`     | Model class, path, default threshold, training metadata               |
|   POST | `/predict`        | Predict from a session — see schema at `/docs` (Swagger) or `/redoc`  |

Example `curl`:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "network_packet_size": 200, "protocol_type": "UDP",
    "login_attempts": 10, "session_duration": 30.0,
    "encryption_used": "None", "ip_reputation_score": 0.85,
    "failed_logins": 5, "browser_type": "Unknown",
    "unusual_time_access": 1
  }'
# {"prediction":1,"probability":0.99,"label":"Attack Detected","risk_level":"CRITICAL"}
```

## Reproducing the training

The committed `saved_models/` lets the container run without retraining. To regenerate them from scratch:

```bash
python scripts/export_models.py
```

This trains Logistic Regression, Random Forest, XGBoost and DNN v2 on the full training split, saves each under `saved_models/`, and regenerates `best_model.joblib` (the champion bundle consumed by `/predict`). Runs in under two minutes on a CPU-only laptop.

The five notebooks in `notebooks/` are the exploratory record used to produce the figures in `reports/figures/` and to select the champion. They all read the same preprocessing pipeline from `src/utils/preprocessing.py`, so numbers are consistent across them.

## MLflow (optional)

We used MLflow to track the 10 runs logged across notebooks 02, 03 and 05. The runs sit in `mlruns/` (gitignored). To inspect them locally:

```bash
mlflow ui --backend-store-uri file:./mlruns
# then open http://127.0.0.1:5000
```

The `mlflow` service in `docker-compose.yml` sits behind `--profile tools` (not started by `docker compose up -d`). If you want to launch it too:

```bash
docker compose --profile tools up -d
```

Note that run `meta.yaml` files reference absolute host paths, so the MLflow UI shows runs and metrics correctly but will not open per-run artefacts from within the container. The training metrics and figures are all captured in the final report.

## Development

```bash
# install deps
pip install -r requirements.txt

# run the API + UI locally (no Docker)
uvicorn src.api.app:app --reload --port 8000

# regenerate models if you change the preprocessing or hyperparams
python scripts/export_models.py

# regenerate the report PDF (requires Docker)
cd reports
docker run --rm -v "$(pwd):/data" -w /data pandoc/latex:3.1 \
    FINAL_REPORT.md -o FINAL_REPORT.pdf \
    --pdf-engine=xelatex --toc --number-sections --shift-heading-level-by=-1 \
    -V documentclass=article -V geometry:margin=2cm -V papersize:a4 -V fontsize:11pt
```

Key design decisions:

- **`scikit-learn==1.7.2` is pinned** because Logistic Regression serialised with 1.8 crashes in 1.7 (the `multi_class` attribute was removed).
- **The DNN is serialised as weights plus an architecture descriptor**, not as a full `.keras` file, because the `.keras` format is sensitive to Keras minor versions (we hit a `quantization_config` incompatibility). `scripts/export_models.py` saves `dnn.weights.h5` and a one-line `dnn_arch.joblib`; the API reconstructs the same `Sequential` model and calls `load_weights` at startup.
- **`saved_models/` is committed**, which is non-standard. We chose this so anyone can run `docker compose up -d` straight from the ZIP without any preparatory step. Only `best_model.joblib` and the seven comparison artefacts are tracked; other `.joblib`/`.pkl` files remain gitignored.
- **`mlruns/` is gitignored.** The run artefacts reference absolute host paths, so including them would produce broken links on any other machine. The figures we care about are exported to `reports/figures/`.
- **The Gradio UI is mounted on FastAPI, not in a separate container.** This keeps one process, one memory footprint, and no internal HTTP hops. The model and the SHAP explainer are shared between the REST API and the interactive UI.

## Team and roles

| Member                            | Role                                          |
|-----------------------------------|-----------------------------------------------|
| BELLEPERCHE Grégoire              | ML Engineer (DNN, MLflow tracking, UI)        |
| QUERREC Thomas                    | Backend / DevOps (preprocessing, FastAPI, Docker) |
| Montenegro Loureiro Marco-Antonio | Data Analyst (EDA, metrics, report writing)   |
| Relut-Vainqueur Xavier            | Research / Presentation (feature eng, SHAP, slides) |

## Dataset

- **Source:** [Cybersecurity Intrusion Detection Dataset — Kaggle](https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset)
- **Records:** 9 537 network sessions, 11 columns (`session_id` + `attack_detected` + 9 features)
- **Target:** `attack_detected` (binary: 0 = Normal, 1 = Attack)
- **Raw features:** `network_packet_size`, `protocol_type`, `login_attempts`, `session_duration`, `encryption_used`, `ip_reputation_score`, `failed_logins`, `browser_type`, `unusual_time_access`
- **Engineered features** (in `src/utils/feature_engineering.py`): `login_fail_ratio`, `packet_rate`, `risk_score`, `high_risk_ip`

## Results in one table

| Model                       | F1         | ROC-AUC | Inference time |
|-----------------------------|-----------:|--------:|---------------:|
| Dummy (most_frequent)       |     0.0000 |  0.5000 |         4.1 ms |
| Logistic Regression         |     0.7297 |  0.8164 |         0.2 ms |
| DNN v2 (128-64-32, batch=128) |   0.8493 |  0.8816 |       146 ms   |
| XGBoost                     |     0.8507 |  0.8832 |         1.8 ms |
| **Random Forest (champion)**|     **0.8550** | 0.8830 |   **67.8 ms** |

All numbers on the stratified 20% test split (1908 sessions). The champion bundle is at `saved_models/best_model.joblib`.
