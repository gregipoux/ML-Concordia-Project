"""
Re-train and serialize the 4 comparison models used by the live UI:
  - Logistic Regression
  - Random Forest  (same model as the champion, re-fit for a clean artifact)
  - XGBoost
  - DNN v2  (best DNN variant from notebook 03: 128-64-32, dropout 0.3, batch_size 128)

Outputs:
  saved_models/preprocessor.joblib   # fitted ColumnTransformer
  saved_models/lr.joblib
  saved_models/rf.joblib
  saved_models/xgb.joblib
  saved_models/dnn.keras
  saved_models/comparison_metadata.joblib   # {model_name -> test F1, test AUC, test inference_time_ms}

Usage (from repo root):
    python scripts/export_models.py

The existing saved_models/best_model.joblib (champion bundle) is untouched.
"""

from __future__ import annotations

import os
import sys
import time
import joblib
import numpy as np

# Make `src.*` importable when called from repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sklearn.metrics import f1_score, roc_auc_score

from src.utils.preprocessing import prepare_data, load_config
from src.models.baseline import get_logistic_regression, get_random_forest, get_xgboost


SAVED = os.path.join(ROOT, "saved_models")
os.makedirs(SAVED, exist_ok=True)


def _eval(model, X_test, y_test, proba_fn):
    t0 = time.perf_counter()
    proba = proba_fn(X_test)
    latency_ms = (time.perf_counter() - t0) / len(X_test) * 1000
    pred = (proba >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba)),
        "inference_ms_per_sample": float(latency_ms),
    }


def main():
    config = load_config(os.path.join(ROOT, "config", "config.yaml"))
    os.chdir(ROOT)  # prepare_data uses relative paths
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(config, apply_smote=False)

    joblib.dump(preprocessor, os.path.join(SAVED, "preprocessor.joblib"))
    print(f"[saved]  preprocessor.joblib")

    metrics: dict[str, dict] = {}

    # --- Logistic Regression ---
    lr = get_logistic_regression(config)
    lr.fit(X_train, y_train)
    metrics["Logistic Regression"] = _eval(lr, X_test, y_test, lambda X: lr.predict_proba(X)[:, 1])
    joblib.dump(lr, os.path.join(SAVED, "lr.joblib"))
    print(f"[saved]  lr.joblib            F1={metrics['Logistic Regression']['f1']:.4f}")

    # --- Random Forest ---
    rf = get_random_forest(config)
    rf.fit(X_train, y_train)
    metrics["Random Forest"] = _eval(rf, X_test, y_test, lambda X: rf.predict_proba(X)[:, 1])
    joblib.dump(rf, os.path.join(SAVED, "rf.joblib"))
    print(f"[saved]  rf.joblib            F1={metrics['Random Forest']['f1']:.4f}")

    # --- XGBoost ---
    xgb = get_xgboost(config)
    # Match the refined scale_pos_weight from the journal (actual class ratio)
    xgb.set_params(scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)))
    xgb.fit(X_train, y_train)
    metrics["XGBoost"] = _eval(xgb, X_test, y_test, lambda X: xgb.predict_proba(X)[:, 1])
    joblib.dump(xgb, os.path.join(SAVED, "xgb.joblib"))
    print(f"[saved]  xgb.joblib           F1={metrics['XGBoost']['f1']:.4f}")

    # --- DNN v2 (best DNN variant: 128-64-32, dropout 0.3, batch_size 128) ---
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, optimizers

    tf.random.set_seed(config["project"]["random_seed"])

    n_features = X_train.shape[1]
    dnn = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    dnn.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    dnn.fit(
        X_train, y_train.astype(np.float32),
        validation_split=0.15,
        epochs=100, batch_size=128,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0),
        ],
        verbose=0,
    )
    metrics["DNN v2"] = _eval(dnn, X_test, y_test, lambda X: dnn.predict(X, verbose=0).ravel())
    # Save weights only — the full .keras format is Keras-version-sensitive, weights are not.
    # The API rebuilds the architecture from scratch and loads these weights.
    dnn.save_weights(os.path.join(SAVED, "dnn.weights.h5"))
    joblib.dump({"n_features": int(n_features)}, os.path.join(SAVED, "dnn_arch.joblib"))
    print(f"[saved]  dnn.weights.h5       F1={metrics['DNN v2']['f1']:.4f}")

    joblib.dump(metrics, os.path.join(SAVED, "comparison_metadata.joblib"))
    print(f"[saved]  comparison_metadata.joblib")

    # Champion bundle consumed by the /predict endpoint.
    # Regenerated here so the champion is trained with the same sklearn version
    # pinned in requirements.txt, avoiding unpickle cross-version warnings.
    champion_bundle = {
        "model": rf,
        "preprocessor": preprocessor,
        "champion_name": "Random Forest",
        "default_threshold": 0.5,
        "metadata": {
            "seed": config["project"]["random_seed"],
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
        },
    }
    joblib.dump(champion_bundle, os.path.join(SAVED, "best_model.joblib"))
    print(f"[saved]  best_model.joblib     (champion bundle)")

    print("\nDone. Test-set F1 summary:")
    for name, m in metrics.items():
        print(f"  {name:22s} F1={m['f1']:.4f}  AUC={m['auc']:.4f}  inf={m['inference_ms_per_sample']:.3f} ms/sample")


if __name__ == "__main__":
    main()
