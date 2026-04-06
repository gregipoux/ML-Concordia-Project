"""
Deep Learning models for Cybersecurity Intrusion Detection.
DNN (Dense Neural Network) with TensorFlow/Keras.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import mlflow
import mlflow.tensorflow
from typing import Dict, Tuple


def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_dnn(input_dim: int, config: dict) -> keras.Model:
    """
    Build a Dense Neural Network for binary classification.
    Architecture from config: layers, dropout, activations.
    """
    dl_config = config["models"]["deep_learning"]["dnn"]

    model = keras.Sequential(name="DNN_IntrusionDetection")
    model.add(layers.Input(shape=(input_dim,)))

    for i, units in enumerate(dl_config["layers"]):
        model.add(layers.Dense(units, activation=dl_config["activation"], name=f"dense_{i}"))
        model.add(layers.BatchNormalization(name=f"bn_{i}"))
        model.add(layers.Dropout(dl_config["dropout"], name=f"dropout_{i}"))

    model.add(layers.Dense(1, activation=dl_config["output_activation"], name="output"))

    optimizer = keras.optimizers.Adam(learning_rate=dl_config["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss=dl_config["loss"],
        metrics=["accuracy", keras.metrics.AUC(name="auc"), keras.metrics.Precision(), keras.metrics.Recall()],
    )

    return model


def get_callbacks(config: dict) -> list:
    """Get training callbacks (early stopping, etc.)."""
    dl_config = config["models"]["deep_learning"]["dnn"]
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=dl_config["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train_dnn(X_train, y_train, X_test, y_test, config: dict) -> Tuple[keras.Model, dict]:
    """Train DNN and log to MLflow."""
    from src.utils.evaluation import compute_metrics

    set_seeds(config["project"]["random_seed"])
    dl_config = config["models"]["deep_learning"]["dnn"]

    model = build_dnn(X_train.shape[1], config)
    model.summary()

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="DNN"):
        # Log architecture params
        mlflow.log_params({
            "model_type": "DNN",
            "layers": str(dl_config["layers"]),
            "dropout": dl_config["dropout"],
            "learning_rate": dl_config["learning_rate"],
            "batch_size": dl_config["batch_size"],
            "epochs": dl_config["epochs"],
        })

        # Train
        history = model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=dl_config["epochs"],
            batch_size=dl_config["batch_size"],
            callbacks=get_callbacks(config),
            verbose=1,
        )

        # Predict
        y_proba = model.predict(X_test).flatten()
        y_pred = (y_proba >= 0.5).astype(int)

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_proba)

        # Log
        mlflow.log_metrics(metrics)
        mlflow.tensorflow.log_model(model, "dnn_model")

        print(f"[DNN] F1: {metrics['f1_score']:.4f} | AUC: {metrics.get('roc_auc', 'N/A')}")

    return model, metrics, y_pred, y_proba, history


def build_autoencoder(input_dim: int, encoding_dim: int = 16) -> Tuple[keras.Model, keras.Model]:
    """
    Build an Autoencoder for anomaly detection (unsupervised approach).
    Returns (autoencoder, encoder).
    """
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation="relu")(inputs)
    encoded = layers.Dense(32, activation="relu")(encoded)
    encoded = layers.Dense(encoding_dim, activation="relu")(encoded)

    # Decoder
    decoded = layers.Dense(32, activation="relu")(encoded)
    decoded = layers.Dense(64, activation="relu")(decoded)
    decoded = layers.Dense(input_dim, activation="sigmoid")(decoded)

    autoencoder = keras.Model(inputs, decoded, name="Autoencoder_AnomalyDetection")
    encoder = keras.Model(inputs, encoded, name="Encoder")

    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder, encoder
