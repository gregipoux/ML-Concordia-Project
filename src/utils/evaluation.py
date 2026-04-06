"""
Model evaluation utilities.
Computes all required metrics and generates comparison tables/plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from typing import Dict, List, Any
import os


def compute_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """Compute all classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str = None):
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_roc_curves(results: Dict[str, Dict], save_path: str = None):
    """
    Plot ROC curves for all models on the same figure.
    results: {model_name: {"y_true": ..., "y_proba": ...}}
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, data in results.items():
        if data.get("y_proba") is not None:
            fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
            auc = roc_auc_score(data["y_true"], data["y_proba"])
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
    return fig


def comparison_table(all_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create a comparison DataFrame from all model metrics."""
    df = pd.DataFrame(all_metrics).T
    df.index.name = "Model"
    df = df.round(4)
    return df.sort_values("f1_score", ascending=False)


def print_classification_report(y_true, y_pred, model_name: str):
    """Print sklearn classification report."""
    print(f"\n{'='*50}")
    print(f"Classification Report — {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"]))
