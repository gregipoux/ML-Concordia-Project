"""
Baseline ML models: Logistic Regression, Random Forest, XGBoost.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
from typing import Dict, Any


def get_logistic_regression(config: dict) -> LogisticRegression:
    params = config["models"]["baselines"]["logistic_regression"]
    return LogisticRegression(
        max_iter=params["max_iter"],
        class_weight=params["class_weight"],
        random_state=config["project"]["random_seed"],
    )


def get_random_forest(config: dict) -> RandomForestClassifier:
    params = config["models"]["baselines"]["random_forest"]
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        class_weight=params["class_weight"],
        random_state=config["project"]["random_seed"],
        n_jobs=-1,
    )


def get_xgboost(config: dict) -> XGBClassifier:
    params = config["models"]["baselines"]["xgboost"]
    return XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        scale_pos_weight=params["scale_pos_weight"],
        random_state=config["project"]["random_seed"],
        eval_metric="logloss",
        use_label_encoder=False,
    )


def get_all_baselines(config: dict) -> Dict[str, Any]:
    """Return dict of all baseline models."""
    return {
        "Logistic Regression": get_logistic_regression(config),
        "Random Forest": get_random_forest(config),
        "XGBoost": get_xgboost(config),
    }


def get_voting_ensemble(config: dict) -> VotingClassifier:
    """Soft voting ensemble of all baselines."""
    baselines = get_all_baselines(config)
    estimators = [(name.lower().replace(" ", "_"), model) for name, model in baselines.items()]
    return VotingClassifier(
        estimators=estimators,
        voting=config["models"]["ensemble"]["voting"],
    )


def get_stacking_ensemble(config: dict) -> StackingClassifier:
    """Stacking ensemble with Logistic Regression meta-learner."""
    baselines = get_all_baselines(config)
    estimators = [(name.lower().replace(" ", "_"), model) for name, model in baselines.items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
    )


def train_and_log_model(model, model_name: str, X_train, y_train, X_test, y_test, config: dict):
    """Train a model and log to MLflow."""
    from src.utils.evaluation import compute_metrics

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_proba)

        # Log to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_name.lower().replace(" ", "_"))

        print(f"[{model_name}] F1: {metrics['f1_score']:.4f} | AUC: {metrics.get('roc_auc', 'N/A')}")

    return model, metrics, y_pred, y_proba
