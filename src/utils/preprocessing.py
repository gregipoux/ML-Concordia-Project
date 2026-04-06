"""
Data preprocessing pipeline for Cybersecurity Intrusion Detection.
Handles loading, cleaning, encoding, scaling, and train/test split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import yaml
import os


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load project configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    """Load the dataset from CSV."""
    df = pd.read_csv(config["data"]["path"])
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """Quick data inspection — returns summary dict for EDA."""
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "target_distribution": df["attack_detected"].value_counts().to_dict(),
        "class_balance_ratio": df["attack_detected"].value_counts(normalize=True).to_dict(),
    }
    return summary


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing ones."""
    df = df.copy()

    # Login fail ratio (handle division by zero)
    df["login_fail_ratio"] = np.where(
        df["login_attempts"] > 0,
        df["failed_logins"] / df["login_attempts"],
        0
    )

    # Packet rate (bytes per second of session)
    df["packet_rate"] = np.where(
        df["session_duration"] > 0,
        df["network_packet_size"] / df["session_duration"],
        0
    )

    # Composite risk score
    df["risk_score"] = (
        df["ip_reputation_score"] * 0.4
        + df["unusual_time_access"] * 0.3
        + df["login_fail_ratio"] * 0.3
    )

    # High risk flag
    df["high_risk_ip"] = (df["ip_reputation_score"] > 0.7).astype(int)

    print(f"Features engineered. New shape: {df.shape}")
    return df


def build_preprocessor(config: dict) -> ColumnTransformer:
    """Build sklearn ColumnTransformer for preprocessing."""
    cat_features = config["data"]["categorical_features"]
    num_features = config["data"]["numerical_features"]
    # Add engineered numerical features
    engineered_num = ["login_fail_ratio", "packet_rate", "risk_score"]

    all_num = num_features + engineered_num

    scaler = StandardScaler() if config["preprocessing"]["scaling"] == "standard" else MinMaxScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, all_num),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_features),
            ("bin", "passthrough", config["data"]["binary_features"] + ["high_risk_ip"]),
        ]
    )
    return preprocessor


def prepare_data(config: dict, apply_smote: bool = True):
    """
    Full pipeline: load → clean → engineer → split → preprocess → (optional) SMOTE.
    Returns X_train, X_test, y_train, y_test, preprocessor.
    """
    df = load_data(config)

    # Drop ID column
    df = df.drop(columns=[config["data"]["id_column"]], errors="ignore")

    # Engineer features
    df = engineer_features(df)

    # Split features and target
    target = config["data"]["target"]
    X = df.drop(columns=[target])
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["project"]["random_seed"],
        stratify=y
    )

    # Build and fit preprocessor
    preprocessor = build_preprocessor(config)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Apply SMOTE if configured and requested
    if apply_smote and config["preprocessing"]["handle_imbalance"] == "smote":
        smote = SMOTE(random_state=config["project"]["random_seed"])
        X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
        print(f"SMOTE applied. Training set: {X_train_processed.shape[0]} samples")

    print(f"Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


if __name__ == "__main__":
    config = load_config()
    summary = inspect_data(load_data(config))
    for k, v in summary.items():
        print(f"{k}: {v}")
