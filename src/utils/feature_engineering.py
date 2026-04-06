"""
Feature engineering module.
Centralized feature creation logic for reuse across notebooks and API.
"""

import pandas as pd
import numpy as np


# Registry of all engineered features
ENGINEERED_FEATURES = [
    "login_fail_ratio",
    "packet_rate",
    "risk_score",
    "high_risk_ip",
]


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps. Used in both training and inference."""
    df = df.copy()

    df["login_fail_ratio"] = np.where(
        df["login_attempts"] > 0,
        df["failed_logins"] / df["login_attempts"],
        0
    )

    df["packet_rate"] = np.where(
        df["session_duration"] > 0,
        df["network_packet_size"] / df["session_duration"],
        0
    )

    df["risk_score"] = (
        df["ip_reputation_score"] * 0.4
        + df["unusual_time_access"] * 0.3
        + df["login_fail_ratio"] * 0.3
    )

    df["high_risk_ip"] = (df["ip_reputation_score"] > 0.7).astype(int)

    return df
