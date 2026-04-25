"""Common data loading helpers (SHAP outputs, accuracy CSV, summary JSON)."""

import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def load_csv(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def load_json(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_shap(
    results_dir: Path | str,
    exp_key: str,
    model: str = "LightGBM",
    task: str = "classification",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load (shap_values, X_test, prediction_probabilities) for one experiment.

    Returns (None, None, None) if the experiment directory is missing.
    Files individually missing are returned as None.
    """
    base = Path(results_dir) / exp_key / "shap"
    if not base.exists():
        return None, None, None

    shap_path = base / f"shap_values_{model}_{task}.csv"
    x_path = base / f"X_test_{model}_{task}.csv"
    pred_path = base / f"shap_prediction_probabilities_{model}_{task}.csv"

    shap_df = pd.read_csv(shap_path, index_col=0) if shap_path.exists() else None
    x_df = pd.read_csv(x_path, index_col=0) if x_path.exists() else None
    pred_df = pd.read_csv(pred_path) if pred_path.exists() else None
    return shap_df, x_df, pred_df
