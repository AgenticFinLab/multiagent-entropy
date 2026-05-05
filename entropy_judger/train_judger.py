"""
Train and serialize the Entropy Judger from existing merged_datasets.csv.

The judger is an ensemble of XGBoost + LightGBM classifiers trained on
all available single-run data. Once trained it is frozen and applied to
new K=20 repeated inference runs without retraining.

Usage:
    python entropy_judger/train_judger.py \
        --data-path data_mining/data/merged_datasets.csv \
        --output-dir entropy_judger/models/
"""

import argparse
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Add data_mining to path so we can reuse its utilities
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "data_mining" / "code"))

from base.model_factory import ModelFactory
from features import DEFAULT_EXCLUDE_COLUMNS


# Columns that identify a row but should not be features
_ID_COLS = ["sample_id"]
# Categorical columns we one-hot encode (dataset, model, architecture drive
# the judger's prior; encoding them lets the model learn per-slice calibration)
_CAT_COLS = ["dataset", "model_name", "architecture"]


def load_and_encode(data_path: str) -> tuple[pd.DataFrame, list[str]]:
    """Load CSV, one-hot encode categoricals, return (df_encoded, feature_cols)."""
    df = pd.read_csv(data_path)

    # One-hot encode the three categorical columns (drop_first avoids multicollinearity)
    df = pd.get_dummies(df, columns=_CAT_COLS, drop_first=False)

    # Build exclude set: DEFAULT_EXCLUDE_COLUMNS + id cols + target
    exclude = set(DEFAULT_EXCLUDE_COLUMNS) | set(_ID_COLS) | {"is_finally_correct"}
    # Also drop any string columns that slipped through (shouldn't be any after OHE)
    numeric_cols = df.select_dtypes(include=[np.number, bool]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    # Fill missing values with column median (same as data_mining pipeline)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    return df, feature_cols


def train(data_path: str, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path} ...")
    df, feature_cols = load_and_encode(data_path)

    X = df[feature_cols].values
    y = df["is_finally_correct"].astype(int).values
    print(f"  {len(df)} samples, {len(feature_cols)} features")
    print(f"  Positive rate: {y.mean():.3f}")

    # Save feature column names so evaluate.py can replicate the same encoding
    feat_path = output_path / "feature_columns.pkl"
    with open(feat_path, "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"Saved feature columns → {feat_path}")

    for name in ("XGBoost", "LightGBM"):
        clf = ModelFactory.classifier(name)
        if clf is None:
            print(f"  {name} not available, skipping")
            continue
        print(f"Training {name} ...")
        clf.fit(X, y)
        model_path = output_path / f"{name.lower()}_judger.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        # Quick sanity check on training data itself
        prob = clf.predict_proba(X)[:, 1]
        print(f"  Mean predicted prob: {prob.mean():.3f}  (min {prob.min():.3f}, max {prob.max():.3f})")
        print(f"  Saved → {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Entropy Judger")
    parser.add_argument(
        "--data-path",
        default="data_mining/data/merged_datasets.csv",
        help="Path to merged_datasets.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="entropy_judger/models/",
        help="Directory to save model pkl files",
    )
    args = parser.parse_args()
    train(args.data_path, args.output_dir)


if __name__ == "__main__":
    main()
