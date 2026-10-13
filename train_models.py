"""Train and evaluate the supervised and unsupervised detection models."""
from pathlib import Path
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from data_generator import FEATURES, LABEL, DEFAULT_PATH, save_dataset

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("AI_MODEL_DIR", ROOT)).expanduser().resolve()
RF_PATH = MODEL_DIR / "random_forest.joblib"
IF_PATH = MODEL_DIR / "isolation_forest.joblib"
CONFIG_PATH = MODEL_DIR / "model_config.json"
SEED = 42


def load_dataset(path=DEFAULT_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run train_models.py to generate it.")
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"Could not read dataset CSV: {exc}") from exc
    required = FEATURES + [LABEL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains non-numeric, NaN, or infinite values.")
    if not set(df[LABEL].unique()).issubset({0, 1}) or df[LABEL].nunique() != 2:
        raise ValueError("label must contain both binary values 0 (NORMAL) and 1 (SUSPICIOUS).")
    return df


def train_models(dataset_path=DEFAULT_PATH, model_dir=MODEL_DIR, contamination=0.08):
    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be greater than 0 and at most 0.5.")
    path = Path(dataset_path)
    if not path.exists() and path == Path(DEFAULT_PATH):
        save_dataset(path)
    df = load_dataset(path)
    X, y = df[FEATURES], df[LABEL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=SEED, stratify=y)
    rf = RandomForestClassifier(n_estimators=240, min_samples_leaf=2,
                                class_weight="balanced_subsample", random_state=SEED,
                                n_jobs=1)
    rf.fit(X_train, y_train)
    # Fit only on normal training samples; test data is never used in fitting.
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=SEED, n_jobs=1)
    iso.fit(X_train.loc[y_train == 0, FEATURES])
    predictions = rf.predict(X_test)
    anomaly_predictions = (iso.predict(X_test) == -1).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "isolation_forest": {
            "precision": float(precision_score(y_test, anomaly_predictions, zero_division=0)),
            "recall": float(recall_score(y_test, anomaly_predictions, zero_division=0)),
            "f1": float(f1_score(y_test, anomaly_predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, anomaly_predictions, labels=[0, 1]).tolist(),
        },
    }
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    rf_path = model_dir / RF_PATH.name
    iso_path = model_dir / IF_PATH.name
    joblib.dump(rf, rf_path)
    joblib.dump(iso, iso_path)
    config = {"features": FEATURES, "label_mapping": {"0": "NORMAL", "1": "SUSPICIOUS"},
              "random_forest": {"n_estimators": 240, "min_samples_leaf": 2, "random_state": SEED},
              "isolation_forest": {"n_estimators": 200, "contamination": contamination, "random_state": SEED},
              "seed": SEED, "metrics": metrics}
    (model_dir / CONFIG_PATH.name).write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {"records": len(df), "training_records": len(X_train), "test_records": len(X_test),
              "random_forest": metrics, "feature_importance": dict(zip(FEATURES, map(float, rf.feature_importances_)))}
    print("Training report (measured on held-out stratified test set)")
    print(classification_report(y_test, predictions, target_names=["NORMAL", "SUSPICIOUS"], zero_division=0))
    print("Random Forest metrics:", json.dumps(metrics, indent=2))
    print("Feature importance:", json.dumps(report["feature_importance"], indent=2))
    print("Isolation Forest metrics:", json.dumps(metrics["isolation_forest"], indent=2))
    print(f"Saved Random Forest model: {rf_path}")
    print(f"Saved Isolation Forest model: {iso_path}")
    print(f"Saved model configuration: {model_dir / CONFIG_PATH.name}")
    return report


if __name__ == "__main__":
    train_models()
