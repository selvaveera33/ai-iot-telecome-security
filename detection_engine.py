"""Validated interface for the trained defensive traffic review models."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from data_generator import FEATURES

ROOT = Path(__file__).resolve().parent
RF_PATH = ROOT / "models" / "random_forest.joblib"
IF_PATH = ROOT / "models" / "isolation_forest.joblib"


def validate_traffic_record(traffic_record):
    if not isinstance(traffic_record, dict):
        raise TypeError("traffic_record must be a dictionary of numeric features.")
    missing = [name for name in FEATURES if name not in traffic_record]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    values = []
    for name in FEATURES:
        try:
            value = float(traffic_record[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{name}' must be numeric.") from exc
        if not np.isfinite(value):
            raise ValueError(f"Feature '{name}' must be finite (not NaN or infinite).")
        if value < 0:
            raise ValueError(f"Feature '{name}' cannot be negative.")
        values.append(value)
    return np.asarray([values], dtype=float)


def analyze_traffic(traffic_record):
    values = validate_traffic_record(traffic_record)
    missing_models = [str(path) for path in (RF_PATH, IF_PATH) if not path.exists()]
    if missing_models:
        raise FileNotFoundError("Trained model file(s) missing. Run 'python train_models.py' first.")
    rf, iso = joblib.load(RF_PATH), joblib.load(IF_PATH)
    row = pd.DataFrame(values, columns=FEATURES)
    rf_class = int(rf.predict(row)[0])
    confidence = float(np.max(rf.predict_proba(row)[0]))
    anomaly = int(iso.predict(row)[0]) == -1
    rf_flag = rf_class == 1
    status, reason = combine_predictions(rf_flag, anomaly)
    return {
        "random_forest_prediction": "SUSPICIOUS" if rf_flag else "NORMAL",
        "random_forest_confidence": confidence,
        "isolation_forest_prediction": "ANOMALY" if anomaly else "NORMAL",
        "combined_status": status,
        "alert_triggered": bool(rf_flag or anomaly),
        "alert_reason": reason,
    }


def combine_predictions(random_forest_flag: bool, isolation_forest_flag: bool):
    """Apply review-only policy to model flags; never block network traffic."""
    if random_forest_flag and isolation_forest_flag:
        return "HIGH_PRIORITY_REVIEW", "Both models flagged the record; review promptly."
    if random_forest_flag or isolation_forest_flag:
        return "REVIEW", "One model flagged the record; review the traffic."
    return "NORMAL", "Neither model flagged the record."
