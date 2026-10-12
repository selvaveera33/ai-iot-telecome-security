"""Generate reproducible simulated (not real-network) telecom traffic data."""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["packets", "bytes_transferred", "duration_sec", "failed_auth_attempts", "connections_per_min", "traffic_rate"]
LABEL = "label"
SEED = 42
DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "traffic_dataset.csv"


def generate_dataset(n_records: int = 3000, random_state: int = SEED) -> pd.DataFrame:
    """Create overlapping normal and suspicious feature distributions."""
    if n_records < 2:
        raise ValueError("n_records must be at least 2.")
    rng = np.random.default_rng(random_state)
    labels = rng.choice([0, 1], size=n_records, p=[0.72, 0.28])
    n = labels == 0
    # Both classes overlap intentionally to avoid a perfectly separable toy set.
    packets = np.where(n, rng.lognormal(4.1, .75, n_records), rng.lognormal(4.55, .9, n_records))
    bytes_ = packets * np.where(n, rng.lognormal(6.0, .7, n_records), rng.lognormal(6.3, .85, n_records))
    duration = np.where(n, rng.gamma(2.4, 18, n_records), rng.gamma(1.9, 20, n_records))
    failed = np.where(n, rng.poisson(.18, n_records), rng.poisson(1.7, n_records))
    # Add benign auth failures and suspicious low-failure cases to preserve ambiguity.
    failed = np.clip(failed + rng.binomial(1, .025, n_records), 0, None)
    conns = np.where(n, rng.lognormal(2.5, .7, n_records), rng.lognormal(3.0, .8, n_records))
    rate = bytes_ / np.maximum(duration, .5)
    rate *= rng.lognormal(0, .35, n_records)
    frame = pd.DataFrame({
        "packets": np.maximum(1, np.rint(packets)).astype(int),
        "bytes_transferred": np.maximum(1, np.rint(bytes_)).astype(int),
        "duration_sec": np.round(duration, 3),
        "failed_auth_attempts": failed.astype(int),
        "connections_per_min": np.round(conns, 3),
        "traffic_rate": np.round(rate, 3),
        "label": labels.astype(int),
    })
    return frame.sample(frac=1, random_state=random_state).reset_index(drop=True)


def save_dataset(path: Path = DEFAULT_PATH, n_records: int = 3000, random_state: int = SEED) -> pd.DataFrame:
    frame = generate_dataset(n_records, random_state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


if __name__ == "__main__":
    df = save_dataset()
    print(f"Saved {len(df)} simulated records to {DEFAULT_PATH}")
