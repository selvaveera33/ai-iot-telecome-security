# AI-Powered 5G Network Telecom Security System

A defensive, local simulation demonstrating supervised classification and unsupervised anomaly detection. **All generated traffic is synthetic; it was not collected from a real 5G network. This is not a production-ready telecom security system.** Alerts request human review and never automatically block traffic.

## Requirements and setup (Windows)

Use Python 3.10 or newer. From this directory:

```powershell
python -m pip install -r requirements.txt
python train_models.py
python -m unittest discover -s tests -v
```

Training generates `data/traffic_dataset.csv` if absent, fits both models and saves them in `models/`. You can also generate the reproducible 3,000-record data file directly with `python data_generator.py`.

## Models

The Random Forest learns NORMAL (0) versus SUSPICIOUS (1) from labeled samples. Training uses a stratified train/test split; only the training partition is used for fitting. The training report measures accuracy, precision, recall, F1, confusion matrix, feature importance and predicted test labels.

The Isolation Forest learns the shape of normal traffic using only normal rows from the training partition. It returns ANOMALY when a sample falls outside patterns it learned. Its configurable `contamination` parameter (default 0.08 in `train_models.py`) controls the expected outlier fraction during fitting. An anomaly is not automatically proof of an attack; benign unusual traffic can be anomalous.

The combined engine requests review if either model flags a record, gives a high-priority review status if both flag it, and reports NORMAL if neither flags it. This does not block network traffic.

## Python demonstration

After training, create `demo.py` with this example or run it in a Python prompt:

```python
from detection_engine import analyze_traffic

samples = [
    {"packets": 80, "bytes_transferred": 6000, "duration_sec": 40,
     "failed_auth_attempts": 0, "connections_per_min": 5, "traffic_rate": 140},
    {"packets": 900, "bytes_transferred": 900000, "duration_sec": 3,
     "failed_auth_attempts": 12, "connections_per_min": 250, "traffic_rate": 250000},
    {"packets": 25, "bytes_transferred": 1800, "duration_sec": 75,
     "failed_auth_attempts": 1, "connections_per_min": 2, "traffic_rate": 30},
]
for sample in samples:
    print(analyze_traffic(sample))
```

## Local web page

Train the models, then start the Python-only local page:

```powershell
python web_app.py
```

Visit `http://127.0.0.1:8000` and submit the six numeric features for review. Stop the server with Ctrl+C. The page runs on localhost and uses Python's built-in HTTP server.

## Input fields

`packets`, `bytes_transferred`, `duration_sec`, `failed_auth_attempts`, `connections_per_min`, and `traffic_rate` must be finite, non-negative numbers. Missing fields are rejected. Confidence is the highest Random Forest class probability; it is not a guarantee of correctness.
