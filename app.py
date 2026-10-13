import os
import json
import sqlite3
from datetime import datetime

from flask import Flask, render_template, request
from detection_engine import analyze_traffic
from data_generator import FEATURES

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "security.db")


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                alert INTEGER NOT NULL,
                result TEXT NOT NULL
            )
        """)


def save_result(result):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO analyses
               (created_at, status, alert, result)
               VALUES (?, ?, ?, ?)""",
            (
                datetime.now().isoformat(timespec="seconds"),
                result["combined_status"],
                int(result["alert_triggered"]),
                json.dumps(result)
            )
        )


def get_stats():
    with sqlite3.connect(DB_PATH) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM analyses"
        ).fetchone()[0]

        alerts = conn.execute(
            "SELECT COUNT(*) FROM analyses WHERE alert = 1"
        ).fetchone()[0]

        recent = conn.execute("""
            SELECT id, created_at, status, result
            FROM analyses ORDER BY id DESC LIMIT 10
        """).fetchall()

    return total, alerts, recent


@app.route("/", methods=["GET", "POST"])
def dashboard():
    result = None
    error = None

    if request.method == "POST":
        try:
            record = {
                feature: request.form.get(feature, "")
                for feature in FEATURES
            }

            result = analyze_traffic(record)
            save_result(result)

        except (ValueError, TypeError, FileNotFoundError, KeyError) as exc:
            error = str(exc)

    total, alerts, recent = get_stats()

    return render_template(
        "index.html",
        features=FEATURES,
        result=result,
        error=error,
        total=total,
        alerts=alerts,
        recent=recent
    )


@app.route("/health")
def health():
    return {
        "message": "5G Network Threat Detection API is running",
        "project": "AI-Powered IoT Telecom Security"
    }


init_db()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        debug=False
    )
