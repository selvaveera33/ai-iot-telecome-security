from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "project": "AI-Powered IoT Telecom Security",
        "message": "5G Network Threat Detection API is running"
    })

@app.route("/healthz")
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(silent=True) or {}

    if not data:
        return jsonify({
            "error": "Please provide network traffic data"
        }), 400

    return jsonify({
        "message": "Network traffic received successfully",
        "received_data": data
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
