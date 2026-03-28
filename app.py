"""
DisastraSense – AI-Based Disaster Prediction and Response System.
Home page -> Get started -> App (Flood, Earthquake, Hurricane).
"""
import logging
import os

logging.getLogger("werkzeug").setLevel(logging.ERROR)

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# ----- Models (lazy load to avoid startup errors if .pkl missing) -----
_flood_model = _flood_features = _eq_model = _hur_model = _hur_features = None


def _load_models():
    global _flood_model, _flood_features, _eq_model, _hur_model, _hur_features
    if _flood_model is None:
        _flood_model = joblib.load(os.path.join(BASE, "random_forest_model.pkl"))
        _flood_features = joblib.load(os.path.join(BASE, "model_features.pkl"))
    if _eq_model is None and os.path.exists(os.path.join(BASE, "earthquake_model.pkl")):
        _eq_model = joblib.load(os.path.join(BASE, "earthquake_model.pkl"))
    if _hur_model is None and os.path.exists(os.path.join(BASE, "hurricane_model.pkl")):
        _hur_model = joblib.load(os.path.join(BASE, "hurricane_model.pkl"))
        _hur_features = joblib.load(os.path.join(BASE, "hurricane_features.pkl"))


# ----- Routes -----
@app.route("/")
def index():
    return render_template("home.html")


@app.route("/app")
def app_page():
    return render_template("main.html")


@app.route("/api/predict/flood", methods=["POST"])
def predict_flood():
    try:
        _load_models()
        data = {k: request.form.get(k) for k in request.form}
        for key in data:
            try:
                data[key] = float(data[key])
            except (TypeError, ValueError):
                pass
        df = pd.DataFrame([data])
        df = pd.get_dummies(df, drop_first=True)
        df = df.reindex(columns=_flood_features, fill_value=0)
        pred = _flood_model.predict(df)[0]
        return jsonify({"prediction": "Yes" if pred == 1 else "No"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict/earthquake", methods=["POST"])
def predict_earthquake():
    try:
        _load_models()
        if _eq_model is None:
            return jsonify({"error": "Earthquake model not found. Run: python create_all_models.py"}), 400
        lat = float(request.form.get("Latitude"))
        lon = float(request.form.get("Longitude"))
        mag = _eq_model.predict(pd.DataFrame([{"Latitude": lat, "Longitude": lon}]))[0]
        return jsonify({"magnitude": float(mag)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict/hurricane", methods=["POST"])
def predict_hurricane():
    try:
        _load_models()
        if _hur_model is None:
            return jsonify({"error": "Hurricane model not found. Run: python create_all_models.py"}), 400
        row = {
            "Latitude": float(request.form.get("Latitude")),
            "Longitude": float(request.form.get("Longitude")),
            "Moderate Wind NE": float(request.form.get("Moderate Wind NE")),
            "Moderate Wind SE": float(request.form.get("Moderate Wind SE")),
            "Moderate Wind SW": float(request.form.get("Moderate Wind SW")),
            "Moderate Wind NW": float(request.form.get("Moderate Wind NW")),
            "Year": int(float(request.form.get("Year"))),
            "Month": int(float(request.form.get("Month"))),
            "Day": int(float(request.form.get("Day"))),
        }
        df = pd.DataFrame([row])
        df = df.reindex(columns=_hur_features, fill_value=0)
        wind = _hur_model.predict(df)[0]
        return jsonify({"max_wind": float(wind)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = 5000
    url = f"http://127.0.0.1:{port}"
    print("\n  DisastraSense running. Open in browser:\n")
    print(f"  {url}\n")
    app.run(debug=True, host="127.0.0.1", port=port)
