"""
DisastraSense – AI-Based Disaster Prediction and Response System.
Home page -> Get started -> App (Flood, Earthquake, Hurricane, Chatbase Chatbot, YOLO).
"""
import base64
import json
import logging
import os
import tempfile
import urllib.request
import urllib.error

# Load .env if present (for CHATBASE_API_KEY, CHATBASE_CHATBOT_ID)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.getLogger("werkzeug").setLevel(logging.ERROR)

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# Chatbase: set CHATBASE_API_KEY and CHATBASE_CHATBOT_ID in env (or .env) to use real API
CHATBASE_API_KEY = os.environ.get("CHATBASE_API_KEY", "").strip()
CHATBASE_CHATBOT_ID = os.environ.get("CHATBASE_CHATBOT_ID", "").strip()

# ----- Models (lazy load to avoid startup errors if .pkl missing) -----
_flood_model = _flood_features = _eq_model = _hur_model = _hur_features = None
_yolo_model = None


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


@app.route("/api/chat", methods=["POST"])
def chat():
    """Chatbase when API key + chatbot ID are set; otherwise stub replies."""
    try:
        body = request.get_json() or {}
        msg = (body.get("message") or "").strip()
        if not msg:
            return jsonify({"reply": "Please type a question."})
        if CHATBASE_API_KEY and CHATBASE_CHATBOT_ID:
            req = urllib.request.Request(
                "https://www.chatbase.co/api/v1/chat",
                data=json.dumps({
                    "messages": [{"role": "user", "content": msg}],
                    "chatbotId": CHATBASE_CHATBOT_ID,
                }).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {CHATBASE_API_KEY}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode())
                    reply = (data.get("text") or data.get("output") or data.get("message") or "")
                    return jsonify({"reply": reply or "No response from Chatbase."})
            except urllib.error.HTTPError as e:
                return jsonify({"reply": f"Chatbase error: {e.code}. Check API key and chatbot ID."})
        # Stub replies when no API key
        m = msg.lower()
        if "flood" in m:
            reply = "Flood safety: move to higher ground, avoid driving through water, follow local alerts."
        elif "earthquake" in m:
            reply = "During an earthquake: drop, cover, hold on. Stay away from windows and heavy objects."
        elif "hurricane" in m:
            reply = "Hurricane prep: secure outdoor items, have an evacuation plan, stock water and supplies."
        else:
            reply = "This is demo mode. Set CHATBASE_API_KEY and CHATBASE_CHATBOT_ID to use the real Chatbase API (see README)."
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _load_yolo():
    """Load YOLOv5 model (person/victim detection) via PyTorch Hub."""
    global _yolo_model
    if _yolo_model is None:
        import torch
        _yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=False, trust_repo=True)
        _yolo_model.conf = 0.35
        _yolo_model.iou = 0.45
        _yolo_model.classes = [0]  # 0 = person (COCO)
    return _yolo_model


@app.route("/api/yolo", methods=["POST"])
def yolo_detect():
    """Run YOLOv5 person (victim) detection on uploaded image. Returns count and annotated image."""
    if "image" not in request.files and "file" not in request.files:
        return jsonify({"error": "No image uploaded. Use field 'image' or 'file'."}), 400
    file = request.files.get("image") or request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    allowed = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if not file.filename.lower().endswith(allowed):
        return jsonify({"error": "Only image files (jpg, png, bmp, webp) are supported."}), 400
    try:
        _load_yolo()
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            path = tmp.name
        try:
            results = _yolo_model(path)
            df = results.pandas().xyxy[0] if len(results.pandas().xyxy) else None
            count = int((df["class"] == 0).sum()) if df is not None and "class" in df.columns else 0
            out_dir = tempfile.mkdtemp()
            results.save(save_dir=out_dir)
            image_base64 = None
            for name in os.listdir(out_dir):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    with open(os.path.join(out_dir, name), "rb") as f:
                        image_base64 = base64.b64encode(f.read()).decode()
                    break
            for name in os.listdir(out_dir):
                try:
                    os.unlink(os.path.join(out_dir, name))
                except Exception:
                    pass
            try:
                os.rmdir(out_dir)
            except Exception:
                pass
            return jsonify({
                "count": count,
                "message": f"Detected {count} person(s)." if count != 1 else "Detected 1 person.",
                "image_base64": image_base64,
            })
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
    except Exception as e:
        return jsonify({"error": str(e), "count": 0}), 400


if __name__ == "__main__":
    port = 5000
    url = f"http://127.0.0.1:{port}"
    print("\n  DisastraSense running. Open in browser:\n")
    print(f"  {url}\n")
    app.run(debug=True, host="127.0.0.1", port=port)
