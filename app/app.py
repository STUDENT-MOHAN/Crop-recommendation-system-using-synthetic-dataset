from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.secret_key = "super-secret-key"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Strict ranges for validation (update as needed)
RANGES = {
    "N": (0, 140),
    "P": (5, 145),
    "K": (5, 205),
    "temperature": (8.0, 43.0),
    "humidity": (20.0, 100.0),
    "ph": (3.5, 9.5),
    "rainfall": (20.0, 300.0),
}

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def paths():
    return {
        "real_model": os.path.join(MODELS_DIR, "model_real.pkl"),
        "synth_model": os.path.join(MODELS_DIR, "model_synth.pkl"),
        "label_encoder": os.path.join(MODELS_DIR, "label_encoder.pkl"),
        "test_split": os.path.join(DATA_DIR, "test.csv"),
        "train_split": os.path.join(DATA_DIR, "train.csv"),
        "synthetic_csv": os.path.join(DATA_DIR, "synthetic_data.csv"),
        "real_csv": os.path.join(DATA_DIR, "crop_recommendation.csv"),
    }

def _load_models():
    p = paths()
    models = {}
    if os.path.exists(p["real_model"]):
        models["real"] = joblib.load(p["real_model"])
    if os.path.exists(p["synth_model"]):
        models["synth"] = joblib.load(p["synth_model"])
    le = joblib.load(p["label_encoder"]) if os.path.exists(p["label_encoder"]) else None
    return models, le

def _validate_inputs(form):
    values = {}
    for feat in FEATURES:
        if feat not in form:
            return None, f"Missing field: {feat}"
        try:
            v = float(form[feat])
        except ValueError:
            return None, f"Invalid number for {feat}"
        lo, hi = RANGES[feat]
        if not (lo <= v <= hi):
            return None, f"{feat} must be between {lo} and {hi}"
        values[feat] = v
    return values, None

@app.route("/")
def index():
    return render_template("index.html", ranges=RANGES, features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    values, err = _validate_inputs(request.form)
    if err:
        flash(err, "error")
        return redirect(url_for("index"))
    X = pd.DataFrame([values])[FEATURES]

    models, le = _load_models()
    outputs = {}

    for name in ["real", "synth"]:
        model = models.get(name)
        if model is None:
            outputs[name] = {"label": "(model missing)", "proba": 0.0}
            continue
        y_pred = model.predict(X)[0]
        if hasattr(model, "predict_proba"):
            proba = float(np.max(model.predict_proba(X)))
        else:
            proba = 0.0
        # decode if label encoder present
        label = y_pred
        if le is not None and isinstance(y_pred, (int, np.integer)):
            try:
                label = le.inverse_transform([int(y_pred)])[0]
            except Exception:
                pass
        outputs[name] = {"label": str(label), "proba": proba}

    return render_template("result.html", values=values, outputs=outputs, ranges=RANGES)

@app.route("/compare")
def compare():
    return render_template("compare.html")

@app.route("/metrics")
def metrics():
    p = paths()
    models, le = _load_models()

    if not os.path.exists(p["test_split"]):
        return jsonify({"error": "No test split. Run training scripts first."}), 400

    test = pd.read_csv(p["test_split"])
    X_test = test[FEATURES].copy()
    y_true = test["label"].copy()

    # If labels were encoded during training, ensure consistency:
    if le is not None and y_true.dtype != np.int64 and y_true.dtype != np.int32:
        y_true_enc = le.transform(y_true)
    else:
        y_true_enc = y_true

    results = {}
    for name in ["real", "synth"]:
        model = models.get(name)
        if model is None:
            results[name] = {"accuracy": None, "f1": None}
            continue
        y_pred = model.predict(X_test)
        # For robust comparison, try to map preds to ints if needed
        if le is not None and not np.issubdtype(y_pred.dtype, np.integer):
            # Try encoding predicted labels
            try:
                y_pred_enc = le.transform(y_pred)
            except Exception:
                y_pred_enc = y_pred
        else:
            y_pred_enc = y_pred
        acc = float(accuracy_score(y_true_enc, y_pred_enc))
        f1 = float(f1_score(y_true_enc, y_pred_enc, average="weighted"))
        results[name] = {"accuracy": acc, "f1": f1}

    payload = {"results": results}
    return jsonify(payload)

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
