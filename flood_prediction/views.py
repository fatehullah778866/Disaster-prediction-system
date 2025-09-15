# views.py
import os
import joblib
import numpy as np

from django.shortcuts import render
from django.http import HttpResponseBadRequest

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "ml_model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        raise RuntimeError("Model artifacts missing. Run train_model.py first.")
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(SCALER_PATH)
    return model, meta

def predict_view(request):
    model, meta = load_artifacts()
    numeric_cols = meta["numeric_cols"]
    categorical_cols = meta["categorical_cols"]

    if request.method == "GET":
        # Build a simple dynamic form context (fields in order)
        fields = numeric_cols + categorical_cols
        # Provide some small hints/defaults for better UX (server-side)
        hints = {c: "" for c in fields}
        # If Region is present, create a sample choices list from meta.ohe.categories_
        region_options = []
        try:
            # map categorical names to their ohe categories
            for i, col in enumerate(categorical_cols):
                cats = list(meta["ohe"].categories_[i])
                if col.lower() in ("region", "province", "district"):
                    region_options = cats
        except Exception:
            region_options = []
        return render(request, "flood_prediction/predict.html", {
            "numeric_fields": numeric_cols,
            "categorical_fields": categorical_cols,
            "region_options": region_options,
        })

    if request.method == "POST":
        # Collect inputs
        payload = {}
        try:
            # numeric
            for col in numeric_cols:
                raw = request.POST.get(col)
                if raw is None or raw.strip() == "":
                    return HttpResponseBadRequest(f"Missing numeric input: {col}")
                payload[col] = float(raw)

            # categorical
            for col in categorical_cols:
                raw = request.POST.get(col)
                if raw is None or raw.strip() == "":
                    # allow empty but convert to "unknown"
                    payload[col] = "unknown"
                else:
                    payload[col] = raw.strip()
        except ValueError:
            return HttpResponseBadRequest("Invalid numeric input.")

        # Prepare feature vector consistent with training
        X_num = np.array([payload[c] for c in numeric_cols]).reshape(1, -1)
        X_cat = np.array([[str(payload[c]) for c in categorical_cols]])
        # apply transforms from meta
        X_num_scaled = meta["scaler"].transform(X_num)
        X_cat_ohe = meta["ohe"].transform(X_cat)
        from scipy.sparse import hstack, csr_matrix
        X_all = hstack([csr_matrix(X_num_scaled), X_cat_ohe])

        # Predict
        y_hat = int(model.predict(X_all)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_all)[0][1])
        else:
            # fallback
            try:
                from scipy.special import expit
                proba = float(expit(model.decision_function(X_all))[0])
            except Exception:
                proba = None

        # Build result
        result = {
            "prediction": y_hat,  # 1 => HighFloodRisk, 0 => Low
            "probability": f"{proba*100:.2f}%" if proba is not None else "N/A",
        }

        return render(request, "flood_prediction/result.html", {
            "result": result,
            "inputs": payload,
        })
