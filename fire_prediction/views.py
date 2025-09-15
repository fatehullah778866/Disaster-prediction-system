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

# Dropdown canonical values (must match dataset encoding)
MONTHS = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
DAYS   = ["mon","tue","wed","thu","fri","sat","sun"]

def _load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        raise RuntimeError("Model or scaler artifacts not found. Train first (run train_model.py).")
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(SCALER_PATH)  # dict with scaler, ohe, feature names
    return model, meta

def predict_view(request):
    if request.method == "GET":
        context = {"months": MONTHS, "days": DAYS}
        return render(request, "fire_prediction/predict.html", context)

    if request.method == "POST":
        # Validate and collect form inputs
        try:
            payload = {
                "X": float(request.POST.get("X")),
                "Y": float(request.POST.get("Y")),
                "FFMC": float(request.POST.get("FFMC")),
                "DMC": float(request.POST.get("DMC")),
                "DC": float(request.POST.get("DC")),
                "ISI": float(request.POST.get("ISI")),
                "temp": float(request.POST.get("temp")),
                "RH": float(request.POST.get("RH")),
                "wind": float(request.POST.get("wind")),
                "rain": float(request.POST.get("rain")),
                "month": request.POST.get("month", "").strip().lower(),
                "day": request.POST.get("day", "").strip().lower(),
            }
        except (TypeError, ValueError):
            return HttpResponseBadRequest("Invalid input types.")

        if payload["month"] not in MONTHS or payload["day"] not in DAYS:
            return HttpResponseBadRequest("Invalid month/day.")

        # Load artifacts
        model, meta = _load_artifacts()

        # Transform to model-ready features
        # Reuse function from training logic replicated inline:
        num_vals = [payload[k] for k in meta["num_features"]]
        cat_vals = [payload[k] for k in meta["cat_features"]]

        # Scale numeric
        X_num_scaled = meta["scaler"].transform(np.array(num_vals).reshape(1, -1))
        # OHE categorical
        X_cat_ohe = meta["ohe"].transform(np.array(cat_vals).reshape(1, -1))

        # Combine sparse/numpy
        from scipy.sparse import hstack, csr_matrix
        X_all = hstack([csr_matrix(X_num_scaled), X_cat_ohe])

        # Predict (class + probability)
        y_hat = model.predict(X_all)[0]
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_all)[0][1])
        else:
            # Fallback: decision_function to pseudo-prob
            if hasattr(model, "decision_function"):
                from scipy.special import expit
                proba = float(expit(model.decision_function(X_all))[0])
            else:
                proba = None

        result = {
            "prediction": int(y_hat),
            "probability": f"{proba*100:.2f}%" if proba is not None else "N/A",
        }

        return render(request, "fire_prediction/result.html", {
            "result": result,
            "inputs": payload,
        })
