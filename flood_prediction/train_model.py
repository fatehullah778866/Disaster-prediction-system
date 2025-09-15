# train_model.py
"""
Training script for Flood Prediction (binary HighFloodRisk).
Saves:
  - ml_model/model.pkl   -> trained classifier
  - ml_model/scaler.pkl  -> dict: scaler, ohe, feature lists
Usage:
  python train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from pprint import pprint

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from scipy.sparse import hstack, csr_matrix

# ---------- Config ----------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATAFILE = "flood_with_pakistan_features.csv"
DATA_PATH = os.path.join(APP_DIR, DATAFILE)
MODEL_DIR = os.path.join(APP_DIR, "ml_model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

RISK_THRESHOLD = 0.5  # threshold on FloodProbability to make binary label

# ---------- Helpers ----------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_dataframe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")
    df = pd.read_csv(path)
    return df

def split_features_target(df):
    if "FloodProbability" not in df.columns:
        raise ValueError("Dataset must contain 'FloodProbability' column.")
    # Build binary label: HighFloodRisk if FloodProbability >= threshold
    y = (df["FloodProbability"] >= RISK_THRESHOLD).astype(int)
    # Drop target and keep features
    X = df.drop(columns=["FloodProbability"])
    return X, y

def detect_feature_types(X: pd.DataFrame):
    # Heuristic: treat integer columns with small unique counts as categorical
    numeric_cols = []
    categorical_cols = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            nunique = X[c].nunique(dropna=True)
            # If the column seems ordinal/categorical (small number unique), treat as categorical
            if nunique <= 12 and (X[c].dtype == np.int64 or nunique <= 8):
                categorical_cols.append(c)
            else:
                numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols

def build_preprocessors(X, numeric_cols, categorical_cols):
    # StandardScaler for numeric
    scaler = StandardScaler()
    X_num = X[numeric_cols].astype(float).values
    scaler.fit(X_num)

    # OneHotEncoder for categorical. Use wrapper to handle sklearn version differences.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    ohe.fit(X[categorical_cols].astype(str).values)

    return scaler, ohe

def transform_features(scaler, ohe, X, numeric_cols, categorical_cols):
    X_num = scaler.transform(X[numeric_cols].astype(float).values)
    X_cat = ohe.transform(X[categorical_cols].astype(str).values)
    X_all = hstack([csr_matrix(X_num), X_cat])
    return X_all

def main():
    print("Loading dataframe...")
    df = load_dataframe(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print("Determining features and target...")
    X, y = split_features_target(df)
    numeric_cols, categorical_cols = detect_feature_types(X)
    print("Numeric features detected:", numeric_cols)
    print("Categorical features detected:", categorical_cols)

    print("Fitting preprocessors...")
    scaler, ohe = build_preprocessors(X, numeric_cols, categorical_cols)

    print("Transforming full dataset...")
    X_all = transform_features(scaler, ohe, X, numeric_cols, categorical_cols)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Training GradientBoostingClassifier...")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating on hold-out test set...")
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    if probs is not None:
        try:
            auc = roc_auc_score(y_test, probs)
            print(f"ROC AUC: {auc:.4f}")
        except Exception:
            pass

    print("Cross-validating (StratifiedKFold=5)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_all, y, cv=skf, scoring="accuracy")
    print(f"CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Persist artifacts
    print("Saving artifacts...")
    ensure_dir(MODEL_DIR)
    joblib.dump(model, MODEL_PATH)
    meta = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "scaler": scaler,
        "ohe": ohe,
        "risk_threshold": RISK_THRESHOLD,
        "notes": "Binary label: FloodProbability >= threshold => HighFloodRisk"
    }
    joblib.dump(meta, SCALER_PATH)

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved preprocess meta -> {SCALER_PATH}")
    print("Done.")

if __name__ == "__main__":
    main()
