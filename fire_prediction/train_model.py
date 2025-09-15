# train_model.py
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from scipy.sparse import hstack, csr_matrix

# --------- Config ---------
APP_DIR = os.path.dirname(os.path.abspath(__file__))  # this file lives in your Django app
DATASET_FILENAME = "forestfires.csv"                  # your dataset in the app directory
DATA_PATH = os.path.join(APP_DIR, DATASET_FILENAME)
MODEL_DIR = os.path.join(APP_DIR, "ml_model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

NUM_FEATURES = ['X','Y','FFMC','DMC','DC','ISI','temp','RH','wind','rain']
CAT_FEATURES = ['month','day']  # will be one-hot encoded

# --------- Helpers ---------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    # Basic sanity checks
    required_cols = set(NUM_FEATURES + CAT_FEATURES + ['area'])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df

def build_target(df: pd.DataFrame) -> pd.Series:
    # Binary classification: any burned area (>0) is considered a "fire event"
    return (df['area'] > 0).astype(int)

def preprocess_fit_transform(df: pd.DataFrame):
    X = df[NUM_FEATURES + CAT_FEATURES].copy()
    y = build_target(df)

    # Numeric
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X[NUM_FEATURES].values)

    # Categorical
    try:
    # For sklearn >= 1.2
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError:
    # For sklearn < 1.2
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)

    X_cat_ohe = ohe.fit_transform(X[CAT_FEATURES].values)

    # Combine
    X_all = hstack([csr_matrix(X_num_scaled), X_cat_ohe])

    meta = {
        "scaler": scaler,
        "ohe": ohe,
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "ohe_categories_": [list(c) for c in ohe.categories_],  # for reference/debugging
    }
    return X_all, y, meta

def preprocess_transform_single(meta, payload: dict):
    """
    payload: dict from user input with keys matching NUM_FEATURES + CAT_FEATURES
    Returns a 1xN sparse matrix ready for model.predict / predict_proba
    """
    # Ensure ordering
    num_vals = [float(payload[k]) for k in meta["num_features"]]
    cat_vals = [payload[k] for k in meta["cat_features"]]

    X_num_scaled = meta["scaler"].transform(np.array(num_vals).reshape(1, -1))
    X_cat_ohe = meta["ohe"].transform(np.array(cat_vals).reshape(1, -1))

    X_all = hstack([csr_matrix(X_num_scaled), X_cat_ohe])
    return X_all

def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)

    print("Fitting preprocessors and transforming features...")
    X_all, y, meta = preprocess_fit_transform(df)

    # Train/test split for honest evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Training model...")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nHold-out Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation for robustness
    print("\nCross-validating (StratifiedKFold=5)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_all, y, cv=skf, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Persist artifacts
    ensure_dir(MODEL_DIR)
    joblib.dump(model, MODEL_PATH)
    # Save preprocessors + feature meta
    joblib.dump(meta, SCALER_PATH)

    print(f"\nArtifacts saved:")
    print(f"  Model     → {MODEL_PATH}")
    print(f"  Scaler/OHE meta → {SCALER_PATH}")
    print("\nDone.")

if __name__ == "__main__":
    main()
