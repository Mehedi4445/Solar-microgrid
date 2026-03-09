"""
train_model.py
--------------
Trains a Random Forest classifier on the solar microgrid dataset
and saves the model + label encoder to disk via joblib.

Run:
    python train_model.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = "solar_microgrid_ai_dataset.csv"
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
ENC_PATH   = os.path.join(MODEL_DIR, "label_encoder.pkl")

# ── Feature / target names ────────────────────────────────────────────────────
FEATURES = [
    "Solar_Irradiance",
    "Cloud_Cover",
    "Temperature",
    "Humidity",
    "Battery_Level",
    "Grid_Price",
    "Solar_Generation_kW",
    "Hospital_Load_kW",
    "Residential_Load_kW",
    "EV_Load_kW",
    "Emergency_Load_kW",
    "Total_Load_kW",
]
TARGET = "Distribution_Action"


def train():
    print("=" * 60)
    print("  Solar Microgrid AI — Model Training")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────
    print(f"\n[1/5] Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"      Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    print(f"      Class distribution:\n{df[TARGET].value_counts().to_string()}")

    # ── Encode target ──────────────────────────────────────────────
    print("\n[2/5] Encoding target labels …")
    le = LabelEncoder()
    y  = le.fit_transform(df[TARGET])
    X  = df[FEATURES].values
    print(f"      Classes: {list(le.classes_)}")

    # ── Train / test split ─────────────────────────────────────────
    print("\n[3/5] Splitting data (80 / 20) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Train Random Forest ────────────────────────────────────────
    print("\n[4/5] Training Random Forest …")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n      Accuracy: {acc * 100:.2f}%")
    print("\n      Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Save artefacts ─────────────────────────────────────────────
    print(f"[5/5] Saving model → {MODEL_PATH}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le,  ENC_PATH)
    print("      Done ✓")
    print("=" * 60)
    return acc


if __name__ == "__main__":
    train()
