"""
predict.py
----------
Prediction helpers used by the Streamlit dashboard.
Loads the trained Random Forest model and label encoder once,
then exposes a simple predict() function.
"""

import os
import numpy as np
import joblib

MODEL_PATH = os.path.join("models", "random_forest_model.pkl")
ENC_PATH   = os.path.join("models", "label_encoder.pkl")

# ── Human-readable labels ──────────────────────────────────────────────────────
ACTION_LABELS = {
    "Emergency_Priority":      "🚨 Emergency Priority Mode",
    "Hospital_Priority":       "🏥 Hospital Priority Mode",
    "Balanced_Distribution":   "⚖️  Balanced Distribution Mode",
    "EV_Charging_Allowed":     "⚡ EV Charging Allowed",
    "Residential_Support":     "🏘️  Residential Support Mode",
}

ACTION_DESCRIPTIONS = {
    "Emergency_Priority":    "All available power is being routed to emergency services. Non-critical loads are shed.",
    "Hospital_Priority":     "Medical facilities take precedence. Hospital and emergency loads are fully supplied.",
    "Balanced_Distribution": "Power is distributed evenly across all consumer categories.",
    "EV_Charging_Allowed":   "Sufficient surplus energy exists — EV charging stations are online.",
    "Residential_Support":   "Residential areas receive priority to maintain household stability.",
}

ACTION_COLORS = {
    "Emergency_Priority":    "#FF4444",
    "Hospital_Priority":     "#FF8C00",
    "Balanced_Distribution": "#00C49A",
    "EV_Charging_Allowed":   "#4FC3F7",
    "Residential_Support":   "#AB47BC",
}

# ── Lazy-load model ────────────────────────────────────────────────────────────
_clf = None
_le  = None


def _load_model():
    global _clf, _le
    if _clf is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. "
                "Please run: python train_model.py"
            )
        _clf = joblib.load(MODEL_PATH)
        _le  = joblib.load(ENC_PATH)


def predict(
    solar_irradiance: float,
    cloud_cover: float,
    temperature: float,
    humidity: float,
    battery_level: float,
    grid_price: float,
    solar_gen_kw: float,
    hospital_load: float,
    residential_load: float,
    ev_load: float,
    emergency_load: float,
    total_load: float,
) -> dict:
    """
    Run inference and return a results dict with:
        action       – raw class string
        label        – friendly display name
        description  – narrative explanation
        color        – hex accent colour
        probabilities – dict of class → probability
    """
    _load_model()

    X = np.array([[
        solar_irradiance, cloud_cover, temperature, humidity,
        battery_level, grid_price, solar_gen_kw,
        hospital_load, residential_load, ev_load,
        emergency_load, total_load,
    ]])

    encoded  = _clf.predict(X)[0]
    action   = _le.inverse_transform([encoded])[0]
    probs    = _clf.predict_proba(X)[0]
    prob_map = {_le.classes_[i]: round(float(p) * 100, 1) for i, p in enumerate(probs)}

    return {
        "action":        action,
        "label":         ACTION_LABELS.get(action, action),
        "description":   ACTION_DESCRIPTIONS.get(action, ""),
        "color":         ACTION_COLORS.get(action, "#FFFFFF"),
        "probabilities": prob_map,
    }


def model_loaded() -> bool:
    """Return True if the model file exists on disk."""
    return os.path.exists(MODEL_PATH)
