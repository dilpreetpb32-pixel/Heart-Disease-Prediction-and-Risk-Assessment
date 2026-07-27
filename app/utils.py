"""
Helper functions for the Heart Disease Prediction & Risk Assessment dashboard.

Keeping these separate from app.py keeps the page script focused on layout,
while all data/formatting/visual-generation logic lives here and can be
unit-tested or reused independently.
"""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "optimized_xgboost.pkl"
ASSETS_DIR = ROOT_DIR / "assets"
IMAGES_DIR = ROOT_DIR / "images"

# Exact column order the model was trained on. Do not reorder -- XGBoost
# (loaded from a plain pickle rather than a Booster with named features)
# relies on positional column order matching training time.
FEATURE_ORDER = [
    "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active",
    "age_years", "bmi",
]

# Risk thresholds used consistently throughout the notebooks and app.
RISK_LOW_MAX = 0.30
RISK_MODERATE_MAX = 0.70

RISK_THEME = {
    "Low Risk": {
        "color": "#1E8A5F",
        "bg": "#E7F5EE",
        "icon": "🟢",
        "verdict": "Low predicted likelihood of heart disease.",
    },
    "Moderate Risk": {
        "color": "#B9791A",
        "bg": "#FBF1DD",
        "icon": "🟡",
        "verdict": "Moderate predicted likelihood -- worth a closer look.",
    },
    "High Risk": {
        "color": "#C0392B",
        "bg": "#FBE9E7",
        "icon": "🔴",
        "verdict": "High predicted likelihood of heart disease.",
    },
}

GENDER_LABELS = {1: "Female", 2: "Male"}
LEVEL_LABELS = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
YES_NO_LABELS = {0: "No", 1: "Yes"}

RISK_TIER = {"Low Risk": "ok", "Moderate Risk": "warn", "High Risk": "danger"}

FEATURE_LABELS = {
    "ap_hi": "Systolic blood pressure",
    "cholesterol": "Cholesterol level",
    "ap_lo": "Diastolic blood pressure",
    "age_years": "Age",
    "active": "Physical activity",
    "smoke": "Smoking status",
    "gluc": "Glucose level",
    "alco": "Alcohol intake",
    "bmi": "Body mass index",
    "weight": "Weight",
    "gender": "Gender",
    "height": "Height",
}

MODEL_METRICS = {
    # Pulled from notebooks/06_Model_Optimization_Risk_Assessment.ipynb
    "Accuracy": 0.7283,
    "Precision": 0.7485,
    "Recall": 0.6973,
    "F1-Score": 0.7220,
    "ROC-AUC": 0.7960,
}

# Base-model comparison, pulled from notebooks/05_Model_Evaluation_and_Comparison.ipynb
MODEL_COMPARISON = pd.DataFrame(
    [
        ["XGBoost", 0.7284, 0.7491, 0.6963, 0.7218, 0.7951],
        ["Support Vector Machine", 0.7277, 0.7527, 0.6877, 0.7187, 0.7885],
        ["Logistic Regression", 0.7229, 0.7463, 0.6854, 0.7145, 0.7875],
        ["Random Forest", 0.6984, 0.7036, 0.6980, 0.7008, 0.7501],
        ["K-Nearest Neighbors", 0.6866, 0.6927, 0.6843, 0.6885, 0.7338],
        ["Decision Tree", 0.6190, 0.6240, 0.6215, 0.6227, 0.6183],
    ],
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
)

# Test-set risk category counts from the optimized model (notebook 6).
RISK_CATEGORY_TEST_COUNTS = {"Low Risk": 3581, "Moderate Risk": 5204, "High Risk": 4200}


# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------

def inject_css() -> None:
    """Load the external stylesheet into the page."""
    css_path = ASSETS_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the optimized XGBoost classifier used for deployment."""
    return joblib.load(MODEL_PATH)


# ------------------------------------------------------------------
# Clinical helpers
# ------------------------------------------------------------------

def compute_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)


def bmi_category(bmi: float) -> tuple[str, str]:
    """Returns (label, tier) where tier drives badge color."""
    if bmi < 18.5:
        return "Underweight", "warn"
    if bmi < 25:
        return "Normal", "ok"
    if bmi < 30:
        return "Overweight", "warn"
    return "Obese", "danger"


def bp_category(ap_hi: float, ap_lo: float) -> tuple[str, str]:
    """Blood pressure category per standard AHA reference ranges."""
    if ap_hi >= 180 or ap_lo >= 120:
        return "Hypertensive Crisis", "danger"
    if ap_hi >= 140 or ap_lo >= 90:
        return "Hypertension Stage 2", "danger"
    if ap_hi >= 130 or ap_lo >= 80:
        return "Hypertension Stage 1", "warn"
    if ap_hi >= 120:
        return "Elevated", "warn"
    return "Normal", "ok"


def categorize_risk(score: float) -> str:
    if score < RISK_LOW_MAX:
        return "Low Risk"
    if score < RISK_MODERATE_MAX:
        return "Moderate Risk"
    return "High Risk"


def contributing_factors(values: dict) -> list[str]:
    """Rule-based explanation of which entered values sit outside normal
    reference ranges -- gives the prediction some interpretability beyond
    a bare probability, grounded in the same fields the model is trained on.
    """
    factors = []

    bp_label, bp_tier = bp_category(values["ap_hi"], values["ap_lo"])
    if bp_tier in ("warn", "danger"):
        factors.append(
            f"Blood pressure of {values['ap_hi']:.0f}/{values['ap_lo']:.0f} mmHg "
            f"falls under **{bp_label}**."
        )

    if values["cholesterol"] >= 2:
        label = "well above normal" if values["cholesterol"] == 3 else "above normal"
        factors.append(f"Cholesterol level is **{label}**.")

    if values["gluc"] >= 2:
        label = "well above normal" if values["gluc"] == 3 else "above normal"
        factors.append(f"Glucose level is **{label}**.")

    bmi_label, bmi_tier = bmi_category(values["bmi"])
    if bmi_tier in ("warn", "danger"):
        factors.append(f"BMI of {values['bmi']:.1f} falls in the **{bmi_label.lower()}** range.")

    if values["smoke"] == 1:
        factors.append("Current **smoking** status is a significant, modifiable risk factor.")

    if values["alco"] == 1:
        factors.append("Regular **alcohol intake** can contribute to elevated blood pressure.")

    if values["active"] == 0:
        factors.append("**Low physical activity** is associated with increased cardiovascular risk.")

    if values["age_years"] >= 55:
        factors.append(f"Age ({values['age_years']:.0f}) is a non-modifiable risk factor.")

    return factors


# ------------------------------------------------------------------
# Visual components (return raw HTML/SVG strings for st.markdown)
# ------------------------------------------------------------------

def badge(label: str, tier: str) -> str:
    """A small colored pill. tier in {ok, warn, danger, info}."""
    return f'<span class="badge badge-{tier}">{label}</span>'


def gauge_svg(risk_pct: float, needle_color: str) -> str:
    """A semicircular clinical-dial gauge (in the style of a sphygmomanometer)
    showing the risk score from 0-100, with colored Low/Moderate/High zones.
    """
    cx, cy, r = 160, 150, 128

    def pt(fraction: float, radius: float = r) -> tuple[float, float]:
        angle_deg = -90 + fraction * 180
        angle_rad = math.radians(angle_deg)
        x = cx + radius * math.sin(angle_rad)
        y = cy - radius * math.cos(angle_rad)
        return x, y

    zones = [(0.0, RISK_LOW_MAX, "#1E8A5F"), (RISK_LOW_MAX, RISK_MODERATE_MAX, "#DD9A2B"),
             (RISK_MODERATE_MAX, 1.0, "#D8432E")]

    arcs = []
    for f0, f1, color in zones:
        x0, y0 = pt(f0)
        x1, y1 = pt(f1)
        arcs.append(
            f'<path d="M {x0:.1f} {y0:.1f} A {r} {r} 0 0 1 {x1:.1f} {y1:.1f}" '
            f'stroke="{color}" stroke-width="22" fill="none" opacity="0.92"/>'
        )

    ticks = []
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        x_out, y_out = pt(f, r + 13)
        x_in, y_in = pt(f, r - 13)
        ticks.append(
            f'<line x1="{x_in:.1f}" y1="{y_in:.1f}" x2="{x_out:.1f}" y2="{y_out:.1f}" '
            f'stroke="#0B2545" stroke-width="2" opacity="0.30"/>'
        )

    fraction = max(0.0, min(1.0, risk_pct / 100))
    nx, ny = pt(fraction, r - 34)
    needle = (
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" '
        f'stroke="{needle_color}" stroke-width="5" stroke-linecap="round"/>'
    )
    hub = f'<circle cx="{cx}" cy="{cy}" r="8" fill="{needle_color}" stroke="#fff" stroke-width="2"/>'

    labels = (
        f'<text x="{cx - r - 4:.0f}" y="{cy + 24}" class="gauge-tick-label" text-anchor="middle">0</text>'
        f'<text x="{cx}" y="{cy - r - 16}" class="gauge-tick-label" text-anchor="middle">50</text>'
        f'<text x="{cx + r + 4:.0f}" y="{cy + 24}" class="gauge-tick-label" text-anchor="middle">100</text>'
    )

    return (
        f'<svg viewBox="0 0 320 180" xmlns="http://www.w3.org/2000/svg" class="gauge-svg">'
        f'{"".join(arcs)}{"".join(ticks)}{needle}{hub}{labels}'
        f'</svg>'
    )


def build_input_dataframe(values: dict) -> pd.DataFrame:
    """Build the single-row model input frame in the exact trained column order."""
    return pd.DataFrame({col: [values[col]] for col in FEATURE_ORDER})


def summary_table(values: dict) -> pd.DataFrame:
    """A human-readable view of the entered patient values."""
    rows = [
        ("Gender", GENDER_LABELS[values["gender"]]),
        ("Age", f"{values['age_years']:.0f} yrs"),
        ("Height", f"{values['height']:.0f} cm"),
        ("Weight", f"{values['weight']:.1f} kg"),
        ("BMI", f"{values['bmi']:.1f}"),
        ("Systolic BP", f"{values['ap_hi']:.0f} mmHg"),
        ("Diastolic BP", f"{values['ap_lo']:.0f} mmHg"),
        ("Cholesterol", LEVEL_LABELS[values["cholesterol"]]),
        ("Glucose", LEVEL_LABELS[values["gluc"]]),
        ("Smoker", YES_NO_LABELS[values["smoke"]]),
        ("Alcohol intake", YES_NO_LABELS[values["alco"]]),
        ("Physically active", YES_NO_LABELS[values["active"]]),
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"])
