# ==========================================================
# Heart Disease Prediction and Risk Assessment Web App
# ==========================================================

# Import Libraries
import streamlit as st
import joblib
import pandas as pd

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction & Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# Load Model and Scaler
# ----------------------------------------------------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "optimized_xgboost.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

# ----------------------------------------------------------
# Application Title
# ----------------------------------------------------------

st.title("❤️ Heart Disease Prediction & Risk Assessment")

st.markdown(
"""
Predict the likelihood of heart disease using Machine Learning and obtain an estimated risk score for better clinical decision-making.
"""
)

st.divider()

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

st.sidebar.header("👤 Personal Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

age = st.sidebar.number_input(
    "Age (Years)",
    min_value=1,
    max_value=120,
    value=50
)

height = st.sidebar.number_input(
    "Height (cm)",
    min_value=100,
    max_value=250,
    value=170
)

weight = st.sidebar.number_input(
    "Weight (kg)",
    min_value=20,
    max_value=250,
    value=70
)

st.sidebar.header("❤️ Health Information")

systolic_bp = st.sidebar.number_input(
    "Systolic Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

diastolic_bp = st.sidebar.number_input(
    "Diastolic Blood Pressure",
    min_value=30,
    max_value=200,
    value=80
)

cholesterol = st.sidebar.selectbox(
    "Cholesterol Level",
    [
        "Normal",
        "Above Normal",
        "Well Above Normal"
    ]
)

glucose = st.sidebar.selectbox(
    "Glucose Level",
    [
        "Normal",
        "Above Normal",
        "Well Above Normal"
    ]
)

bmi = st.sidebar.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=24.5,
    step=0.1
)

st.sidebar.header("🏃 Lifestyle Information")

smoke = st.sidebar.selectbox(
    "Smoking",
    ["No", "Yes"]
)

alcohol = st.sidebar.selectbox(
    "Alcohol Intake",
    ["No", "Yes"]
)

active = st.sidebar.selectbox(
    "Physical Activity",
    ["Yes", "No"]
)

st.sidebar.divider()

st.sidebar.markdown("---")

st.sidebar.info(
"""
This application predicts the likelihood of heart disease using an Optimized XGBoost Machine Learning model.
"""
)

# ==========================================================
# Convert User Inputs into Model Format
# ==========================================================

gender = 1 if gender == "Male" else 0

cholesterol = {
    "Normal": 1,
    "Above Normal": 2,
    "Well Above Normal": 3
}[cholesterol]

glucose = {
    "Normal": 1,
    "Above Normal": 2,
    "Well Above Normal": 3
}[glucose]

smoke = 1 if smoke == "Yes" else 0
alcohol = 1 if alcohol == "Yes" else 0
active = 1 if active == "Yes" else 0

patient_data = pd.DataFrame({

    "gender": [gender],
    "height": [height],
    "weight": [weight],
    "ap_hi": [systolic_bp],
    "ap_lo": [diastolic_bp],
    "cholesterol": [cholesterol],
    "gluc": [glucose],
    "smoke": [smoke],
    "alco": [alcohol],
    "active": [active],
    "age_years": [age],
    "bmi": [bmi]

})

patient_data = patient_data[
    [
        "gender",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "age_years",
        "bmi"
    ]
]

numerical_features = [
    "age_years",
    "height",
    "weight",
    "bmi",
    "ap_hi",
    "ap_lo"
]

patient_data[numerical_features] = scaler.transform(
    patient_data[numerical_features]
)

# ==========================================================
# Step 6: Predict Heart Disease and Display Results
# ==========================================================

if st.button("❤️ Predict Heart Disease"):

    # Predict
    prediction = model.predict(patient_data)[0]

    # Prediction probability
    probability = model.predict_proba(patient_data)[0]

    # Heart disease risk score
    risk_score = probability[1] * 100

    # Prediction result
    if prediction == 1:
        prediction_result = "❤️ Heart Disease Detected"
    else:
        prediction_result = "💚 No Heart Disease"

    # Risk category
    if risk_score < 40:
        risk_level = "🟢 Low Risk"
    elif risk_score < 70:
        risk_level = "🟡 Moderate Risk"
    else:
        risk_level = "🔴 High Risk"

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("❤️ Heart Disease Detected")
    else:
        st.success("💚 No Heart Disease Detected")

    st.metric(
        label="Heart Disease Risk Score",
        value=f"{risk_score:.2f}%"
    )

    if risk_score < 40:
        st.success("🟢 Low Risk")
    elif risk_score < 70:
        st.warning("🟡 Moderate Risk")
    else:
        st.error("🔴 High Risk")
        
    st.write(f"Probability of Heart Disease : **{risk_score:.2f}%**")
    
    st.markdown("---")

    st.caption(
    "Developed by Dilpreet Kaur | Heart Disease Prediction & Risk Assessment using Machine Learning"
    )