# ============================================================
# Step 8.1 - Import Required Libraries
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# Step 8.2 - Load Optimized XGBoost Model
# ============================================================

model = joblib.load("models/optimized_xgboost.pkl")

# ============================================================
# Step 8.3 - Configure Streamlit Page
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction & Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Step 8.4 - Display Application Title
# ============================================================

st.title("❤️ Heart Disease Prediction & Risk Assessment")

st.markdown(
    """
    Predict the likelihood of heart disease using an optimized
    XGBoost Machine Learning model trained on clinical health data.
    """
)

st.divider()

# ============================================================
# Step 8.5 - Create Sidebar
# ============================================================

with st.sidebar:

    st.header("📋 Patient Information")

    st.markdown(
        """
        Enter the patient's clinical information to predict the
        likelihood of heart disease and assess the overall risk level.
        """
    )

    st.divider()

    st.info(
        """
        **Model Used**

        Optimized XGBoost Classifier
        """
    )
    
# ============================================================
# Step 8.6 - Create Patient Input Fields
# ============================================================

with st.sidebar:

    gender = st.selectbox(
        "Gender",
        options=[1, 2],
        format_func=lambda x: "Female" if x == 1 else "Male"
    )

    age_years = st.number_input(
        "Age (Years)",
        min_value=18,
        max_value=100,
        value=50
    )

    height = st.number_input(
        "Height (cm)",
        min_value=100,
        max_value=250,
        value=170
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=250.0,
        value=70.0
    )

    ap_hi = st.number_input(
        "Systolic Blood Pressure",
        min_value=50,
        max_value=300,
        value=120
    )

    ap_lo = st.number_input(
        "Diastolic Blood Pressure",
        min_value=30,
        max_value=200,
        value=80
    )

    cholesterol = st.selectbox(
        "Cholesterol Level",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "Normal",
            2: "Above Normal",
            3: "Well Above Normal"
        }[x]
    )

    gluc = st.selectbox(
        "Glucose Level",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "Normal",
            2: "Above Normal",
            3: "Well Above Normal"
        }[x]
    )

    smoke = st.selectbox(
        "Smoking",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    alco = st.selectbox(
        "Alcohol Intake",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    active = st.selectbox(
        "Physically Active",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    bmi = weight / ((height / 100) ** 2)

    st.metric(
        label="Calculated BMI",
        value=f"{bmi:.2f}"
    )

    predict_button = st.button(
        "Predict Heart Disease Risk",
        use_container_width=True
    )
    
# ============================================================
# Step 8.7 - Create Input DataFrame
# ============================================================

if predict_button:

    input_data = pd.DataFrame({
        "gender": [gender],
        "height": [height],
        "weight": [weight],
        "ap_hi": [ap_hi],
        "ap_lo": [ap_lo],
        "cholesterol": [cholesterol],
        "gluc": [gluc],
        "smoke": [smoke],
        "alco": [alco],
        "active": [active],
        "age_years": [age_years],
        "bmi": [bmi]
    })

    st.subheader("Patient Information")

    st.dataframe(
        input_data,
        use_container_width=True
    )
    
# ============================================================
# Step 8.8 - Generate Heart Disease Prediction
# ============================================================

    prediction = model.predict(input_data)[0]

    prediction_probability = model.predict_proba(input_data)[0]

    risk_score = prediction_probability[1]
    
# ============================================================
# Step 8.9 - Display Heart Disease Prediction
# ============================================================

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")
        
# ============================================================
# Step 8.10 - Display Heart Disease Risk Score
# ============================================================

    st.subheader("Heart Disease Risk Score")

    st.progress(float(risk_score))

    st.metric(
        label="Predicted Risk Score",
        value=f"{risk_score * 100:.2f}%"
    )
    
# ============================================================
# Step 8.11 - Assign Risk Category
# ============================================================

    if risk_score < 0.30:
        risk_category = "Low Risk"

    elif risk_score < 0.70:
        risk_category = "Moderate Risk"

    else:
        risk_category = "High Risk"
        
# ============================================================
# Step 8.12 - Display Risk Category
# ============================================================

    st.subheader("Risk Category")

    if risk_category == "Low Risk":
        st.success(f"🟢 {risk_category}")

    elif risk_category == "Moderate Risk":
        st.warning(f"🟡 {risk_category}")

    else:
        st.error(f"🔴 {risk_category}")
        
# ============================================================
# Step 8.13 - Display Personalized Health Recommendation
# ============================================================

    st.subheader("Health Recommendation")

    if risk_category == "Low Risk":

        st.success(
            """
            ✅ Your predicted heart disease risk is **Low**.

            **Recommendations:**
            - Maintain a healthy and balanced diet.
            - Continue regular physical activity.
            - Monitor your blood pressure periodically.
            - Attend routine health check-ups.
            """
        )

    elif risk_category == "Moderate Risk":

        st.warning(
            """
            ⚠️ Your predicted heart disease risk is **Moderate**.

            **Recommendations:**
            - Improve your diet by reducing salt and saturated fats.
            - Exercise regularly (at least 30 minutes daily).
            - Monitor blood pressure and cholesterol levels.
            - Consult a healthcare professional if symptoms develop.
            """
        )

    else:

        st.error(
            """
            🚨 Your predicted heart disease risk is **High**.

            **Recommendations:**
            - Consult a cardiologist or healthcare professional immediately.
            - Monitor blood pressure, cholesterol and blood glucose regularly.
            - Follow a heart-healthy diet.
            - Avoid smoking and alcohol consumption.
            - Engage in physical activity only as advised by your doctor.
            """
        )
        
# ============================================================
# Step 8.14 - Display Model Information
# ============================================================

st.divider()

with st.expander("Model Information"):

    st.markdown("""
    **Machine Learning Model:** Optimized XGBoost Classifier

    **Project:** Heart Disease Prediction & Risk Assessment Using Machine Learning

    **Input Features:**
    - Gender
    - Age
    - Height
    - Weight
    - Body Mass Index (BMI)
    - Systolic Blood Pressure
    - Diastolic Blood Pressure
    - Cholesterol Level
    - Glucose Level
    - Smoking Status
    - Alcohol Consumption
    - Physical Activity

    **Output:**
    - Heart Disease Prediction
    - Risk Score
    - Risk Category
    """)
    
# ============================================================
# Step 8.15 - Display Application Footer
# ============================================================

st.divider()

st.markdown(
    """
    <div style='text-align: center; color: gray;'>

    ❤️ <b>Heart Disease Prediction & Risk Assessment</b><br>

    Developed using <b>Streamlit</b> and an <b>Optimized XGBoost</b> Machine Learning Model.

    <br><br>

    <i>This application is intended for educational purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment.</i>

    </div>
    """,
    unsafe_allow_html=True
)

