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

# ==========================================================
# Hide Streamlit Default Menu & Footer
# ==========================================================

hide_streamlit_style = """
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ==========================================================
# Custom Medical Theme
# ==========================================================

st.markdown("""
<style>

/* Main App Background */
.stApp {
    background-color: #F5F7FA;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0F4C81;
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background-color: #E63946;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    padding: 12px;
}

.stButton > button:hover {
    background-color: #C1121F;
    color: white;
}

/* Input Boxes */
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    border-radius: 8px;
}

/* Labels */
label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    color: #1D3557 !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

/* Headings */
h1 {
    color: #0F4C81;
    font-weight: 700;
}

h2 {
    color: #0F4C81;
    font-weight: 700;
}

h3 {
    color: #1D3557;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# Load Model and Scaler
# ----------------------------------------------------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "optimized_xgboost.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

# ==========================================================
# Hero Section
# ==========================================================

st.markdown("""
<h1 style='text-align: center; color: #0F4C81;'>
❤️ Heart Disease Prediction & Risk Assessment
</h1>

<h4 style='text-align: center; color: #555555;'>
AI Powered Clinical Decision Support System
</h4>

<p style='text-align: center; font-size:18px; color:#666666;'>
Predict the likelihood of heart disease using an Optimized XGBoost Machine Learning model
and receive an instant personalized risk assessment.
</p>
""", unsafe_allow_html=True)

st.divider()
st.write("")

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("🩺 Dashboard")
st.sidebar.markdown("---")

st.sidebar.subheader("📋 About Project")

st.sidebar.info(
    """
This application predicts the likelihood of heart disease
using an Optimized XGBoost Machine Learning model and
provides a personalized risk assessment.
"""
)

st.sidebar.subheader("🤖 Model Information")

st.sidebar.write("**Algorithm:** Optimized XGBoost")
st.sidebar.write("**Problem Type:** Binary Classification")

st.sidebar.subheader("📊 Model Performance")

st.sidebar.write("Accuracy : **73.77%**")
st.sidebar.write("Precision : **76.09%**")
st.sidebar.write("Recall : **68.34%**")
st.sidebar.write("F1-Score : **72.01%**")
st.sidebar.write("ROC-AUC : **73.70%**")

st.sidebar.subheader("👩‍💻 Developer")

st.sidebar.write("**Dilpreet Kaur**")
st.sidebar.write("B.Sc. Information Technology")
st.sidebar.write("Lovely Professional University")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Heart Disease Prediction System")

# ==========================================================
# Patient Information
# ==========================================================

st.markdown("""
<h2 style="
color:#0F4C81;
font-weight:700;
margin-top:15px;
margin-bottom:10px;">
👤 Patient Information
</h2>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:

    gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
    )
    gender = 1 if gender == "Male" else 0

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
with col2:

    weight = st.number_input(
        "Weight (kg)",
        min_value=30,
        max_value=200,
        value=70
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=24.0
    )
    
# ==========================================================
# Clinical Measurements
# ==========================================================

st.markdown("""
<h2 style="
color:#0F4C81;
font-weight:700;
margin-top:20px;
margin-bottom:10px;">
🩺 Clinical Measurements
</h2>
""", unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:

    ap_hi = st.number_input(
        "Systolic Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )

    cholesterol = st.selectbox(
    "Cholesterol Level",
    [
        "Normal",
        "Above Normal",
        "Well Above Normal"
    ]
    )
    cholesterol = {
        "Normal": 1,
        "Above Normal": 2,
        "Well Above Normal": 3
    }[cholesterol]
    
with col4:

    ap_lo = st.number_input(
        "Diastolic Blood Pressure",
        min_value=30,
        max_value=200,
        value=80
    )

    gluc = st.selectbox(
    "Glucose Level",
    [
        "Normal",
        "Above Normal",
        "Well Above Normal"
    ]
    )
    gluc = {
            "Normal": 1,
            "Above Normal": 2,
            "Well Above Normal": 3
    }[gluc]
    
# ==========================================================
# Lifestyle Information
# ==========================================================

st.markdown("""
<h2 style="
color:#0F4C81;
font-weight:700;
margin-top:20px;
margin-bottom:10px;">
🚶 Lifestyle Information
</h2>
""", unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:

    smoke = st.selectbox(
    "Smoking Habit",
    ["No", "Yes"]
    )
    smoke = 1 if smoke == "Yes" else 0

    alco = st.selectbox(
    "Alcohol Consumption",
    ["No", "Yes"]
    )
    alco = 1 if alco == "Yes" else 0
    
with col6:

    active = st.selectbox(
    "Physically Active",
    ["No", "Yes"]
    )
    active = 1 if active == "Yes" else 0

patient_data = pd.DataFrame({

    "gender": [gender],
    "age_years": [age_years],
    "height": [height],
    "weight": [weight],
    "ap_hi": [ap_hi],
    "ap_lo": [ap_lo],
    "cholesterol": [cholesterol],
    "gluc": [gluc],
    "smoke": [smoke],
    "alco": [alco],
    "active": [active],
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
    
    st.write(patient_data)

    # Predict
    prediction = model.predict(patient_data)[0]

    # Prediction probability
    probability = model.predict_proba(patient_data)[0]

    # Heart disease risk score
    risk_score = float(probability[1]) * 100
    
    st.markdown("---")

    st.markdown("""
    <h2 style='text-align:center; color:#0F4C81;'>
    👤 Patient Summary
    </h2>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Age", f"{age_years} Years")
        st.metric("Gender", "Male" if gender == 1 else "Female")

    with col2:
        st.metric("Height", f"{height} cm")
        st.metric("Weight", f"{weight} kg")

    with col3:
        st.metric("BMI", f"{bmi:.1f}")
        st.metric("Blood Pressure", f"{ap_hi}/{ap_lo}")

    # Prediction result
    if prediction == 1:
        prediction_result = "❤️ Heart Disease Detected"
    else:
        prediction_result = "💚 No Heart Disease Detected"

    st.markdown("""
    <h2 style='text-align:center;color:#0F4C81;'>
    📋 Prediction Report
    </h2>
    """, unsafe_allow_html=True)

    if prediction == 1:
        st.markdown("""
        <div style="
            background-color:#FDECEC;
            padding:20px;
            border-radius:15px;
            border-left:8px solid #E63946;
            margin-bottom:20px;
        ">
            <h3 style="color:#C1121F;">❤️ Heart Disease Detected</h3>
            <p style="font-size:18px; color:#333333;">
            The model predicts that the patient is at risk of heart disease.
            Medical consultation is recommended.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            background-color:#EDF7ED;
            padding:20px;
            border-radius:15px;
            border-left:8px solid #2E8B57;
            margin-bottom:20px;
        ">
            <h3 style="color:#2E8B57;">💚 No Heart Disease Detected</h3>
            <p style="font-size:18px; color:#333333;">
            The model predicts a low likelihood of heart disease.
            Continue maintaining a healthy lifestyle.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Risk category
    if risk_score < 40:
        risk_level = "🟢 Low Risk"
    elif risk_score < 70:
        risk_level = "🟡 Moderate Risk"
    else:
        risk_level = "🔴 High Risk"

    st.markdown("---")

    st.markdown("""
    <h2 style='text-align:center;color:#0F4C81;'>
    📋 Prediction Report
    </h2>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <h4 style="color:#1D3557;">Risk Score</h4>
        <h2 style="color:#E63946;">{risk_score:.2f}%</h2>
        """, unsafe_allow_html=True)


    with col2:
        prediction_text = "Positive" if prediction == 1 else "Negative"

        st.markdown(f"""
        <h4 style="color:#1D3557;">Prediction</h4>
        <h2 style="color:#0F4C81;">{prediction_text}</h2>
        """, unsafe_allow_html=True)

    with col3:
        if risk_score < 40:
            st.success("🟢 Low Risk")
        elif risk_score < 70:
            st.warning("🟡 Moderate Risk")
        else:
            st.error("🔴 High Risk")
        
    st.progress(float(risk_score) / 100)

    st.info(f"Estimated Probability of Heart Disease : {risk_score:.2f}%")
    
    st.markdown("---")

    st.markdown("""
    <h2 style="color:#0F4C81;">
    💡 Personalized Health Recommendations
    </h2>
    """, unsafe_allow_html=True)

    if risk_score < 40:

        st.markdown("""
        <div style="background:#E8F5E9;
                padding:20px;
                border-radius:12px;
                border-left:8px solid #2E7D32;
                color:#222;">
        <h3>🟢 Low Risk</h3>

        ✅ Continue regular physical activity.<br>
        ✅ Maintain a balanced diet.<br>
        ✅ Drink plenty of water.<br>
        ✅ Get routine health check-ups.<br>

        </div>
        """, unsafe_allow_html=True)

    elif risk_score < 70:

        st.markdown("""
        <div style="background:#FFF8E1;
                padding:20px;
                border-radius:12px;
                border-left:8px solid #F9A825;
                color:#222;">
        <h3>🟡 Moderate Risk</h3>

        ⚠ Monitor your blood pressure regularly.<br>
        ⚠ Reduce salt and unhealthy fats.<br>
        ⚠ Exercise for at least 30 minutes daily.<br>
        ⚠ Visit a doctor if symptoms appear.<br>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style="background:#FFEBEE;
                padding:20px;
                border-radius:12px;
                border-left:8px solid #D32F2F;
                color:#222;">
        <h3>🔴 High Risk</h3>

        🚨 Consult a cardiologist immediately.<br>
        💊 Follow prescribed medications.<br>
        🥗 Follow a heart-healthy diet.<br>
        🚭 Avoid smoking and alcohol.<br>
        🏃 Exercise as advised by your doctor.<br>
        📅 Schedule regular medical check-ups.<br>

        </div>
        """, unsafe_allow_html=True)

    st.caption(
    "Developed by Dilpreet Kaur | Heart Disease Prediction & Risk Assessment using Machine Learning"
    )