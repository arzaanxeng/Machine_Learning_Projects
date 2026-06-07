import streamlit as st
import joblib
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton > button {
        background-color: #e63946;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        width: 100%;
    }
    .stButton > button:hover { background-color: #c1121f; }
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        margin-top: 1rem;
    }
    .result-positive {
        background-color: #ffe0e0;
        color: #c1121f;
        border: 1.5px solid #e63946;
    }
    .result-negative {
        background-color: #d4edda;
        color: #1a7c3e;
        border: 1.5px solid #28a745;
    }
    h1 { color: #e63946; }
    .section-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model artifacts ──────────────────────────────────────────────────────
from pathlib import Path
import joblib

@st.cache_resource
def load_artifacts():
    BASE_DIR = Path(__file__).parent

    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    model = joblib.load(BASE_DIR / "KNN_heart.pkl")
    columns = joblib.load(BASE_DIR / "columns.pkl")

    return scaler, model, columns

try:
    scaler, model, columns = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(f"❌ Could not load model files: {e}\n\nMake sure `scaler.pkl`, `KNN_heart.pkl`, and `columns.pkl` are in the same directory.")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🫀 Heart Disease Predictor")
st.caption("KNN-based classifier · Fill in the patient details below and hit **Predict**.")
st.divider()

if artifacts_loaded:

    # ── Input form ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Demographics</p>', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", ["Male", "Female"])

        st.markdown('<p class="section-title">Vitals</p>', unsafe_allow_html=True)
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=600, value=200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", ["No", "Yes"])
        max_hr = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)

    with col2:
        st.markdown('<p class="section-title">Clinical Findings</p>', unsafe_allow_html=True)
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ASY – Asymptomatic", "ATA – Atypical Angina", "NAP – Non-Anginal Pain", "TA – Typical Angina"]
        )
        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST – ST-T wave abnormality", "LVH – Left ventricular hypertrophy"]
        )
        exercise_angina = st.selectbox("Exercise-Induced Angina?", ["No", "Yes"])
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    st.divider()

    # ── Predict ───────────────────────────────────────────────────────────────
    if st.button("🔍 Predict"):

        # Map inputs → one-hot encoded feature vector matching columns.pkl
        cp_code = chest_pain.split("–")[0].strip()       # ASY / ATA / NAP / TA
        ecg_code = resting_ecg.split("–")[0].strip()     # Normal / ST / LVH
        slope_code = st_slope                             # Up / Flat / Down

        feature_dict = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": 1 if fasting_bs == "Yes" else 0,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            "Sex_F": 1 if sex == "Female" else 0,
            "Sex_M": 1 if sex == "Male" else 0,
            "ChestPainType_ASY": 1 if cp_code == "ASY" else 0,
            "ChestPainType_ATA": 1 if cp_code == "ATA" else 0,
            "ChestPainType_NAP": 1 if cp_code == "NAP" else 0,
            "ChestPainType_TA":  1 if cp_code == "TA"  else 0,
            "RestingECG_LVH":    1 if ecg_code == "LVH"    else 0,
            "RestingECG_Normal": 1 if ecg_code == "Normal" else 0,
            "RestingECG_ST":     1 if ecg_code == "ST"     else 0,
            "ExerciseAngina_N":  1 if exercise_angina == "No"  else 0,
            "ExerciseAngina_Y":  1 if exercise_angina == "Yes" else 0,
            "ST_Slope_Down": 1 if slope_code == "Down" else 0,
            "ST_Slope_Flat": 1 if slope_code == "Flat" else 0,
            "ST_Slope_Up":   1 if slope_code == "Up"   else 0,
        }

        # Build array in exact column order
        input_array = np.array([[feature_dict[col] for col in columns]])

        # Scale & predict
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        # Result display
        if prediction == 1:
            st.markdown(
                f'<div class="result-box result-positive">'
                f'⚠️ High risk of Heart Disease detected<br>'
                f'<span style="font-size:0.9rem;font-weight:400;">Model confidence: {proba[1]*100:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box result-negative">'
                f'✅ No Heart Disease detected<br>'
                f'<span style="font-size:0.9rem;font-weight:400;">Model confidence: {proba[0]*100:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Probability bar
        st.markdown("**Prediction Probability**")
        prob_col1, prob_col2 = st.columns(2)
        prob_col1.metric("No Disease", f"{proba[0]*100:.1f}%")
        prob_col2.metric("Heart Disease", f"{proba[1]*100:.1f}%")
        st.progress(float(proba[1]))

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption("⚠️ This tool is for educational purposes only and is **not** a substitute for professional medical advice.")
