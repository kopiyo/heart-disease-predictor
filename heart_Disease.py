import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS - VIBRANT ANIMATED BACKGROUND
st.markdown("""
<style>
    /* VIBRANT ANIMATED GRADIENT BACKGROUND */
    .stApp {
        background: linear-gradient(-45deg, 
            #FF6B9D,  /* Hot Pink */
            #C44569,  /* Red Wine */
            #FFA502,  /* Orange */
            #FF6348,  /* Coral */
            #786FA6,  /* Purple */
            #F8B500   /* Gold */
        );
        background-size: 400% 400%;
        animation: gradientShift 12s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        25% { background-position: 50% 100%; }
        50% { background-position: 100% 50%; }
        75% { background-position: 50% 0%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main container with white background */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.92);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(15px);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Sidebar with vibrant edge */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, 
            rgba(255, 107, 157, 0.15) 0%,
            rgba(255, 255, 255, 0.95) 20%
        );
        backdrop-filter: blur(10px);
        border-right: 3px solid rgba(255, 107, 157, 0.3);
    }
    
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #FF6B9D 0%, #C44569 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.3rem;
        font-weight: bold;
        line-height: 1.2;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: titlePulse 3s ease-in-out infinite;
    }
    
    @keyframes titlePulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .sub-header {
        font-size: 1rem;
        color: #34495E;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    h2 { font-size: 1.5rem !important; margin-top: 0.5rem !important; margin-bottom: 0.5rem !important; }
    h3 { font-size: 1.2rem !important; margin-top: 0.5rem !important; margin-bottom: 0.5rem !important; }
    h4 { 
        font-size: 1rem !important; 
        margin-top: 0.3rem !important; 
        margin-bottom: 0.3rem !important;
        color: #C44569;
        font-weight: 600;
    }
    
    /* VIBRANT RISK BOXES */
    .risk-box {
        padding: 0.7rem;
        border-radius: 10px;
        margin: 0.4rem 0;
        border-left: 4px solid;
        animation: slideInBounce 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
    }
    
    @keyframes slideInBounce {
        0% {
            opacity: 0;
            transform: translateY(-30px) scale(0.9);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    .risk-low { 
        background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
        border-color: #2F80ED;
        color: white;
    }
    .risk-low h2, .risk-low h3, .risk-low p { color: white !important; }
    
    .risk-medium { 
        background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%);
        border-color: #F2994A;
        color: white;
    }
    .risk-medium h2, .risk-medium h3, .risk-medium p { color: white !important; }
    
    .risk-high { 
        background: linear-gradient(135deg, #EB5757 0%, #FF6B9D 100%);
        border-color: #EB5757;
        color: white;
    }
    .risk-high h2, .risk-high h3, .risk-high p { color: white !important; }
    
    .risk-box h2 { font-size: 1.1rem !important; margin: 0 !important; font-weight: bold; }
    .risk-box h3 { font-size: 0.95rem !important; margin: 0.2rem 0 !important; }
    .risk-box p { font-size: 0.88rem !important; margin: 0 !important; }
    
    /* GLOWING ANIMATED BUTTON */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #FF6B9D 0%, #C44569 100%);
        color: white;
        font-size: 1.1rem;
        padding: 0.7rem;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        transition: all 0.4s ease;
        box-shadow: 0 5px 15px rgba(255, 107, 157, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover:before {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(255, 107, 157, 0.6);
        background: linear-gradient(135deg, #C44569 0%, #FF6B9D 100%);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* Input fields with color accents */
    .stNumberInput input, .stSelectbox select {
        border-radius: 8px;
        border: 2px solid #E0E0E0;
        transition: all 0.3s ease;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #FF6B9D;
        box-shadow: 0 0 0 3px rgba(255, 107, 157, 0.2);
        outline: none;
    }
    
    /* Colorful metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
        padding: 0.6rem;
        border-radius: 10px;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.08);
        border-left: 3px solid #FF6B9D;
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.12);
    }
    
    /* Expander with vibrant accent */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(255, 107, 157, 0.1) 0%, rgba(196, 69, 105, 0.1) 100%);
        border-radius: 8px;
        border-left: 3px solid #FF6B9D;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(255, 107, 157, 0.2) 0%, rgba(196, 69, 105, 0.2) 100%);
        transform: translateX(5px);
    }
    
    /* Download button with glow */
    .stDownloadButton button {
        background: linear-gradient(135deg, #FFA502 0%, #FF6348 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.6rem !important;
        box-shadow: 0 4px 12px rgba(255, 165, 2, 0.4) !important;
        animation: glowPulse 2s ease-in-out infinite;
        border: none !important;
    }
    
    @keyframes glowPulse {
        0%, 100% {
            box-shadow: 0 4px 12px rgba(255, 165, 2, 0.4);
        }
        50% {
            box-shadow: 0 4px 20px rgba(255, 165, 2, 0.7);
        }
    }
    
    .stDownloadButton button:hover {
        transform: scale(1.03);
        background: linear-gradient(135deg, #FF6348 0%, #FFA502 100%) !important;
    }
    
    .element-container { margin-bottom: 0.5rem; }
    ul, ol { margin-top: 0.3rem; margin-bottom: 0.3rem; padding-left: 1.5rem; }
    li { margin-bottom: 0.2rem; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load("heart_disease_model.joblib")
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Run Heart_Disease.ipynb first (cell 21)")
        st.stop()

model = load_model()

# SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-with-pulse.png", width=80)
    
    st.markdown("### 🎯 Risk Levels")
    st.success("**Low**: < 30%")
    st.warning("**Medium**: 30-70%")
    st.error("**High**: > 70%")
    
    st.markdown("---")
    st.warning("⚠️ **Disclaimer**  \nEducational tool only.  \nNot for medical diagnosis.")

# HEADER
st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Logistic Regression | ROC-AUC: 0.9154</p>', unsafe_allow_html=True)

# THREE COLUMN LAYOUT
col1, col2, results_col = st.columns([1, 1, 1.2])

# COLUMN 1
with col1:
    st.markdown("#### 👤 Demographics & Vitals")
    age = st.number_input("Age", 1, 120, 50, key="age")
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male", key="sex")
    trestbps = st.number_input("BP (mm Hg)", 50, 250, 120, key="bp")
    chol = st.number_input("Cholesterol", 50, 600, 200, key="chol")
    thalach = st.number_input("Max HR", 50, 250, 150, key="hr")

# COLUMN 2
with col2:
    st.markdown("#### 🏥 Blood & ECG Tests")
    fbs = st.selectbox("Fasting Sugar >120", [0, 1], 
                      format_func=lambda x: "No" if x==0 else "Yes", key="fbs")
    restecg = st.selectbox("Resting ECG", [0, 1, 2],
                          format_func=lambda x: ["Normal", "Abnormal", "LVH"][x], key="ecg")
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, 0.1, key="st")
    slope = st.selectbox("ST Slope", [0, 1, 2],
                        format_func=lambda x: ["Up", "Flat", "Down"][x], key="slope")
    
    st.markdown("#### 💊 Clinical")
    cp = st.selectbox("Chest Pain", [0, 1, 2, 3],
                     format_func=lambda x: ["Typical", "Atypical", "Non-anginal", "Asymptomatic"][x], key="cp")
    exang = st.selectbox("Exercise Angina", [0, 1],
                        format_func=lambda x: "No" if x==0 else "Yes", key="exang")
    ca = st.selectbox("Vessels (0-4)", [0, 1, 2, 3, 4], key="ca")
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                       format_func=lambda x: ["Normal", "Fixed", "Reversible", "Unknown"][x], key="thal")

# PREDICT BUTTON
btn_col1, btn_col2 = st.columns([1, 1])
with btn_col1:
    predict_clicked = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

# COLUMN 3 - RESULTS
with results_col:
    st.markdown("#### 📊 Prediction Results")
    
    if predict_clicked:
        input_df = pd.DataFrame([{
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }])
        
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0, 1]
            
            st.session_state.prediction = prediction
            st.session_state.probability = probability
            st.session_state.input_df = input_df
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Display results
if hasattr(st.session_state, 'probability'):
    probability = st.session_state.probability
    prediction = st.session_state.prediction
    
    if probability < 0.3:
        risk_level, risk_color, risk_class, risk_icon = "LOW", "#2F80ED", "risk-low", "✅"
        recommendation = "Continue healthy lifestyle and regular check-ups."
    elif probability < 0.7:
        risk_level, risk_color, risk_class, risk_icon = "MEDIUM", "#F2994A", "risk-medium", "⚡"
        recommendation = "Enhanced monitoring recommended. Schedule follow-up with your doctor."
    else:
        risk_level, risk_color, risk_class, risk_icon = "HIGH", "#EB5757", "risk-high", "⚠️"
        recommendation = "Immediate medical consultation strongly advised. Contact a cardiologist."
    
    with results_col:
        st.markdown(f"""
        <div class="risk-box {risk_class}">
            <h2>{risk_icon} {risk_level} RISK</h2>
            <h3>Probability: {probability*100:.1f}%</h3>
            <p><strong>{'Disease Likely' if prediction==1 else 'No Disease Detected'}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        m1, m2 = st.columns(2)
        m1.metric("Risk", risk_level, f"{probability*100:.0f}%")
        m2.metric("Confidence", f"{max(probability, 1-probability)*100:.0f}%")
        
        m3, m4 = st.columns(2)
        m3.metric("Result", "Disease" if prediction==1 else "Healthy")
        m4.metric("Model", "LogReg")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk %", 'font': {'size': 11, 'color': '#C44569'}},
            number={'font': {'size': 22, 'color': '#C44569'}},
            gauge={
                'axis': {'range': [None, 100], 'tickfont': {'size': 9}},
                'bar': {'color': risk_color, 'thickness': 0.7},
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(47, 128, 237, 0.2)'},
                    {'range': [30, 70], 'color': 'rgba(242, 153, 74, 0.2)'},
                    {'range': [70, 100], 'color': 'rgba(235, 87, 87, 0.2)'}
                ],
                'threshold': {'line': {'color': "#EB5757", 'width': 3}, 'value': 50}
            }
        ))
        fig.update_layout(height=160, margin=dict(l=5, r=5, t=25, b=5), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"<p style='font-size: 0.9rem; margin-top: 0.3rem; color: #C44569;'><strong>💡 {recommendation}</strong></p>", unsafe_allow_html=True)

# EXPANDABLE SECTIONS
if hasattr(st.session_state, 'probability'):
    with col1:
        st.markdown("---")
        
        with st.expander("🔍 Risk Factors Analysis", expanded=False):
            risk_factors = []
            if age > 55: risk_factors.append(f"Age ({age} years)")
            if chol > 240: risk_factors.append(f"High cholesterol ({chol} mg/dL)")
            if trestbps > 140: risk_factors.append(f"Elevated blood pressure ({trestbps} mm Hg)")
            if fbs == 1: risk_factors.append("Fasting blood sugar > 120 mg/dL")
            if exang == 1: risk_factors.append("Exercise-induced angina present")
            if oldpeak > 2: risk_factors.append(f"Significant ST depression ({oldpeak})")
            if ca > 0: risk_factors.append(f"Coronary artery blockage ({ca} vessel(s))")
            
            if risk_factors:
                for rf in risk_factors:
                    st.markdown(f"• {rf}")
            else:
                st.success("✅ No major risk factors identified")
        
        with st.expander("📋 Patient Summary", expanded=False):
            st.markdown(f"""
            **Demographics:**
            - Age: {age} years
            - Sex: {'Male' if sex == 1 else 'Female'}
            
            **Vitals:**
            - Blood Pressure: {trestbps} mm Hg
            - Cholesterol: {chol} mg/dL
            - Max Heart Rate: {thalach} bpm
            
            **Test Results:**
            - Chest Pain: {['Typical Angina', 'Atypical Angina', 'Non-anginal', 'Asymptomatic'][cp]}
            - Resting ECG: {['Normal', 'ST-T Abnormality', 'LV Hypertrophy'][restecg]}
            - ST Depression: {oldpeak}
            - Major Vessels: {ca}
            - Thalassemia: {['Normal', 'Fixed Defect', 'Reversible Defect', 'Unknown'][thal]}
            """)
        
        with st.expander("📊 Statistical Insights", expanded=False):
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", "83.6%", help="Model accuracy")
            m1.metric("Dataset", "302 patients")
            m2.metric("ROC-AUC", "0.9154", help="Cross-validation")
            m2.metric("Validation", "5-Fold CV")
    
    st.markdown("---")
    
    report = f"""HEART DISEASE RISK ASSESSMENT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================

PATIENT DATA:
Age: {age} years | Sex: {'Male' if sex == 1 else 'Female'}
Blood Pressure: {trestbps} mm Hg
Cholesterol: {chol} mg/dL
Max Heart Rate: {thalach} bpm
ST Depression: {oldpeak}

PREDICTION:
Risk Level: {risk_level}
Probability: {probability*100:.1f}%
Classification: {'Disease' if prediction == 1 else 'No Disease'}
Confidence: {max(probability, 1-probability)*100:.1f}%

RECOMMENDATION:
{recommendation}

RISK FACTORS:
{chr(10).join('• ' + rf for rf in risk_factors) if risk_factors else '• None identified'}

================================================================
DISCLAIMER: Educational tool only. Not for medical diagnosis.
Model: Logistic Regression | ROC-AUC: 0.9154
"""
    
    st.download_button(
        "📄 Download Report",
        report,
        f"heart_risk_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85rem; padding: 0.5rem;'>
    <strong style='color: #C44569;'>Heart Disease Prediction System</strong><br>
    Logistic Regression | ROC-AUC: 0.9154<br>
    ⚠️ Educational tool - Not for medical diagnosis
</div>
""", unsafe_allow_html=True)