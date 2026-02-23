import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from io import BytesIO
import base64

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Animated gradient background */
.stApp {
    background: linear-gradient(-45deg, #FF6B9D, #C44569, #FFA502, #FF6348, #786FA6, #F8B500);
    background-size: 400% 400%;
    animation: gradientShift 14s ease infinite;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    25%  { background-position: 50% 100%; }
    50%  { background-position: 100% 50%; }
    75%  { background-position: 50% 0%; }
    100% { background-position: 0% 50%; }
}

/* Main card */
.main .block-container {
    background: rgba(255,255,255,0.93);
    border-radius: 22px;
    padding: 1.8rem 2.2rem;
    box-shadow: 0 12px 45px rgba(0,0,0,0.22);
    backdrop-filter: blur(18px);
    border: 2px solid rgba(255,255,255,0.35);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,107,157,0.18) 0%, rgba(255,255,255,0.97) 18%);
    backdrop-filter: blur(12px);
    border-right: 3px solid rgba(255,107,157,0.35);
}

/* Header */
.main-header {
    font-size: 2.6rem;
    background: linear-gradient(135deg, #FF6B9D 0%, #C44569 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    font-weight: 900;
    letter-spacing: -1px;
    animation: titlePulse 3s ease-in-out infinite;
}
@keyframes titlePulse {
    0%,100% { transform: scale(1); }
    50%      { transform: scale(1.015); }
}
.sub-header {
    font-size: 0.95rem; color: #555;
    text-align: center; margin-bottom: 0.4rem; font-weight: 600;
}

/* Section cards */
.section-card {
    background: linear-gradient(135deg, rgba(255,107,157,0.06) 0%, rgba(120,111,166,0.06) 100%);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    border: 1.5px solid rgba(255,107,157,0.18);
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 10px rgba(196,69,105,0.07);
}

/* Section headings */
.sec-title {
    font-size: 1rem; font-weight: 700;
    color: #C44569; margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 6px;
}

/* Risk boxes */
.risk-box {
    padding: 0.85rem 1rem;
    border-radius: 14px;
    margin: 0.5rem 0;
    border-left: 5px solid;
    animation: popIn 0.5s cubic-bezier(0.34,1.56,0.64,1);
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
}
@keyframes popIn {
    0%   { opacity: 0; transform: scale(0.85); }
    100% { opacity: 1; transform: scale(1); }
}
.risk-low    { background: linear-gradient(135deg,#43C6AC,#191654); border-color:#43C6AC; color:white; }
.risk-medium { background: linear-gradient(135deg,#F2994A,#F2C94C); border-color:#F2994A; color:white; }
.risk-high   { background: linear-gradient(135deg,#EB5757,#FF6B9D); border-color:#EB5757; color:white; }
.risk-box h2 { font-size:1.2rem!important; margin:0!important; color:white!important; }
.risk-box h3 { font-size:1rem!important; margin:0.2rem 0!important; color:white!important; }
.risk-box p  { font-size:0.85rem!important; margin:0!important; color:white!important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg,#F8F9FA,#E9ECEF);
    padding: 0.55rem; border-radius: 10px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.08);
    border-left: 3px solid #FF6B9D;
    transition: all 0.3s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.13);
}

/* Predict button */
.stButton>button {
    background: linear-gradient(135deg,#FF6B9D,#C44569);
    color: white; font-size: 1.05rem; font-weight: 700;
    padding: 0.65rem; border-radius: 12px; border: none;
    width: 100%; transition: all 0.35s;
    box-shadow: 0 5px 18px rgba(255,107,157,0.42);
    position: relative; overflow: hidden;
}
.stButton>button:before {
    content:''; position:absolute; top:0; left:-100%; width:100%; height:100%;
    background: linear-gradient(90deg,transparent,rgba(255,255,255,0.28),transparent);
    transition: left 0.45s;
}
.stButton>button:hover:before { left:100%; }
.stButton>button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 10px 28px rgba(255,107,157,0.58);
}

/* Download button special style */
.download-btn>button {
    background: linear-gradient(135deg,#11998e,#38ef7d) !important;
    box-shadow: 0 5px 18px rgba(17,153,142,0.42) !important;
}
.download-btn>button:hover {
    box-shadow: 0 10px 28px rgba(17,153,142,0.58) !important;
}

/* Progress bars */
.stProgress > div > div { background: linear-gradient(90deg,#FF6B9D,#C44569); }

/* Expander */
.streamlit-expanderHeader {
    background: linear-gradient(135deg,rgba(255,107,157,0.08),rgba(196,69,105,0.08));
    border-radius: 8px;
    border-left: 3px solid #FF6B9D;
}

/* Divider line */
hr { border-color: rgba(196,69,105,0.2) !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,107,157,0.06);
    border-radius: 10px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; font-weight: 600;
    color: #C44569;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#FF6B9D,#C44569) !important;
    color: white !important;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 2px 10px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-green  { background:#d4edda; color:#155724; }
.badge-yellow { background:#fff3cd; color:#856404; }
.badge-red    { background:#f8d7da; color:#721c24; }

/* Info card */
.info-card {
    background: white; border-radius: 12px;
    padding: 0.8rem 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    border-top: 3px solid #FF6B9D;
    margin: 0.4rem 0;
}
.info-card .label { font-size:0.76rem; color:#888; text-transform:uppercase; letter-spacing:0.8px; }
.info-card .value { font-size:1.1rem; font-weight:700; color:#2d2d2d; }

h2 { font-size:1.4rem!important; margin-top:0.4rem!important; margin-bottom:0.4rem!important; }
h3 { font-size:1.1rem!important; margin-top:0.3rem!important; margin-bottom:0.3rem!important; }
h4 { font-size:0.95rem!important; color:#C44569; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("heart_disease_model.joblib")
    except:
        return None

model = load_model()

# ── PDF generation (pure HTML → base64 for download) ─────────────────────
def generate_pdf_report(patient_data, prediction, probability, risk_level,
                         recommendation, risk_factors, clinical_notes):
    """Generate a styled HTML report that renders as a proper medical PDF."""

    sex_str   = "Male" if patient_data["sex"] == 1 else "Female"
    cp_str    = ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"][patient_data["cp"]]
    ecg_str   = ["Normal","ST-T Abnormality","Left Ventricular Hypertrophy"][patient_data["restecg"]]
    slope_str = ["Upsloping","Flat","Downsloping"][patient_data["slope"]]
    exang_str = "Yes" if patient_data["exang"] == 1 else "No"
    fbs_str   = "Yes (>120 mg/dL)" if patient_data["fbs"] == 1 else "No"
    thal_str  = ["Normal","Fixed Defect","Reversible Defect","Unknown"][patient_data["thal"]]

    risk_color_map = {"LOW": "#27AE60", "MEDIUM": "#F39C12", "HIGH": "#E74C3C"}
    risk_bg_map    = {"LOW": "#EAFAF1", "MEDIUM": "#FEF9E7", "HIGH": "#FDEDEC"}
    risk_color = risk_color_map.get(risk_level, "#333")
    risk_bg    = risk_bg_map.get(risk_level, "#fff")
    risk_icon  = {"LOW": "✓", "MEDIUM": "!", "HIGH": "⚠"}[risk_level]

    rf_html = "".join(f"<li>{rf}</li>" for rf in risk_factors) if risk_factors else "<li>No major risk factors identified</li>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; color: #2d2d2d; background: white; }}
  
  .header {{
    background: linear-gradient(135deg, #C44569, #FF6B9D);
    color: white; padding: 22px 32px;
    display: flex; align-items: center; gap: 16px;
  }}
  .header-title {{ font-size: 24px; font-weight: 900; letter-spacing: -0.5px; }}
  .header-sub   {{ font-size: 12px; opacity: 0.85; margin-top: 2px; }}
  .header-heart {{ font-size: 36px; }}
  .header-meta  {{ margin-left: auto; text-align: right; font-size: 11px; opacity: 0.85; }}

  .body {{ padding: 24px 32px; }}

  /* Risk banner */
  .risk-banner {{
    background: {risk_bg};
    border: 2px solid {risk_color};
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 22px;
    display: flex; align-items: center; gap: 20px;
  }}
  .risk-icon {{
    width: 52px; height: 52px; border-radius: 50%;
    background: {risk_color}; color: white;
    font-size: 22px; font-weight: 900;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }}
  .risk-title  {{ font-size: 20px; font-weight: 800; color: {risk_color}; }}
  .risk-prob   {{ font-size: 13px; color: #555; margin-top: 3px; }}
  .risk-result {{ font-size: 13px; font-weight: 600; margin-top: 2px; color: {risk_color}; }}
  .risk-rec    {{ margin-left: auto; background: white; border: 1px solid {risk_color};
                  border-radius: 8px; padding: 10px 14px; max-width: 280px; }}
  .risk-rec-title {{ font-size: 11px; font-weight: 700; text-transform: uppercase;
                     letter-spacing: 0.5px; color: {risk_color}; margin-bottom: 4px; }}
  .risk-rec-text  {{ font-size: 12px; color: #444; line-height: 1.45; }}

  /* Gauge bar */
  .gauge-bar-wrap {{ margin-bottom: 22px; }}
  .gauge-label {{ font-size: 11px; color: #888; text-transform: uppercase;
                  letter-spacing: 0.7px; margin-bottom: 6px; font-weight: 600; }}
  .gauge-track {{
    height: 16px; border-radius: 8px;
    background: linear-gradient(90deg, #27AE60 0%, #27AE60 30%, #F39C12 30%, #F39C12 70%, #E74C3C 70%, #E74C3C 100%);
    position: relative;
  }}
  .gauge-needle {{
    position: absolute; top: -4px;
    width: 4px; height: 24px; background: #1a1a1a;
    border-radius: 2px; left: {probability*100:.1f}%;
    transform: translateX(-50%);
  }}
  .gauge-ticks {{ display: flex; justify-content: space-between;
                  font-size: 10px; color: #888; margin-top: 4px; }}

  /* Grid sections */
  .section-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 20px; }}
  .section-grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; margin-bottom: 20px; }}

  .card {{
    background: #FAFAFA; border-radius: 10px;
    padding: 14px 16px; border: 1px solid #EBEBEB;
  }}
  .card-title {{
    font-size: 12px; font-weight: 700; color: #C44569;
    text-transform: uppercase; letter-spacing: 0.7px;
    margin-bottom: 10px; padding-bottom: 6px;
    border-bottom: 1px solid #F0E0E5;
  }}
  table.data-table {{ width: 100%; border-collapse: collapse; }}
  table.data-table td {{ padding: 4px 0; font-size: 12px; border-bottom: 1px solid #F5F5F5; }}
  table.data-table td:first-child {{ color: #888; width: 52%; }}
  table.data-table td:last-child  {{ font-weight: 600; color: #2d2d2d; }}

  /* Risk factors */
  .rf-list {{ list-style: none; padding: 0; }}
  .rf-list li {{ font-size: 12px; padding: 5px 0;
                 border-bottom: 1px solid #F5F5F5; display: flex; align-items: center; gap: 6px; }}
  .rf-list li::before {{ content: "⚠"; color: #E74C3C; font-size: 11px; flex-shrink: 0; }}
  .rf-ok {{ color: #27AE60; font-weight: 600; font-size: 12px; }}

  /* Clinical notes */
  .notes-card {{
    background: #FFF8E7; border-radius: 10px;
    padding: 14px 16px; border: 1px solid #F39C12;
    margin-bottom: 20px;
  }}
  .notes-title {{ font-size: 12px; font-weight: 700; color: #E67E22;
                  text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 8px; }}
  .notes-text   {{ font-size: 12px; color: #444; line-height: 1.6; white-space: pre-wrap; }}

  /* Model performance */
  .perf-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px; }}
  .perf-cell {{
    background: white; border-radius: 8px;
    padding: 10px; text-align: center;
    border: 1px solid #EBEBEB;
  }}
  .perf-value {{ font-size: 18px; font-weight: 800; color: #C44569; }}
  .perf-label {{ font-size: 10px; color: #888; text-transform: uppercase;
                 letter-spacing: 0.5px; margin-top: 2px; }}

  /* Footer */
  .footer {{
    background: #F8F8F8; border-top: 2px solid #EBEBEB;
    padding: 14px 32px;
    display: flex; justify-content: space-between; align-items: center;
  }}
  .footer-left  {{ font-size: 11px; color: #888; }}
  .footer-right {{ font-size: 11px; color: #C44569; font-weight: 600; }}
  .disclaimer {{
    background: #FFF3CD; border: 1px solid #FFEEBA;
    border-radius: 8px; padding: 10px 14px;
    font-size: 11px; color: #856404;
    margin-bottom: 18px; line-height: 1.5;
  }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-heart">❤️</div>
  <div>
    <div class="header-title">Heart Disease Risk Assessment</div>
    <div class="header-sub">Powered by Logistic Regression | ROC-AUC: 0.9154</div>
  </div>
  <div class="header-meta">
    Report ID: HDR-{datetime.now().strftime('%Y%m%d%H%M%S')}<br>
    Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}<br>
    Classification: EDUCATIONAL USE ONLY
  </div>
</div>

<div class="body">

<!-- DISCLAIMER -->
<div class="disclaimer">
  ⚠️ <strong>DISCLAIMER:</strong> This report is generated by an educational machine learning tool and is intended for informational and academic purposes only. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for any health concerns.
</div>

<!-- RISK BANNER -->
<div class="risk-banner">
  <div class="risk-icon">{risk_icon}</div>
  <div>
    <div class="risk-title">{risk_level} RISK</div>
    <div class="risk-prob">Predicted probability of heart disease: <strong>{probability*100:.1f}%</strong></div>
    <div class="risk-result">Classification: {'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'}</div>
  </div>
  <div class="risk-rec">
    <div class="risk-rec-title">📋 Clinical Recommendation</div>
    <div class="risk-rec-text">{recommendation}</div>
  </div>
</div>

<!-- RISK GAUGE BAR -->
<div class="gauge-bar-wrap">
  <div class="gauge-label">Risk Probability Gauge</div>
  <div class="gauge-track">
    <div class="gauge-needle"></div>
  </div>
  <div class="gauge-ticks"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
</div>

<!-- PATIENT DATA GRID -->
<div class="section-grid">

  <!-- Demographics & Vitals -->
  <div class="card">
    <div class="card-title">👤 Demographics &amp; Vitals</div>
    <table class="data-table">
      <tr><td>Age</td><td>{patient_data['age']} years</td></tr>
      <tr><td>Sex</td><td>{sex_str}</td></tr>
      <tr><td>Resting Blood Pressure</td><td>{patient_data['trestbps']} mm Hg</td></tr>
      <tr><td>Serum Cholesterol</td><td>{patient_data['chol']} mg/dL</td></tr>
      <tr><td>Maximum Heart Rate</td><td>{patient_data['thalach']} bpm</td></tr>
    </table>
  </div>

  <!-- Blood & ECG Tests -->
  <div class="card">
    <div class="card-title">🏥 Blood &amp; ECG Tests</div>
    <table class="data-table">
      <tr><td>Fasting Blood Sugar &gt; 120</td><td>{fbs_str}</td></tr>
      <tr><td>Resting ECG</td><td>{ecg_str}</td></tr>
      <tr><td>ST Depression (Oldpeak)</td><td>{patient_data['oldpeak']}</td></tr>
      <tr><td>ST Slope</td><td>{slope_str}</td></tr>
    </table>
  </div>

</div>

<!-- CLINICAL SECTION -->
<div class="section-grid">

  <div class="card">
    <div class="card-title">💊 Clinical Findings</div>
    <table class="data-table">
      <tr><td>Chest Pain Type</td><td>{cp_str}</td></tr>
      <tr><td>Exercise-Induced Angina</td><td>{exang_str}</td></tr>
      <tr><td>Major Vessels (Fluoroscopy)</td><td>{patient_data['ca']}</td></tr>
      <tr><td>Thalassemia Type</td><td>{thal_str}</td></tr>
    </table>
  </div>

  <!-- Risk Factors -->
  <div class="card">
    <div class="card-title">⚠️ Identified Risk Factors</div>
    { f'<ul class="rf-list">{rf_html}</ul>' if risk_factors else '<span class="rf-ok">✅ No major clinical risk factors identified from the entered data.</span>' }
  </div>

</div>

<!-- MODEL PERFORMANCE -->
<div class="perf-grid">
  <div class="perf-cell"><div class="perf-value">83.6%</div><div class="perf-label">Accuracy</div></div>
  <div class="perf-cell"><div class="perf-value">84.9%</div><div class="perf-label">Recall</div></div>
  <div class="perf-cell"><div class="perf-value">84.9%</div><div class="perf-label">F1-Score</div></div>
  <div class="perf-cell"><div class="perf-value">0.917</div><div class="perf-label">ROC-AUC</div></div>
</div>

<!-- CLINICAL NOTES -->
{"" if not clinical_notes.strip() else f'''
<div class="notes-card">
  <div class="notes-title">📝 Clinician Notes</div>
  <div class="notes-text">{clinical_notes}</div>
</div>
'''}

<!-- RECOMMENDATIONS TABLE -->
<div class="card" style="margin-bottom:20px;">
  <div class="card-title">📋 Detailed Recommendations by Risk Level</div>
  <table class="data-table">
    <tr><td style="color:#27AE60;font-weight:700;">LOW Risk (&lt;30%)</td><td>Maintain healthy diet, 150 min/week moderate exercise, annual check-ups, monitor BP and cholesterol.</td></tr>
    <tr><td style="color:#F39C12;font-weight:700;">MEDIUM Risk (30–70%)</td><td>Schedule GP appointment within 4 weeks. Consider lipid panel and ECG. Lifestyle modification: reduce saturated fat, increase physical activity.</td></tr>
    <tr><td style="color:#E74C3C;font-weight:700;">HIGH Risk (&gt;70%)</td><td>Seek urgent cardiology referral. Do not ignore symptoms. Comprehensive cardiac evaluation (stress test, echocardiogram) recommended.</td></tr>
  </table>
</div>

</div><!-- /body -->

<!-- FOOTER -->
<div class="footer">
  <div class="footer-left">
    Heart Disease Prediction System &nbsp;|&nbsp; Logistic Regression &nbsp;|&nbsp;
    Dataset: UCI Heart Disease (n=302 after deduplication) &nbsp;|&nbsp; 5-Fold CV
  </div>
  <div class="footer-right">⚠️ Educational Tool — Not for Clinical Diagnosis</div>
</div>

</body>
</html>"""
    return html


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image("https://img.icons8.com/color/96/000000/heart-with-pulse.png", width=72)
    except:
        st.markdown("# ❤️")

    st.markdown("### 🎯 Risk Levels")
    st.markdown("""
    <div class='risk-box risk-low' style='padding:0.5rem 0.8rem;margin:0.3rem 0;'>
      <p style='margin:0;font-weight:700;'>✅ LOW &nbsp; < 30%</p>
    </div>
    <div class='risk-box risk-medium' style='padding:0.5rem 0.8rem;margin:0.3rem 0;'>
      <p style='margin:0;font-weight:700;'>⚡ MEDIUM &nbsp; 30–70%</p>
    </div>
    <div class='risk-box risk-high' style='padding:0.5rem 0.8rem;margin:0.3rem 0;'>
      <p style='margin:0;font-weight:700;'>⚠️ HIGH &nbsp; > 70%</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info("**Logistic Regression**\nROC-AUC: **0.9154**\nAccuracy: **83.6%**\nRecall: **84.9%**\nDataset: **302 patients**")

    st.markdown("---")
    st.markdown("### 🩺 How to Use")
    st.markdown("""
    1. Fill in patient details across all **4 sections**
    2. Add any clinical notes
    3. Click **Predict Risk**
    4. Review results & download **PDF report**
    """)

    st.markdown("---")
    st.warning("⚠️ **Disclaimer**\nEducational tool only.\nNot for medical diagnosis.")

# ── Header ────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">❤️ Heart Disease Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Logistic Regression &nbsp;|&nbsp; ROC-AUC: 0.9154 &nbsp;|&nbsp; UCI Heart Disease Dataset &nbsp;|&nbsp; Educational Tool</p>', unsafe_allow_html=True)

st.markdown("---")

# ── PATIENT INFO ROW ──────────────────────────────────────────────────────
st.markdown("#### 🆔 Patient Identification (Optional)")
pid_c1, pid_c2, pid_c3 = st.columns(3)
patient_name = pid_c1.text_input("Patient Name / ID", value="", placeholder="e.g. Patient 001")
patient_dob  = pid_c2.text_input("Date of Birth", value="", placeholder="DD/MM/YYYY")
patient_ref  = pid_c3.text_input("Referring Clinician", value="", placeholder="Dr. ...")

st.markdown("---")

# ── FOUR COLUMN LAYOUT ────────────────────────────────────────────────────
st.markdown("### 📋 Clinical Input — Enter Patient Data")

col1, col2, col3, col4 = st.columns(4)

# ── COLUMN 1: Demographics & Vitals ──────────────────────────────────────
with col1:
    st.markdown("""<div class='sec-title'>👤 Demographics & Vitals</div>""", unsafe_allow_html=True)
    with st.container():
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, key="age",
                               help="Patient's age in years")

        # Age risk indicator
        if age < 45:
            st.markdown("<span class='badge badge-green'>Low age risk</span>", unsafe_allow_html=True)
        elif age < 60:
            st.markdown("<span class='badge badge-yellow'>Moderate age risk</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge badge-red'>Elevated age risk</span>", unsafe_allow_html=True)

        sex = st.selectbox("Sex", [0, 1],
                           format_func=lambda x: "♀ Female" if x == 0 else "♂ Male",
                           key="sex", help="Biological sex of the patient")

        trestbps = st.number_input("Resting BP (mm Hg)", min_value=50, max_value=250, value=120,
                                    key="bp", help="Resting blood pressure at hospital admission")
        # BP badge
        if trestbps < 120:
            st.markdown("<span class='badge badge-green'>Normal BP</span>", unsafe_allow_html=True)
        elif trestbps < 140:
            st.markdown("<span class='badge badge-yellow'>Elevated BP</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge badge-red'>High BP</span>", unsafe_allow_html=True)

        chol = st.number_input("Cholesterol (mg/dL)", min_value=50, max_value=600, value=200,
                                key="chol", help="Serum cholesterol level")
        # Cholesterol badge
        if chol < 200:
            st.markdown("<span class='badge badge-green'>Desirable</span>", unsafe_allow_html=True)
        elif chol < 240:
            st.markdown("<span class='badge badge-yellow'>Borderline high</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge badge-red'>High cholesterol</span>", unsafe_allow_html=True)

        thalach = st.number_input("Max Heart Rate (bpm)", min_value=50, max_value=250, value=150,
                                   key="hr", help="Maximum heart rate achieved during exercise test")
        # HR badge
        hr_max_est = 220 - age
        hr_pct = int(thalach / hr_max_est * 100) if hr_max_est > 0 else 0
        st.markdown(f"<span class='badge badge-{'green' if hr_pct >= 85 else 'yellow' if hr_pct >= 70 else 'red'}'>{hr_pct}% of age-predicted max</span>", unsafe_allow_html=True)

# ── COLUMN 2: Blood & ECG ─────────────────────────────────────────────────
with col2:
    st.markdown("""<div class='sec-title'>🏥 Blood & ECG Tests</div>""", unsafe_allow_html=True)

    fbs = st.selectbox("Fasting Blood Sugar > 120",
                        [0, 1], format_func=lambda x: "No" if x == 0 else "Yes (>120 mg/dL)",
                        key="fbs", help="Is fasting blood sugar greater than 120 mg/dL?")
    if fbs == 1:
        st.markdown("<span class='badge badge-red'>Elevated fasting glucose</span>", unsafe_allow_html=True)

    restecg = st.selectbox("Resting ECG Result", [0, 1, 2],
                            format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x],
                            key="ecg", help="Results of resting electrocardiographic measurement")
    if restecg > 0:
        st.markdown("<span class='badge badge-yellow'>ECG abnormality noted</span>", unsafe_allow_html=True)

    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0,
                               value=1.0, step=0.1, key="st",
                               help="ST depression induced by exercise relative to rest")
    st.progress(min(oldpeak / 6.0, 1.0))
    if oldpeak == 0:
        st.markdown("<span class='badge badge-green'>No ST depression</span>", unsafe_allow_html=True)
    elif oldpeak <= 2:
        st.markdown("<span class='badge badge-yellow'>Mild ST depression</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge badge-red'>Significant ST depression</span>", unsafe_allow_html=True)

    slope = st.selectbox("ST Slope", [0, 1, 2],
                          format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
                          key="slope", help="Slope of the peak exercise ST segment")
    slope_info = ["Generally favourable", "Intermediate significance", "Associated with ischaemia"][slope]
    badge_color = ["green", "yellow", "red"][slope]
    st.markdown(f"<span class='badge badge-{badge_color}'>{slope_info}</span>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📈 Quick ECG Summary**")
    ecg_score = (1 if restecg > 0 else 0) + (1 if oldpeak > 2 else 0) + (1 if slope == 2 else 0)
    st.progress(ecg_score / 3)
    st.caption(f"ECG risk indicators: {ecg_score}/3")

# ── COLUMN 3: Clinical ────────────────────────────────────────────────────
with col3:
    st.markdown("""<div class='sec-title'>💊 Clinical Findings</div>""", unsafe_allow_html=True)

    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                       format_func=lambda x: ["Typical Angina", "Atypical Angina",
                                               "Non-anginal Pain", "Asymptomatic"][x],
                       key="cp", help="Type of chest pain experienced")
    cp_risk = ["⚠️ Classic cardiac symptom", "Possibly cardiac-related",
               "Often non-cardiac", "Silent — requires careful evaluation"][cp]
    cp_badge = ["red", "yellow", "green", "yellow"][cp]
    st.markdown(f"<span class='badge badge-{cp_badge}'>{cp_risk}</span>", unsafe_allow_html=True)

    exang = st.selectbox("Exercise-Induced Angina", [0, 1],
                          format_func=lambda x: "No" if x == 0 else "Yes",
                          key="exang", help="Angina induced by exercise")
    if exang == 1:
        st.markdown("<span class='badge badge-red'>Positive exercise angina</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge badge-green'>No exercise angina</span>", unsafe_allow_html=True)

    ca = st.selectbox("Major Vessels (Fluoroscopy)", [0, 1, 2, 3, 4],
                       key="ca", help="Number of major vessels coloured by fluoroscopy (0–4)")
    ca_badge = ["green", "yellow", "red", "red", "red"][ca]
    ca_text  = ["No blockage", "1 vessel blocked", "2 vessels blocked",
                "3 vessels blocked", "4 vessels blocked"][ca]
    st.markdown(f"<span class='badge badge-{ca_badge}'>{ca_text}</span>", unsafe_allow_html=True)

    thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                         format_func=lambda x: ["Normal", "Fixed Defect",
                                                "Reversible Defect", "Unknown"][x],
                         key="thal", help="Thalassemia type from nuclear stress test")
    thal_risk = ["Normal perfusion", "Permanent defect", "Ischaemic pattern — significant", "Undetermined"][thal]
    thal_badge = ["green", "yellow", "red", "yellow"][thal]
    st.markdown(f"<span class='badge badge-{thal_badge}'>{thal_risk}</span>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # Clinical risk score (simple sum)
    clin_score = (1 if ca > 0 else 0) + (1 if exang == 1 else 0) + (1 if cp == 0 else 0) + (1 if thal == 2 else 0)
    st.markdown("**🩺 Clinical Risk Score**")
    st.progress(clin_score / 4)
    st.caption(f"Clinical risk indicators: {clin_score}/4")

# ── COLUMN 4: Results & Notes ─────────────────────────────────────────────
with col4:
    st.markdown("""<div class='sec-title'>📊 Prediction Results</div>""", unsafe_allow_html=True)

    # Clinical notes input
    clinical_notes = st.text_area(
        "📝 Clinician Notes",
        placeholder="Enter any additional clinical observations, history, medications, or notes to include in the PDF report...",
        height=110, key="notes"
    )

    # Predict button
    predict_clicked = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

    # Results display
    if predict_clicked:
        input_df = pd.DataFrame([{
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }])
        if model is not None:
            try:
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0, 1]
                st.session_state.prediction   = int(prediction)
                st.session_state.probability  = float(probability)
                st.session_state.input_data   = {
                    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
                    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
                    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
                }
                st.session_state.clinical_notes = clinical_notes
                st.session_state.patient_name   = patient_name
                st.session_state.patient_dob    = patient_dob
                st.session_state.patient_ref    = patient_ref
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        else:
            # Demo mode when no model file is loaded
            import random
            st.session_state.probability  = round(random.uniform(0.25, 0.75), 3)
            st.session_state.prediction   = 1 if st.session_state.probability > 0.5 else 0
            st.session_state.input_data   = {
                'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
                'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
                'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
            }
            st.session_state.clinical_notes = clinical_notes
            st.session_state.patient_name   = patient_name
            st.session_state.patient_dob    = patient_dob
            st.session_state.patient_ref    = patient_ref
            st.warning("⚠️ Demo mode — model file not found. Showing simulated result.")

    if hasattr(st.session_state, 'probability'):
        prob       = st.session_state.probability
        pred       = st.session_state.prediction
        risk_level = "LOW" if prob < 0.3 else "MEDIUM" if prob < 0.7 else "HIGH"
        risk_class = {"LOW":"risk-low","MEDIUM":"risk-medium","HIGH":"risk-high"}[risk_level]
        risk_icon  = {"LOW":"✅","MEDIUM":"⚡","HIGH":"⚠️"}[risk_level]
        recommendation = {
            "LOW":    "Continue healthy lifestyle and regular annual check-ups.",
            "MEDIUM": "Enhanced monitoring recommended. Schedule follow-up with your GP within 4 weeks.",
            "HIGH":   "Immediate cardiology consultation strongly advised. Do not delay."
        }[risk_level]
        risk_color = {"LOW":"#27AE60","MEDIUM":"#F39C12","HIGH":"#E74C3C"}[risk_level]

        st.markdown(f"""
        <div class="risk-box {risk_class}">
            <h2>{risk_icon} {risk_level} RISK</h2>
            <h3>Probability: {prob*100:.1f}%</h3>
            <p>{'❤️ Heart Disease Detected' if pred == 1 else '💚 No Disease Detected'}</p>
        </div>
        """, unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("Confidence", f"{max(prob, 1-prob)*100:.0f}%")
        m2.metric("Model", "LogReg")

        # Compact gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Disease Risk %", 'font': {'size': 10, 'color': '#C44569'}},
            number={'font': {'size': 20, 'color': risk_color}, 'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100], 'tickfont': {'size': 8}},
                'bar': {'color': risk_color, 'thickness': 0.72},
                'steps': [
                    {'range': [0, 30],  'color': 'rgba(39,174,96,0.15)'},
                    {'range': [30, 70], 'color': 'rgba(243,156,18,0.15)'},
                    {'range': [70, 100],'color': 'rgba(231,76,60,0.15)'}
                ],
                'threshold': {'line': {'color': "#C44569", 'width': 3}, 'value': 50}
            }
        ))
        fig.update_layout(height=155, margin=dict(l=5,r=5,t=28,b=5), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"<p style='font-size:0.82rem;color:{risk_color};font-weight:700;margin-top:-6px;'>💡 {recommendation}</p>",
                    unsafe_allow_html=True)

# ── RESULTS DETAIL TABS (below 4 columns) ────────────────────────────────
if hasattr(st.session_state, 'probability'):
    st.markdown("---")
    prob       = st.session_state.probability
    pred       = st.session_state.prediction
    risk_level = "LOW" if prob < 0.3 else "MEDIUM" if prob < 0.7 else "HIGH"
    risk_color = {"LOW":"#27AE60","MEDIUM":"#F39C12","HIGH":"#E74C3C"}[risk_level]
    recommendation = {
        "LOW":    "Continue healthy lifestyle and regular annual check-ups.",
        "MEDIUM": "Enhanced monitoring recommended. Schedule follow-up with your GP within 4 weeks.",
        "HIGH":   "Immediate cardiology consultation strongly advised. Do not delay."
    }[risk_level]
    d = st.session_state.input_data

    # Risk factors
    risk_factors = []
    if d['age'] > 55:      risk_factors.append(f"Age {d['age']} yrs — elevated risk above 55")
    if d['chol'] > 240:    risk_factors.append(f"Cholesterol {d['chol']} mg/dL — high (>240)")
    if d['trestbps'] > 140:risk_factors.append(f"Blood pressure {d['trestbps']} mm Hg — hypertensive range")
    if d['fbs'] == 1:      risk_factors.append("Fasting blood sugar > 120 mg/dL — possible diabetes indicator")
    if d['exang'] == 1:    risk_factors.append("Exercise-induced angina — significant cardiac indicator")
    if d['oldpeak'] > 2:   risk_factors.append(f"ST depression {d['oldpeak']} — significant ischaemic marker")
    if d['ca'] > 0:        risk_factors.append(f"{d['ca']} major vessel(s) blocked — coronary artery disease indicator")
    if d['thal'] == 2:     risk_factors.append("Reversible thalassemia defect — ischaemic pattern")
    if d['cp'] == 0:       risk_factors.append("Typical angina chest pain — classic cardiac presentation")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Risk Analysis", "📋 Patient Summary", "📊 Model Performance", "📄 Download PDF Report"
    ])

    with tab1:
        t1c1, t1c2 = st.columns(2)
        with t1c1:
            st.markdown("#### ⚠️ Identified Risk Factors")
            if risk_factors:
                for rf in risk_factors:
                    st.markdown(f"🔸 {rf}")
            else:
                st.success("✅ No major clinical risk factors identified from the entered data.")

        with t1c2:
            st.markdown("#### 💡 Clinical Recommendations")
            if risk_level == "LOW":
                st.success(f"**{risk_level} RISK**")
                st.markdown("""
                - ✅ Maintain regular exercise (150+ min/week moderate intensity)
                - ✅ Heart-healthy diet (Mediterranean or DASH diet)
                - ✅ Annual cardiovascular screening
                - ✅ Monitor blood pressure and cholesterol regularly
                - ✅ Avoid smoking, limit alcohol
                """)
            elif risk_level == "MEDIUM":
                st.warning(f"**{risk_level} RISK**")
                st.markdown("""
                - ⚡ Schedule GP appointment within **4 weeks**
                - ⚡ Request fasting lipid panel and full blood count
                - ⚡ Consider resting and exercise ECG
                - ⚡ Lifestyle modification: reduce saturated fat, increase activity
                - ⚡ Monitor blood pressure twice weekly at home
                - ⚡ Discuss medication review with your doctor
                """)
            else:
                st.error(f"**{risk_level} RISK**")
                st.markdown("""
                - 🚨 Seek **immediate** cardiology referral
                - 🚨 Do not ignore symptoms (chest pain, shortness of breath, dizziness)
                - 🚨 Comprehensive cardiac evaluation: stress test, echocardiogram
                - 🚨 Review all medications with a cardiologist
                - 🚨 Avoid strenuous activity until evaluated
                - 🚨 Consider ambulatory blood pressure monitoring
                """)

            # Risk probability bar chart
            fig2 = go.Figure(go.Bar(
                x=["No Disease", "Heart Disease"],
                y=[(1-prob)*100, prob*100],
                marker_color=["#27AE60" if pred==0 else "#CCC", risk_color],
                text=[f"{(1-prob)*100:.1f}%", f"{prob*100:.1f}%"],
                textposition="auto"
            ))
            fig2.update_layout(
                title="Predicted Class Probabilities",
                height=220, margin=dict(l=10,r=10,t=36,b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis_title="Probability (%)", yaxis_range=[0,100],
                showlegend=False, font=dict(size=11)
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        t2c1, t2c2, t2c3 = st.columns(3)
        with t2c1:
            st.markdown("**👤 Demographics**")
            st.markdown(f"- Age: **{d['age']} years**")
            st.markdown(f"- Sex: **{'Male' if d['sex']==1 else 'Female'}**")
            if patient_name: st.markdown(f"- Name/ID: **{patient_name}**")
            if patient_dob:  st.markdown(f"- DOB: **{patient_dob}**")
            if patient_ref:  st.markdown(f"- Clinician: **{patient_ref}**")

            st.markdown("**💓 Vitals**")
            st.markdown(f"- Resting BP: **{d['trestbps']} mm Hg**")
            st.markdown(f"- Cholesterol: **{d['chol']} mg/dL**")
            st.markdown(f"- Max Heart Rate: **{d['thalach']} bpm**")

        with t2c2:
            st.markdown("**🏥 Tests**")
            st.markdown(f"- Fasting Blood Sugar: **{'Yes' if d['fbs']==1 else 'No'}**")
            st.markdown(f"- Resting ECG: **{['Normal','ST-T Abnormality','LV Hypertrophy'][d['restecg']]}**")
            st.markdown(f"- ST Depression: **{d['oldpeak']}**")
            st.markdown(f"- ST Slope: **{['Upsloping','Flat','Downsloping'][d['slope']]}**")

            st.markdown("**💊 Clinical**")
            st.markdown(f"- Chest Pain: **{['Typical Angina','Atypical Angina','Non-anginal','Asymptomatic'][d['cp']]}**")
            st.markdown(f"- Exercise Angina: **{'Yes' if d['exang']==1 else 'No'}**")
            st.markdown(f"- Vessels: **{d['ca']}**")
            st.markdown(f"- Thalassemia: **{['Normal','Fixed Defect','Reversible Defect','Unknown'][d['thal']]}**")

        with t2c3:
            st.markdown("**🎯 Prediction**")
            st.metric("Risk Level", risk_level)
            st.metric("Probability", f"{prob*100:.1f}%")
            st.metric("Classification", "Disease" if pred==1 else "No Disease")
            st.metric("Confidence", f"{max(prob,1-prob)*100:.0f}%")
            if st.session_state.get('clinical_notes','').strip():
                st.markdown("**📝 Notes**")
                st.info(st.session_state.clinical_notes)

    with tab3:
        t3c1, t3c2 = st.columns(2)
        with t3c1:
            st.markdown("#### Model Evaluation Metrics")
            metrics_df = pd.DataFrame({
                "Metric":    ["Accuracy","Recall (Sensitivity)","F1-Score","ROC-AUC","Specificity","Precision"],
                "Score":     ["83.61%","84.85%","84.85%","91.67%","82.14%","84.85%"],
                "Interpretation": [
                    "Overall correct predictions",
                    "True heart disease cases caught",
                    "Balance of precision & recall",
                    "Overall discriminative ability",
                    "True healthy patients identified",
                    "Of predicted disease, correctly flagged"
                ]
            })
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        with t3c2:
            st.markdown("#### Confusion Matrix (Test Set, n=61)")
            cm_fig = go.Figure(data=go.Heatmap(
                z=[[23, 5],[5, 28]],
                x=["Predicted: No Disease","Predicted: Disease"],
                y=["Actual: No Disease","Actual: Disease"],
                colorscale=[[0,"#FFFFFF"],[1,"#C44569"]],
                text=[[23,5],[5,28]], texttemplate="%{text}",
                textfont={"size":18, "color":"#1a1a1a"},
                showscale=False
            ))
            cm_fig.update_layout(height=220, margin=dict(l=10,r=10,t=20,b=10),
                                  paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(cm_fig, use_container_width=True)

        st.markdown("#### Dataset & Training Summary")
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Original Dataset Size | 1,025 records |
        | After Deduplication   | 302 unique records |
        | Duplicates Removed    | 723 records (70.5%) |
        | Training Set (80%)    | 241 records |
        | Test Set (20%)        | 61 records |
        | Cross-Validation      | 5-Fold Stratified CV |
        | Best C Parameter      | 0.1 (L2 regularisation) |
        | Best CV ROC-AUC       | 91.54% |
        """)

    with tab4:
        st.markdown("#### 📄 Generate & Download PDF Report")
        st.info("The PDF report includes: patient data, risk assessment, gauge chart, identified risk factors, detailed recommendations, model performance metrics, and your clinical notes.")

        col_pdf1, col_pdf2 = st.columns(2)
        with col_pdf1:
            include_notes     = st.checkbox("Include clinician notes", value=True)
            include_recs      = st.checkbox("Include recommendations table", value=True)
        with col_pdf2:
            include_perf      = st.checkbox("Include model performance", value=True)
            st.markdown(f"**Report timestamp:** {datetime.now().strftime('%d %b %Y, %H:%M')}")

        notes_for_pdf = st.session_state.get('clinical_notes','') if include_notes else ""
        d = st.session_state.input_data
        rf_for_pdf = risk_factors

        # Generate HTML report
        html_report = generate_pdf_report(
            patient_data   = d,
            prediction     = pred,
            probability    = prob,
            risk_level     = risk_level,
            recommendation = recommendation,
            risk_factors   = rf_for_pdf,
            clinical_notes = notes_for_pdf
        )

        # Encode as base64 for download
        b64 = base64.b64encode(html_report.encode()).decode()
        filename = f"HeartRisk_{patient_name.replace(' ','_') or 'Patient'}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"

        # HTML download (opens directly in browser as PDF-quality report)
        st.markdown(f"""
        <div class="download-btn">
        <a href="data:text/html;base64,{b64}" download="{filename}" style="text-decoration:none;">
            <button style="
                width:100%; background:linear-gradient(135deg,#11998e,#38ef7d);
                color:white; font-size:1.05rem; font-weight:700;
                padding:0.65rem 1rem; border-radius:12px; border:none;
                cursor:pointer; box-shadow:0 5px 18px rgba(17,153,142,0.42);
                transition:all 0.3s;
            ">
                📄 Download Report (HTML — opens & prints as PDF)
            </button>
        </a>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**📋 Plain Text Summary (Copy/Paste)**")
        sex_str_t   = "Male" if d['sex'] == 1 else "Female"
        cp_str_t    = ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"][d['cp']]
        ecg_str_t   = ["Normal","ST-T Abnormality","Left Ventricular Hypertrophy"][d['restecg']]
        slope_str_t = ["Upsloping","Flat","Downsloping"][d['slope']]
        fbs_str_t   = "Yes (>120 mg/dL)" if d['fbs'] == 1 else "No"
        thal_str_t  = ["Normal","Fixed Defect","Reversible Defect","Unknown"][d['thal']]

        plain_report = f"""HEART DISEASE RISK ASSESSMENT REPORT
Generated: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}
Report ID: HDR-{datetime.now().strftime('%Y%m%d%H%M%S')}
{'Patient: ' + patient_name if patient_name else ''}
{'DOB: ' + patient_dob if patient_dob else ''}
{'Clinician: ' + patient_ref if patient_ref else ''}
{'='*60}

PREDICTION RESULT
━━━━━━━━━━━━━━━━
Risk Level:      {risk_level}
Probability:     {prob*100:.1f}%
Classification:  {'Heart Disease Detected' if pred==1 else 'No Heart Disease Detected'}
Confidence:      {max(prob,1-prob)*100:.0f}%

RECOMMENDATION
━━━━━━━━━━━━━━
{recommendation}

PATIENT DATA
━━━━━━━━━━━━
Demographics:
  Age:              {d['age']} years
  Sex:              {sex_str_t}
  
Vitals:
  Resting BP:       {d['trestbps']} mm Hg
  Cholesterol:      {d['chol']} mg/dL
  Max Heart Rate:   {d['thalach']} bpm

Tests:
  Fasting Sugar>120:{fbs_str_t}
  Resting ECG:      {ecg_str_t}
  ST Depression:    {d['oldpeak']}
  ST Slope:         {slope_str_t}

Clinical:
  Chest Pain:       {cp_str_t}
  Exercise Angina:  {'Yes' if d['exang']==1 else 'No'}
  Major Vessels:    {d['ca']}
  Thalassemia:      {thal_str_t}

RISK FACTORS IDENTIFIED
━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join('• ' + rf for rf in rf_for_pdf) if rf_for_pdf else '• None identified'}

{'CLINICAL NOTES' + chr(10) + '━'*22 + chr(10) + notes_for_pdf if notes_for_pdf.strip() else ''}

MODEL PERFORMANCE
━━━━━━━━━━━━━━━━━
Accuracy:    83.61% | Recall: 84.85% | F1: 84.85% | ROC-AUC: 91.67%
Dataset:     302 unique patients (UCI Heart Disease, after deduplication)
Validation:  5-Fold Stratified Cross-Validation

{'='*60}
⚠️  DISCLAIMER: This report is generated by an educational ML tool
and does NOT constitute medical advice or diagnosis.
Always consult a qualified healthcare professional.
Model: Logistic Regression | UCI Heart Disease Dataset
"""
        st.text_area("Plain text report", plain_report, height=220)
        st.download_button(
            "📥 Download Plain Text (.txt)",
            plain_report,
            f"HeartRisk_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            use_container_width=True
        )

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;font-size:0.83rem;padding:0.5rem;line-height:1.8;'>
    <strong style='color:#C44569;font-size:0.95rem;'>❤️ Heart Disease Risk Predictor</strong><br>
    Logistic Regression &nbsp;|&nbsp; ROC-AUC: 0.9154 &nbsp;|&nbsp; UCI Heart Disease Dataset (n=302)<br>
    5-Fold Cross-Validation &nbsp;|&nbsp; Accuracy: 83.6% &nbsp;|&nbsp; Recall: 84.9%<br>
    <span style='color:#E74C3C;font-weight:600;'>⚠️ Educational & Research Tool — Not for Clinical Diagnosis</span>
</div>
""", unsafe_allow_html=True)
