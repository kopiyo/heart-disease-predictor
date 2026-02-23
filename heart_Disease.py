import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import base64

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* ── Animated background ── */
.stApp {
    background: linear-gradient(-45deg,#FF6B9D,#C44569,#FFA502,#FF6348,#786FA6,#F8B500);
    background-size: 400% 400%;
    animation: gradientShift 14s ease infinite;
}
@keyframes gradientShift {
    0%{background-position:0% 50%} 25%{background-position:50% 100%}
    50%{background-position:100% 50%} 75%{background-position:50% 0%}
    100%{background-position:0% 50%}
}

/* ── Main container: tight padding, no scroll ── */
.main .block-container {
    background: rgba(255,255,255,0.94);
    border-radius: 18px;
    padding: 0.7rem 1.2rem 0.5rem 1.2rem !important;
    max-width: 100% !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
}

/* ── Typography: smaller everywhere ── */
h1,h2,h3,h4,p,label,div { font-family:'Segoe UI',Arial,sans-serif !important; }
h1 { font-size:1.5rem !important; margin:0 0 0.1rem 0 !important; }
h4 { font-size:0.78rem !important; margin:0.25rem 0 0.1rem 0 !important;
     color:#C44569 !important; font-weight:700 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu,footer,[data-testid="stToolbar"]{display:none!important;}
.stDeployButton{display:none!important;}
section[data-testid="stSidebar"]{display:none!important;}

/* ── Shrink ALL input widgets ── */
.stNumberInput input {
    font-size:0.78rem !important; padding:2px 6px !important;
    height:28px !important; border-radius:6px !important;
}
.stNumberInput [data-testid="stNumberInputStepDown"],
.stNumberInput [data-testid="stNumberInputStepUp"] {
    width:22px !important; height:28px !important; font-size:0.75rem !important;
}
.stSelectbox > div > div {
    font-size:0.78rem !important; min-height:28px !important;
    padding:2px 8px !important; border-radius:6px !important;
}
.stTextArea textarea {
    font-size:0.75rem !important; padding:4px 6px !important;
    border-radius:6px !important; resize:none !important;
}
.stTextInput input {
    font-size:0.75rem !important; padding:2px 6px !important;
    height:26px !important; border-radius:6px !important;
}

/* ── Remove label padding ── */
.stNumberInput label, .stSelectbox label,
.stTextArea label, .stTextInput label {
    font-size:0.72rem !important; margin-bottom:1px !important;
    padding:0 !important; color:#555 !important; font-weight:600 !important;
}

/* ── Compact widget spacing ── */
[data-testid="stVerticalBlock"] > div { gap:0 !important; }
.element-container { margin-bottom:0 !important; padding-bottom:0 !important; }
div[data-testid="column"] { padding:0 3px !important; }

/* ── Section header pill ── */
.sec-head {
    font-size:0.72rem; font-weight:700; color:white;
    background:linear-gradient(135deg,#C44569,#FF6B9D);
    border-radius:6px; padding:2px 8px; display:inline-block;
    margin-bottom:4px; letter-spacing:0.3px;
}

/* ── Risk result card ── */
.risk-card {
    border-radius:10px; padding:8px 10px; margin:4px 0;
    border-left:4px solid; animation:popIn 0.4s ease;
}
@keyframes popIn{0%{opacity:0;transform:scale(0.9)}100%{opacity:1;transform:scale(1)}}
.r-low    {background:linear-gradient(135deg,#43C6AC,#191654);border-color:#43C6AC;}
.r-medium {background:linear-gradient(135deg,#F2994A,#F2C94C);border-color:#F2994A;}
.r-high   {background:linear-gradient(135deg,#EB5757,#FF6B9D);border-color:#EB5757;}
.risk-card *{color:white !important;}
.risk-card .rt{font-size:1rem;font-weight:800;margin:0;}
.risk-card .rp{font-size:0.82rem;margin:1px 0 0 0;}
.risk-card .rr{font-size:0.72rem;margin:1px 0 0 0;opacity:0.9;}

/* ── Predict button ── */
.stButton > button {
    background:linear-gradient(135deg,#FF6B9D,#C44569) !important;
    color:white !important; font-weight:700 !important;
    font-size:0.85rem !important; padding:5px 0 !important;
    border-radius:8px !important; border:none !important; width:100% !important;
    box-shadow:0 4px 14px rgba(255,107,157,0.45) !important;
    transition:all 0.3s !important; height:34px !important;
}
.stButton > button:hover {
    transform:translateY(-1px) !important;
    box-shadow:0 6px 20px rgba(255,107,157,0.6) !important;
}

/* ── Download button ── */
.dl-btn a button {
    background:linear-gradient(135deg,#11998e,#38ef7d) !important;
    color:white !important;
}

/* ── Metric boxes ── */
[data-testid="metric-container"] {
    background:linear-gradient(135deg,#fafafa,#f0f0f0) !important;
    border-radius:8px !important; padding:4px 8px !important;
    border-left:3px solid #FF6B9D !important;
    box-shadow:0 2px 6px rgba(0,0,0,0.07) !important;
}
[data-testid="metric-container"] label{font-size:0.65rem !important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-size:0.9rem !important;}

/* ── Divider ── */
hr{margin:4px 0 !important;border-color:rgba(196,69,105,0.18) !important;}

/* ── Info badge ── */
.badge {
    display:inline-block; font-size:0.65rem; font-weight:700;
    padding:1px 7px; border-radius:10px; margin-top:1px;
}
.bg{background:#d4edda;color:#155724;}
.by{background:#fff3cd;color:#856404;}
.br{background:#f8d7da;color:#721c24;}

/* ── Disclaimer ── */
.disc {
    font-size:0.62rem; color:#999; text-align:center;
    padding:2px 0; border-top:1px solid rgba(196,69,105,0.15); margin-top:3px;
}

/* ── Panel card ── */
.panel {
    background:rgba(255,107,157,0.04);
    border:1px solid rgba(255,107,157,0.15);
    border-radius:10px; padding:6px 8px; height:100%;
}

/* ── Scrollable results area ── */
.results-inner{max-height:82vh;overflow-y:auto;padding-right:4px;}
</style>
""", unsafe_allow_html=True)

# ── Model loader ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("heart_disease_model.joblib")
    except:
        return None

model = load_model()

# ── PDF HTML generator ────────────────────────────────────────────────────
def make_report(d, pred, prob, risk, rec, rfs, notes, pname, pdob, pref):
    rc = {"LOW":"#27AE60","MEDIUM":"#F39C12","HIGH":"#E74C3C"}[risk]
    ri = {"LOW":"✓","MEDIUM":"!","HIGH":"⚠"}[risk]
    rb = {"LOW":"#EAFAF1","MEDIUM":"#FEF9E7","HIGH":"#FDEDEC"}[risk]
    rf_html = "".join(f"<li>{x}</li>" for x in rfs) if rfs else "<li>No major risk factors identified</li>"
    sex_s   = "Male" if d["sex"]==1 else "Female"
    cp_s    = ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"][d["cp"]]
    ecg_s   = ["Normal","ST-T Abnormality","Left Ventricular Hypertrophy"][d["restecg"]]
    sl_s    = ["Upsloping","Flat","Downsloping"][d["slope"]]
    th_s    = ["Normal","Fixed Defect","Reversible Defect","Unknown"][d["thal"]]
    fbs_s   = "Yes (>120 mg/dL)" if d["fbs"]==1 else "No"
    ex_s    = "Yes" if d["exang"]==1 else "No"
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:'Segoe UI',Arial,sans-serif;}}
body{{font-size:12px;color:#222;background:white;}}
.hdr{{background:linear-gradient(135deg,#C44569,#FF6B9D);color:white;padding:18px 28px;
      display:flex;align-items:center;gap:14px;}}
.hdr-t{{font-size:20px;font-weight:900;}}.hdr-s{{font-size:11px;opacity:0.85;margin-top:2px;}}
.hdr-m{{margin-left:auto;text-align:right;font-size:10px;opacity:0.85;}}
.body{{padding:18px 28px;}}
.disc{{background:#FFF3CD;border:1px solid #FFEEBA;border-radius:6px;padding:8px 12px;
       font-size:10px;color:#856404;margin-bottom:14px;}}
.risk-b{{background:{rb};border:2px solid {rc};border-radius:10px;padding:14px 18px;
         margin-bottom:16px;display:flex;align-items:center;gap:16px;}}
.ri{{width:44px;height:44px;border-radius:50%;background:{rc};color:white;
     font-size:18px;font-weight:900;display:flex;align-items:center;justify-content:center;flex-shrink:0;}}
.rt{{font-size:17px;font-weight:800;color:{rc};}}.rp{{font-size:12px;color:#555;margin-top:2px;}}
.rec-box{{margin-left:auto;background:white;border:1px solid {rc};border-radius:6px;
          padding:8px 12px;max-width:260px;}}
.rec-t{{font-size:10px;font-weight:700;text-transform:uppercase;color:{rc};margin-bottom:3px;}}
.rec-b{{font-size:11px;color:#444;line-height:1.4;}}
.g-wrap{{margin-bottom:16px;}}
.g-lbl{{font-size:10px;color:#888;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;font-weight:600;}}
.g-track{{height:13px;border-radius:6px;
  background:linear-gradient(90deg,#27AE60 0%,#27AE60 30%,#F39C12 30%,#F39C12 70%,#E74C3C 70%,#E74C3C 100%);
  position:relative;}}
.g-needle{{position:absolute;top:-3px;width:3px;height:20px;background:#1a1a1a;
           border-radius:2px;left:{prob*100:.1f}%;transform:translateX(-50%);}}
.g-ticks{{display:flex;justify-content:space-between;font-size:9px;color:#999;margin-top:3px;}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px;}}
.card{{background:#FAFAFA;border-radius:8px;padding:12px 14px;border:1px solid #EEE;}}
.card-t{{font-size:10px;font-weight:700;color:#C44569;text-transform:uppercase;
          letter-spacing:0.5px;margin-bottom:8px;padding-bottom:5px;border-bottom:1px solid #F0E0E5;}}
table.dt{{width:100%;border-collapse:collapse;}}
table.dt td{{padding:3px 0;font-size:11px;border-bottom:1px solid #F5F5F5;}}
table.dt td:first-child{{color:#888;width:52%;}}
table.dt td:last-child{{font-weight:600;}}
.rf-list{{list-style:none;padding:0;}}
.rf-list li{{font-size:11px;padding:3px 0;border-bottom:1px solid #F5F5F5;}}
.rf-list li::before{{content:"⚠ ";color:#E74C3C;}}
.perf{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px;}}
.pc{{background:white;border-radius:6px;padding:8px;text-align:center;border:1px solid #EEE;}}
.pv{{font-size:16px;font-weight:800;color:#C44569;}}.pl{{font-size:9px;color:#888;}}
.notes-c{{background:#FFF8E7;border:1px solid #F39C12;border-radius:8px;
           padding:12px 14px;margin-bottom:14px;}}
.notes-t{{font-size:10px;font-weight:700;color:#E67E22;margin-bottom:5px;}}
.notes-b{{font-size:11px;color:#444;line-height:1.5;white-space:pre-wrap;}}
.ftr{{background:#F8F8F8;border-top:2px solid #EEE;padding:10px 28px;
      display:flex;justify-content:space-between;align-items:center;}}
.ftr-l{{font-size:10px;color:#999;}}.ftr-r{{font-size:10px;color:#C44569;font-weight:600;}}
</style></head><body>
<div class="hdr">
  <div style="font-size:30px">❤️</div>
  <div>
    <div class="hdr-t">Heart Disease Risk Assessment</div>
    <div class="hdr-s">Logistic Regression | ROC-AUC: 0.9154</div>
  </div>
  <div class="hdr-m">
    Report: HDR-{datetime.now().strftime('%Y%m%d%H%M%S')}<br>
    {datetime.now().strftime('%d %B %Y, %H:%M')}<br>
    {'Patient: '+pname if pname else ''}{' | DOB: '+pdob if pdob else ''}{' | Ref: '+pref if pref else ''}
  </div>
</div>
<div class="body">
<div class="disc">⚠️ <strong>DISCLAIMER:</strong> Educational ML tool only. Not medical advice or diagnosis. Consult a qualified healthcare professional.</div>
<div class="risk-b">
  <div class="ri">{ri}</div>
  <div>
    <div class="rt">{risk} RISK</div>
    <div class="rp">Disease probability: <strong>{prob*100:.1f}%</strong> &nbsp;|&nbsp; {'Heart Disease Detected' if pred==1 else 'No Heart Disease Detected'}</div>
  </div>
  <div class="rec-box"><div class="rec-t">📋 Recommendation</div><div class="rec-b">{rec}</div></div>
</div>
<div class="g-wrap">
  <div class="g-lbl">Risk Probability Gauge</div>
  <div class="g-track"><div class="g-needle"></div></div>
  <div class="g-ticks"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
</div>
<div class="grid2">
  <div class="card"><div class="card-t">👤 Demographics &amp; Vitals</div>
    <table class="dt">
      <tr><td>Age</td><td>{d['age']} years</td></tr>
      <tr><td>Sex</td><td>{sex_s}</td></tr>
      <tr><td>Resting BP</td><td>{d['trestbps']} mm Hg</td></tr>
      <tr><td>Cholesterol</td><td>{d['chol']} mg/dL</td></tr>
      <tr><td>Max Heart Rate</td><td>{d['thalach']} bpm</td></tr>
    </table>
  </div>
  <div class="card"><div class="card-t">🏥 Blood &amp; ECG Tests</div>
    <table class="dt">
      <tr><td>Fasting Sugar&gt;120</td><td>{fbs_s}</td></tr>
      <tr><td>Resting ECG</td><td>{ecg_s}</td></tr>
      <tr><td>ST Depression</td><td>{d['oldpeak']}</td></tr>
      <tr><td>ST Slope</td><td>{sl_s}</td></tr>
      <tr><td>Exercise Angina</td><td>{ex_s}</td></tr>
    </table>
  </div>
</div>
<div class="grid2">
  <div class="card"><div class="card-t">💊 Clinical Findings</div>
    <table class="dt">
      <tr><td>Chest Pain</td><td>{cp_s}</td></tr>
      <tr><td>Major Vessels</td><td>{d['ca']}</td></tr>
      <tr><td>Thalassemia</td><td>{th_s}</td></tr>
    </table>
  </div>
  <div class="card"><div class="card-t">⚠️ Risk Factors</div>
    <ul class="rf-list">{rf_html}</ul>
  </div>
</div>
<div class="perf">
  <div class="pc"><div class="pv">83.6%</div><div class="pl">Accuracy</div></div>
  <div class="pc"><div class="pv">84.9%</div><div class="pl">Recall</div></div>
  <div class="pc"><div class="pv">84.9%</div><div class="pl">F1-Score</div></div>
  <div class="pc"><div class="pv">0.917</div><div class="pl">ROC-AUC</div></div>
</div>
{"<div class='notes-c'><div class='notes-t'>📝 Clinician Notes</div><div class='notes-b'>"+notes+"</div></div>" if notes.strip() else ""}
<div class="card"><div class="card-t">📋 Recommendations by Risk Level</div>
  <table class="dt">
    <tr><td style="color:#27AE60;font-weight:700">LOW &lt;30%</td><td>Healthy lifestyle, annual check-ups, monitor BP &amp; cholesterol.</td></tr>
    <tr><td style="color:#F39C12;font-weight:700">MEDIUM 30–70%</td><td>GP appointment within 4 weeks. Lipid panel, ECG. Lifestyle changes.</td></tr>
    <tr><td style="color:#E74C3C;font-weight:700">HIGH &gt;70%</td><td>Urgent cardiology referral. Comprehensive cardiac evaluation required.</td></tr>
  </table>
</div>
</div>
<div class="ftr">
  <div class="ftr-l">Heart Disease Prediction System | Logistic Regression | UCI Heart Disease (n=302) | 5-Fold CV</div>
  <div class="ftr-r">⚠️ Educational Tool — Not for Clinical Diagnosis</div>
</div>
</body></html>"""

# ══════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════

# ── Compact header ────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:4px 0 2px 0;">
  <span style="font-size:1.55rem;font-weight:900;
    background:linear-gradient(135deg,#FF6B9D,#C44569);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    ❤️ Heart Disease Risk Predictor
  </span><br>
  <span style="font-size:0.72rem;color:#888;">
    Logistic Regression &nbsp;|&nbsp; ROC-AUC: 0.9154 &nbsp;|&nbsp; UCI Dataset &nbsp;|&nbsp; Educational Tool Only
  </span>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ── Patient ID row (compact, single line) ─────────────────────────────────
pi1, pi2, pi3, pi4 = st.columns([2, 1.5, 1.5, 0.5])
patient_name = pi1.text_input("Patient Name / ID", placeholder="e.g. Patient 001", label_visibility="collapsed", key="pname")
pi1.markdown("<span style='font-size:0.65rem;color:#aaa;'>👤 Patient Name / ID</span>", unsafe_allow_html=True)
patient_dob  = pi2.text_input("DOB", placeholder="DD/MM/YYYY", label_visibility="collapsed", key="pdob")
pi2.markdown("<span style='font-size:0.65rem;color:#aaa;'>📅 Date of Birth</span>", unsafe_allow_html=True)
patient_ref  = pi3.text_input("Clinician", placeholder="Dr. ...", label_visibility="collapsed", key="pref")
pi3.markdown("<span style='font-size:0.65rem;color:#aaa;'>🩺 Referring Clinician</span>", unsafe_allow_html=True)

st.markdown("<hr style='margin:2px 0 4px 0;'/>", unsafe_allow_html=True)

# ── MAIN LAYOUT: 4 input cols + 1 results col ─────────────────────────────
c1, c2, c3, c4, c5 = st.columns([1.05, 1.05, 1.05, 1.05, 1.3])

# ── COL 1: Demographics & Vitals ─────────────────────────────────────────
with c1:
    st.markdown("<div class='sec-head'>👤 Demographics & Vitals</div>", unsafe_allow_html=True)
    age      = st.number_input("Age (yrs)", 1, 120, 50, key="age")
    sex      = st.selectbox("Sex", [0,1], format_func=lambda x:"♀ Female" if x==0 else "♂ Male", key="sex")
    trestbps = st.number_input("Resting BP (mmHg)", 50, 250, 120, key="bp")

    # Compact BP badge
    bp_badge = ("bg","Normal BP") if trestbps<120 else ("by","Elevated BP") if trestbps<140 else ("br","High BP")
    st.markdown(f"<span class='badge {bp_badge[0]}'>{bp_badge[1]}</span>", unsafe_allow_html=True)

    chol     = st.number_input("Cholesterol (mg/dL)", 50, 600, 200, key="chol")
    ch_badge = ("bg","Desirable") if chol<200 else ("by","Borderline") if chol<240 else ("br","High")
    st.markdown(f"<span class='badge {ch_badge[0]}'>{ch_badge[1]}</span>", unsafe_allow_html=True)

    thalach  = st.number_input("Max HR (bpm)", 50, 250, 150, key="hr")
    hr_pct   = int(thalach / max(220-age, 1) * 100)
    hr_badge = ("bg",f"{hr_pct}% est.max") if hr_pct>=85 else ("by",f"{hr_pct}% est.max") if hr_pct>=70 else ("br",f"{hr_pct}% est.max")
    st.markdown(f"<span class='badge {hr_badge[0]}'>{hr_badge[1]}</span>", unsafe_allow_html=True)

# ── COL 2: Blood & ECG ───────────────────────────────────────────────────
with c2:
    st.markdown("<div class='sec-head'>🏥 Blood & ECG</div>", unsafe_allow_html=True)
    fbs     = st.selectbox("Fasting Sugar>120", [0,1], format_func=lambda x:"No" if x==0 else "Yes (>120)", key="fbs")
    restecg = st.selectbox("Resting ECG", [0,1,2],
                            format_func=lambda x:["Normal","ST-T Abnl.","LVH"][x], key="ecg")
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, 0.1, key="st")
    op_badge= ("bg","None") if oldpeak==0 else ("by","Mild") if oldpeak<=2 else ("br","Significant")
    st.markdown(f"<span class='badge {op_badge[0]}'>ST: {op_badge[1]}</span>", unsafe_allow_html=True)

    slope   = st.selectbox("ST Slope", [0,1,2],
                            format_func=lambda x:["Upsloping","Flat","Downsloping"][x], key="slope")
    sl_badge= ("bg","Favourable") if slope==0 else ("by","Intermediate") if slope==1 else ("br","Ischaemic")
    st.markdown(f"<span class='badge {sl_badge[0]}'>{sl_badge[1]}</span>", unsafe_allow_html=True)

    # Mini ECG score
    ecg_score = (1 if restecg>0 else 0)+(1 if oldpeak>2 else 0)+(1 if slope==2 else 0)
    ecg_col   = "#27AE60" if ecg_score==0 else "#F39C12" if ecg_score==1 else "#E74C3C"
    st.markdown(f"""
    <div style="font-size:0.65rem;color:#888;margin-top:4px;">ECG indicators: {ecg_score}/3
      <div style="height:5px;border-radius:3px;background:#eee;margin-top:2px;">
        <div style="height:5px;border-radius:3px;width:{ecg_score/3*100:.0f}%;
          background:{ecg_col};"></div>
      </div>
    </div>""", unsafe_allow_html=True)

# ── COL 3: Clinical ──────────────────────────────────────────────────────
with c3:
    st.markdown("<div class='sec-head'>💊 Clinical</div>", unsafe_allow_html=True)
    cp    = st.selectbox("Chest Pain", [0,1,2,3],
                          format_func=lambda x:["Typical Angina","Atypical","Non-anginal","Asymptomatic"][x], key="cp")
    cp_badge=("br","Classic cardiac") if cp==0 else ("by","Possible cardiac") if cp==1 else ("bg","Less likely") if cp==2 else ("by","Silent — evaluate")
    st.markdown(f"<span class='badge {cp_badge[0]}'>{cp_badge[1]}</span>", unsafe_allow_html=True)

    exang = st.selectbox("Exercise Angina", [0,1], format_func=lambda x:"No" if x==0 else "Yes", key="exang")
    ex_badge=("br","Positive") if exang==1 else ("bg","Negative")
    st.markdown(f"<span class='badge {ex_badge[0]}'>{ex_badge[1]}</span>", unsafe_allow_html=True)

    ca    = st.selectbox("Vessels (0–4)", [0,1,2,3,4], key="ca")
    ca_badge=["bg","by","br","br","br"][ca]
    ca_text=[("bg","No blockage"),("by","1 blocked"),("br","2 blocked"),("br","3 blocked"),("br","4 blocked")][ca]
    st.markdown(f"<span class='badge {ca_text[0]}'>{ca_text[1]}</span>", unsafe_allow_html=True)

    thal  = st.selectbox("Thalassemia", [0,1,2,3],
                          format_func=lambda x:["Normal","Fixed Defect","Reversible","Unknown"][x], key="thal")
    th_badge=("bg","Normal perfusion") if thal==0 else ("by","Permanent defect") if thal==1 else ("br","Ischaemic") if thal==2 else ("by","Undetermined")
    st.markdown(f"<span class='badge {th_badge[0]}'>{th_badge[1]}</span>", unsafe_allow_html=True)

    # Mini clinical score
    clin_score=(1 if ca>0 else 0)+(1 if exang==1 else 0)+(1 if cp==0 else 0)+(1 if thal==2 else 0)
    cl_col="#27AE60" if clin_score==0 else "#F39C12" if clin_score<=2 else "#E74C3C"
    st.markdown(f"""
    <div style="font-size:0.65rem;color:#888;margin-top:4px;">Clinical risk: {clin_score}/4
      <div style="height:5px;border-radius:3px;background:#eee;margin-top:2px;">
        <div style="height:5px;border-radius:3px;width:{clin_score/4*100:.0f}%;
          background:{cl_col};"></div>
      </div>
    </div>""", unsafe_allow_html=True)

# ── COL 4: Notes + Predict ───────────────────────────────────────────────
with c4:
    st.markdown("<div class='sec-head'>📝 Notes & Action</div>", unsafe_allow_html=True)
    clinical_notes = st.text_area(
        "Clinician Notes",
        placeholder="Clinical notes, medications, history...",
        height=138, key="notes", label_visibility="collapsed"
    )
    st.markdown("<span style='font-size:0.65rem;color:#aaa;'>📝 Clinician notes (included in report)</span>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

    # Disclaimer block
    st.markdown("""
    <div style="margin-top:6px;padding:5px 7px;background:#FFF3CD;border-radius:6px;
      border-left:3px solid #F39C12;font-size:0.63rem;color:#856404;line-height:1.4;">
      ⚠️ <strong>Educational tool only.</strong><br>
      Not for medical diagnosis.<br>
      Low: &lt;30% | Medium: 30–70% | High: &gt;70%
    </div>
    """, unsafe_allow_html=True)

# ── COL 5: Results ───────────────────────────────────────────────────────
with c5:
    st.markdown("<div class='sec-head'>📊 Prediction Results</div>", unsafe_allow_html=True)

    # Run prediction
    if predict_btn:
        inp = pd.DataFrame([{
            'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,'chol':chol,
            'fbs':fbs,'restecg':restecg,'thalach':thalach,'exang':exang,
            'oldpeak':oldpeak,'slope':slope,'ca':ca,'thal':thal
        }])
        if model is not None:
            try:
                pred = int(model.predict(inp)[0])
                prob = float(model.predict_proba(inp)[0,1])
            except Exception as e:
                st.error(f"Error: {e}"); pred, prob = 0, 0.5
        else:
            import random; prob = round(random.uniform(0.25,0.75),3); pred = 1 if prob>0.5 else 0
            st.caption("⚠️ Demo mode – model file not found")

        st.session_state.update({
            'pred':pred,'prob':prob,
            'inp':{'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,'chol':chol,
                   'fbs':fbs,'restecg':restecg,'thalach':thalach,'exang':exang,
                   'oldpeak':oldpeak,'slope':slope,'ca':ca,'thal':thal},
            'notes':clinical_notes,
            'pname':patient_name,'pdob':patient_dob,'pref':patient_ref
        })

    # Show results
    if 'prob' in st.session_state:
        prob  = st.session_state.prob
        pred  = st.session_state.pred
        risk  = "LOW" if prob<0.3 else "MEDIUM" if prob<0.7 else "HIGH"
        rc    = {"LOW":"r-low","MEDIUM":"r-medium","HIGH":"r-high"}[risk]
        ri    = {"LOW":"✅","MEDIUM":"⚡","HIGH":"⚠️"}[risk]
        rec   = {"LOW":"Maintain healthy lifestyle & annual check-ups.",
                 "MEDIUM":"Schedule GP appointment within 4 weeks.",
                 "HIGH":"Seek urgent cardiology referral immediately."}[risk]
        rcolor= {"LOW":"#27AE60","MEDIUM":"#F39C12","HIGH":"#E74C3C"}[risk]

        # Risk card
        st.markdown(f"""
        <div class="risk-card {rc}">
          <p class="rt">{ri} {risk} RISK</p>
          <p class="rp">Probability: <strong>{prob*100:.1f}%</strong></p>
          <p class="rr">{'❤️ Heart Disease Detected' if pred==1 else '💚 No Disease Detected'}</p>
        </div>""", unsafe_allow_html=True)

        # Metrics row
        m1, m2 = st.columns(2)
        m1.metric("Confidence", f"{max(prob,1-prob)*100:.0f}%")
        m2.metric("Result", "Disease" if pred==1 else "Healthy")

        # Compact gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            number={'font':{'size':16,'color':rcolor},'suffix':'%'},
            title={'text':"Risk %",'font':{'size':9,'color':'#C44569'}},
            gauge={
                'axis':{'range':[0,100],'tickfont':{'size':7}},
                'bar':{'color':rcolor,'thickness':0.65},
                'steps':[
                    {'range':[0,30],'color':'rgba(39,174,96,0.12)'},
                    {'range':[30,70],'color':'rgba(243,156,18,0.12)'},
                    {'range':[70,100],'color':'rgba(231,76,60,0.12)'}
                ],
                'threshold':{'line':{'color':'#C44569','width':2},'value':50}
            }
        ))
        fig.update_layout(height=130, margin=dict(l=4,r=4,t=22,b=4),
                          paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        # Recommendation
        st.markdown(f"<p style='font-size:0.72rem;color:{rcolor};font-weight:700;margin:0;'>💡 {rec}</p>",
                    unsafe_allow_html=True)

        # Risk factors (compact)
        d = st.session_state.inp
        rfs = []
        if d['age']>55:       rfs.append(f"Age {d['age']} — elevated risk")
        if d['chol']>240:     rfs.append(f"Cholesterol {d['chol']} mg/dL — high")
        if d['trestbps']>140: rfs.append(f"BP {d['trestbps']} mmHg — hypertensive")
        if d['fbs']==1:       rfs.append("Fasting sugar >120 mg/dL")
        if d['exang']==1:     rfs.append("Exercise-induced angina")
        if d['oldpeak']>2:    rfs.append(f"ST depression {d['oldpeak']} — significant")
        if d['ca']>0:         rfs.append(f"{d['ca']} vessel(s) blocked")
        if d['thal']==2:      rfs.append("Reversible thal. defect")
        if d['cp']==0:        rfs.append("Typical angina presentation")

        if rfs:
            with st.expander(f"⚠️ {len(rfs)} risk factor(s)", expanded=False):
                for r in rfs:
                    st.markdown(f"<span style='font-size:0.72rem;'>🔸 {r}</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge bg' style='font-size:0.68rem;'>✅ No major risk factors</span>",
                        unsafe_allow_html=True)

        # PDF download
        st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
        html_r = make_report(
            d=d, pred=pred, prob=prob, risk=risk, rec=rec, rfs=rfs,
            notes=st.session_state.get('notes',''),
            pname=st.session_state.get('pname',''),
            pdob=st.session_state.get('pdob',''),
            pref=st.session_state.get('pref','')
        )
        b64 = base64.b64encode(html_r.encode()).decode()
        fname = f"HeartRisk_{(st.session_state.get('pname','Patient')).replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        st.markdown(f"""
        <a href="data:text/html;base64,{b64}" download="{fname}" style="text-decoration:none;display:block;margin-top:2px;">
          <button style="width:100%;background:linear-gradient(135deg,#11998e,#38ef7d);
            color:white;font-size:0.8rem;font-weight:700;padding:5px;border-radius:8px;
            border:none;cursor:pointer;box-shadow:0 3px 12px rgba(17,153,142,0.4);">
            📄 Download PDF Report
          </button>
        </a>""", unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("""
        <div style="text-align:center;padding:30px 10px;color:#ccc;">
          <div style="font-size:2.5rem;">❤️</div>
          <div style="font-size:0.8rem;margin-top:8px;color:#bbb;">
            Fill in patient data and click<br><strong>Predict Risk</strong>
          </div>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disc">
  ❤️ Heart Disease Risk Predictor &nbsp;|&nbsp; Logistic Regression &nbsp;|&nbsp; ROC-AUC: 0.9154 &nbsp;|&nbsp;
  UCI Dataset (n=302) &nbsp;|&nbsp; 5-Fold CV &nbsp;|&nbsp;
  <span style="color:#E74C3C;font-weight:600;">⚠️ Educational Tool — Not for Clinical Diagnosis</span>
</div>
""", unsafe_allow_html=True)
