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
/* ── Animated gradient background ── */
.stApp {
    background: linear-gradient(-45deg,#FF6B9D,#C44569,#FFA502,#FF6348,#786FA6,#F8B500);
    background-size:400% 400%;
    animation:gradientShift 14s ease infinite;
}
@keyframes gradientShift {
    0%{background-position:0% 50%}25%{background-position:50% 100%}
    50%{background-position:100% 50%}75%{background-position:50% 0%}100%{background-position:0% 50%}
}

/* ── White card ── */
.main .block-container {
    background:rgba(255,255,255,0.95);
    border-radius:16px;
    padding:0.4rem 0.9rem 0.3rem 0.9rem !important;
    max-width:100% !important;
    box-shadow:0 8px 32px rgba(0,0,0,0.18);
}

/* ── Hide chrome & sidebar ── */
#MainMenu,footer,[data-testid="stToolbar"],.stDeployButton,
[data-testid="collapsedControl"],
section[data-testid="stSidebar"]{display:none!important;}

/* ── Global font ── */
*{font-family:'Segoe UI',Arial,sans-serif!important;}

/* ── Input widget sizes ── */
.stNumberInput input{
    font-size:0.76rem!important;padding:1px 5px!important;
    height:26px!important;border-radius:5px!important;
}
.stNumberInput [data-testid="stNumberInputStepDown"],
.stNumberInput [data-testid="stNumberInputStepUp"]{
    width:20px!important;height:26px!important;
}
.stSelectbox>div>div{
    font-size:0.76rem!important;min-height:26px!important;
    padding:1px 6px!important;border-radius:5px!important;
}
.stTextArea textarea{
    font-size:0.73rem!important;padding:3px 5px!important;
    border-radius:5px!important;resize:none!important;line-height:1.3!important;
}
.stTextInput input{
    font-size:0.73rem!important;padding:1px 5px!important;
    height:24px!important;border-radius:5px!important;
}

/* ── Labels ── */
.stNumberInput label,.stSelectbox label,
.stTextArea label,.stTextInput label{
    font-size:0.69rem!important;margin-bottom:0!important;
    padding:0!important;color:#555!important;font-weight:600!important;
    line-height:1.2!important;
}

/* ── Kill gaps ── */
[data-testid="stVerticalBlock"]>div{gap:0!important;}
.element-container{margin:0!important;padding:0!important;}
div[data-testid="column"]{padding:0 2px!important;}

/* ── Section pill ── */
.spill{
    font-size:0.67rem;font-weight:800;color:white;
    background:linear-gradient(135deg,#C44569,#FF6B9D);
    border-radius:5px;padding:1px 7px;display:inline-block;margin-bottom:2px;
}

/* ── Tiny badges ── */
.badge{
    display:inline-block;font-size:0.6rem;font-weight:700;
    padding:0 5px;border-radius:8px;margin:1px 0 2px 0;line-height:1.45;
}
.bg{background:#d4edda;color:#155724;}
.by{background:#fff3cd;color:#856404;}
.br{background:#f8d7da;color:#721c24;}

/* ── Risk card ── */
.rcard{
    border-radius:9px;padding:7px 9px;margin:3px 0;
    border-left:4px solid;animation:popIn 0.3s ease;
}
@keyframes popIn{0%{opacity:0;transform:scale(0.9)}100%{opacity:1;transform:scale(1)}}
.r-low   {background:linear-gradient(135deg,#43C6AC,#191654);border-color:#43C6AC;}
.r-med   {background:linear-gradient(135deg,#F2994A,#F2C94C);border-color:#F2994A;}
.r-high  {background:linear-gradient(135deg,#EB5757,#FF6B9D);border-color:#EB5757;}
.rcard p{color:white!important;margin:0!important;}
.rcard .rt{font-size:0.92rem;font-weight:900;}
.rcard .rp{font-size:0.76rem;margin-top:1px!important;}
.rcard .rr{font-size:0.67rem;opacity:0.9;margin-top:1px!important;}

/* ── Predict button ── */
.stButton>button{
    background:linear-gradient(135deg,#FF6B9D,#C44569)!important;
    color:white!important;font-weight:800!important;font-size:0.8rem!important;
    padding:3px 0!important;height:30px!important;border-radius:7px!important;
    border:none!important;width:100%!important;
    box-shadow:0 3px 10px rgba(255,107,157,0.4)!important;transition:all 0.25s!important;
}
.stButton>button:hover{
    transform:translateY(-1px)!important;
    box-shadow:0 5px 16px rgba(255,107,157,0.55)!important;
}

/* ── Metric boxes ── */
[data-testid="metric-container"]{
    background:linear-gradient(135deg,#fafafa,#f0f0f0)!important;
    border-radius:6px!important;padding:2px 6px!important;
    border-left:3px solid #FF6B9D!important;
    box-shadow:0 1px 4px rgba(0,0,0,0.06)!important;
}
[data-testid="metric-container"] label{font-size:0.58rem!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-size:0.82rem!important;}

/* ── Disclaimer ── */
.disc{
    font-size:0.59rem;color:#999;text-align:center;
    border-top:1px solid rgba(196,69,105,0.12);padding-top:2px;margin-top:3px;
}
hr{margin:2px 0!important;border-color:rgba(196,69,105,0.15)!important;}
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

# ── PDF generator — clean medical report with notes ───────────────────────
def make_pdf(d, pred, prob, risk, rec, rfs, notes, pname, pdob, pref):
    rc  = {"LOW":"#27AE60","MEDIUM":"#F39C12","HIGH":"#E74C3C"}[risk]
    rb  = {"LOW":"#EAFAF1","MEDIUM":"#FEF9E7","HIGH":"#FDEDEC"}[risk]
    ri  = {"LOW":"✓","MEDIUM":"!","HIGH":"⚠"}[risk]
    sex_s = "Male" if d["sex"]==1 else "Female"
    cp_s  = ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"][d["cp"]]
    ecg_s = ["Normal","ST-T Abnormality","LV Hypertrophy"][d["restecg"]]
    sl_s  = ["Upsloping","Flat","Downsloping"][d["slope"]]
    th_s  = ["Normal","Fixed Defect","Reversible Defect","Unknown"][d["thal"]]
    fbs_s = "Yes" if d["fbs"]==1 else "No"
    ex_s  = "Yes" if d["exang"]==1 else "No"
    rf_li = "".join(f"<li>{x}</li>" for x in rfs) if rfs else "<li>No major risk factors identified</li>"

    pt_row = ""
    parts = []
    if pname: parts.append(f"<strong>Patient:</strong> {pname}")
    if pdob:  parts.append(f"<strong>DOB:</strong> {pdob}")
    if pref:  parts.append(f"<strong>Clinician:</strong> {pref}")
    if parts:
        pt_row = "<div style='background:#F0F4FF;border-radius:5px;padding:5px 10px;margin-bottom:10px;font-size:10px;'>" + " &nbsp;|&nbsp; ".join(parts) + "</div>"

    notes_block = ""
    if notes.strip():
        notes_block = f"""
        <div style="background:#FFF8E7;border:1px solid #F39C12;border-radius:7px;
             padding:10px 12px;margin-bottom:10px;">
          <div style="font-size:9px;font-weight:700;color:#E67E22;text-transform:uppercase;
               margin-bottom:4px;">📝 Clinician Notes</div>
          <div style="font-size:10px;color:#444;line-height:1.55;white-space:pre-wrap;">{notes}</div>
        </div>"""

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;
   font-family:'Segoe UI',Arial,sans-serif;font-size:11px;color:#222;}}
body{{background:white;}}
@media print{{
  @page{{margin:12mm 15mm;size:A4;}}
  .no-print{{display:none!important;}}
  body{{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}
}}
.hdr{{background:linear-gradient(135deg,#C44569,#FF6B9D);color:white;
      padding:13px 20px;display:flex;align-items:center;gap:12px;}}
.hdr-t{{font-size:16px;font-weight:900;}}
.hdr-s{{font-size:9px;opacity:0.85;margin-top:2px;}}
.hdr-m{{margin-left:auto;text-align:right;font-size:9px;opacity:0.85;line-height:1.5;}}
.body{{padding:12px 20px;}}
.disc{{background:#FFF3CD;border:1px solid #FFEEBA;border-radius:5px;
       padding:5px 9px;font-size:9px;color:#856404;margin-bottom:10px;}}
.risk-b{{background:{rb};border:1.5px solid {rc};border-radius:8px;
         padding:10px 14px;margin-bottom:10px;display:flex;align-items:center;gap:12px;}}
.ri{{width:36px;height:36px;border-radius:50%;background:{rc};color:white;
     font-size:15px;font-weight:900;display:flex;align-items:center;
     justify-content:center;flex-shrink:0;}}
.rt{{font-size:14px;font-weight:800;color:{rc};}}
.rp{{font-size:10px;color:#555;margin-top:2px;}}
.rec-box{{margin-left:auto;background:white;border:1px solid {rc};
          border-radius:5px;padding:6px 9px;max-width:240px;}}
.rec-t{{font-size:9px;font-weight:700;text-transform:uppercase;
        color:{rc};margin-bottom:2px;}}
.rec-b{{font-size:10px;color:#444;line-height:1.4;}}
.g-track{{height:10px;border-radius:5px;margin:3px 0 2px 0;
  background:linear-gradient(90deg,#27AE60 0%,#27AE60 30%,
  #F39C12 30%,#F39C12 70%,#E74C3C 70%,#E74C3C 100%);position:relative;}}
.g-needle{{position:absolute;top:-3px;width:3px;height:16px;background:#1a1a1a;
  border-radius:2px;left:{prob*100:.1f}%;transform:translateX(-50%);}}
.g-ticks{{display:flex;justify-content:space-between;
  font-size:8px;color:#999;margin-bottom:8px;}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-bottom:9px;}}
.card{{background:#FAFAFA;border-radius:6px;padding:9px 11px;border:1px solid #EEE;}}
.ct{{font-size:9px;font-weight:700;color:#C44569;text-transform:uppercase;
     letter-spacing:0.4px;margin-bottom:5px;padding-bottom:3px;
     border-bottom:1px solid #F0E0E5;}}
table.dt{{width:100%;border-collapse:collapse;}}
table.dt td{{padding:2px 0;font-size:10px;border-bottom:1px solid #F5F5F5;}}
table.dt td:first-child{{color:#888;width:46%;}}
table.dt td:last-child{{font-weight:600;}}
.rf-list{{list-style:none;padding:0;}}
.rf-list li{{font-size:10px;padding:2px 0;border-bottom:1px solid #F5F5F5;}}
.rf-list li::before{{content:"⚠ ";color:#E74C3C;}}
.ftr{{background:#F8F8F8;border-top:1px solid #EEE;padding:7px 20px;
      display:flex;justify-content:space-between;align-items:center;}}
.ftr-l{{font-size:9px;color:#999;}}
.ftr-r{{font-size:9px;color:#C44569;font-weight:700;}}
</style></head><body>

<div class="hdr">
  <div style="font-size:26px;">❤️</div>
  <div>
    <div class="hdr-t">Heart Disease Risk Assessment</div>
    <div class="hdr-s">Logistic Regression | ROC-AUC: 0.9154 | UCI Heart Disease Dataset</div>
  </div>
  <div class="hdr-m">
    Report: HDR-{datetime.now().strftime('%Y%m%d%H%M%S')}<br>
    {datetime.now().strftime('%d %B %Y, %H:%M')}<br>
    EDUCATIONAL USE ONLY
  </div>
</div>

<div class="body">

<div class="disc">⚠️ <strong>DISCLAIMER:</strong> This is an educational ML tool only.
It does not constitute medical advice, diagnosis, or treatment.
Always consult a qualified healthcare professional.</div>

{pt_row}

<div class="risk-b">
  <div class="ri">{ri}</div>
  <div>
    <div class="rt">{risk} RISK</div>
    <div class="rp">
      Disease probability: <strong>{prob*100:.1f}%</strong>
      &nbsp;|&nbsp; {'Heart Disease Detected' if pred==1 else 'No Heart Disease Detected'}
    </div>
  </div>
  <div class="rec-box">
    <div class="rec-t">📋 Recommendation</div>
    <div class="rec-b">{rec}</div>
  </div>
</div>

<div style="font-size:9px;color:#888;font-weight:600;text-transform:uppercase;
     letter-spacing:0.4px;margin-bottom:2px;">Risk Probability Gauge</div>
<div class="g-track"><div class="g-needle"></div></div>
<div class="g-ticks">
  <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
</div>

<div class="grid2">
  <div class="card">
    <div class="ct">👤 Demographics & Vitals</div>
    <table class="dt">
      <tr><td>Age</td><td>{d['age']} years</td></tr>
      <tr><td>Sex</td><td>{sex_s}</td></tr>
      <tr><td>Resting BP</td><td>{d['trestbps']} mm Hg</td></tr>
      <tr><td>Cholesterol</td><td>{d['chol']} mg/dL</td></tr>
      <tr><td>Max Heart Rate</td><td>{d['thalach']} bpm</td></tr>
    </table>
  </div>
  <div class="card">
    <div class="ct">🏥 Blood & ECG</div>
    <table class="dt">
      <tr><td>Fasting Sugar &gt;120</td><td>{fbs_s}</td></tr>
      <tr><td>Resting ECG</td><td>{ecg_s}</td></tr>
      <tr><td>ST Depression</td><td>{d['oldpeak']}</td></tr>
      <tr><td>ST Slope</td><td>{sl_s}</td></tr>
    </table>
  </div>
</div>

<div class="grid2">
  <div class="card">
    <div class="ct">💊 Clinical Findings</div>
    <table class="dt">
      <tr><td>Chest Pain</td><td>{cp_s}</td></tr>
      <tr><td>Exercise Angina</td><td>{ex_s}</td></tr>
      <tr><td>Major Vessels</td><td>{d['ca']}</td></tr>
      <tr><td>Thalassemia</td><td>{th_s}</td></tr>
    </table>
  </div>
  <div class="card">
    <div class="ct">⚠️ Risk Factors</div>
    <ul class="rf-list">{rf_li}</ul>
  </div>
</div>

{notes_block}

</div>

<div class="ftr">
  <div class="ftr-l">
    Heart Disease Prediction System | Logistic Regression | UCI (n=302) |
    5-Fold CV | Accuracy: 83.6% | Recall: 84.9%
  </div>
  <div class="ftr-r">⚠️ Educational Tool — Not for Clinical Diagnosis</div>
</div>

<div class="no-print" style="text-align:center;padding:10px;background:#f5f5f5;">
  <button onclick="window.print()"
    style="background:linear-gradient(135deg,#C44569,#FF6B9D);color:white;
    border:none;padding:7px 22px;border-radius:6px;font-size:12px;
    font-weight:700;cursor:pointer;margin-right:8px;">
    🖨️ Print / Save as PDF
  </button>
  <button onclick="window.close()"
    style="background:#888;color:white;border:none;padding:7px 16px;
    border-radius:6px;font-size:12px;cursor:pointer;">
    Close
  </button>
</div>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════════════════════════════════════

# ── Header (1 line) ───────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;padding:1px 2px 0 2px;">
  <span style="font-size:1.28rem;font-weight:900;
    background:linear-gradient(135deg,#FF6B9D,#C44569);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    ❤️ Heart Disease Risk Predictor
  </span>
  <span style="font-size:0.62rem;color:#aaa;">
    Logistic Regression &nbsp;·&nbsp; ROC-AUC 0.9154 &nbsp;·&nbsp; UCI Dataset &nbsp;·&nbsp;
    <span style="color:#E74C3C;font-weight:700;">Educational Only</span>
  </span>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# ── Patient ID row ────────────────────────────────────────────────────────
r0c1, r0c2, r0c3 = st.columns([2, 1.5, 1.5])
# Use unique keys that do NOT clash with any clinical widget keys
pname = r0c1.text_input("Patient Name / ID", placeholder="e.g. Patient 001", key="w_pname")
pdob  = r0c2.text_input("Date of Birth", placeholder="DD/MM/YYYY", key="w_pdob")
pref  = r0c3.text_input("Referring Clinician", placeholder="Dr. ...", key="w_pref")

st.markdown("<hr style='margin:2px 0 3px 0;'/>", unsafe_allow_html=True)

# ── MAIN 5-COLUMN LAYOUT ──────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.15])

# ── COL 1 · Demographics & Vitals ────────────────────────────────────────
with c1:
    st.markdown("<div class='spill'>👤 Demographics & Vitals</div>", unsafe_allow_html=True)

    age      = st.number_input("Age (yrs)", 1, 120, 50, key="w_age")
    ab       = ("bg","Low risk") if age<45 else ("by","Moderate") if age<60 else ("br","Elevated")
    st.markdown(f"<span class='badge {ab[0]}'>{ab[1]}</span>", unsafe_allow_html=True)

    sex      = st.selectbox("Sex", [0,1],
                             format_func=lambda x:"♀ Female" if x==0 else "♂ Male", key="w_sex")

    trestbps = st.number_input("Resting BP (mmHg)", 50, 250, 120, key="w_bp")
    bb       = ("bg","Normal") if trestbps<120 else ("by","Elevated") if trestbps<140 else ("br","High BP")
    st.markdown(f"<span class='badge {bb[0]}'>{bb[1]}</span>", unsafe_allow_html=True)

    chol     = st.number_input("Cholesterol (mg/dL)", 50, 600, 200, key="w_chol")
    cb       = ("bg","Desirable") if chol<200 else ("by","Borderline") if chol<240 else ("br","High")
    st.markdown(f"<span class='badge {cb[0]}'>{cb[1]}</span>", unsafe_allow_html=True)

    thalach  = st.number_input("Max HR (bpm)", 50, 250, 150, key="w_hr")
    pct      = int(thalach / max(220 - age, 1) * 100)
    hb       = ("bg",f"{pct}% est.max") if pct>=85 else ("by",f"{pct}% est.max") if pct>=70 else ("br",f"{pct}% est.max")
    st.markdown(f"<span class='badge {hb[0]}'>{hb[1]}</span>", unsafe_allow_html=True)

# ── COL 2 · Blood & ECG ──────────────────────────────────────────────────
with c2:
    st.markdown("<div class='spill'>🏥 Blood & ECG</div>", unsafe_allow_html=True)

    fbs     = st.selectbox("Fasting Sugar >120", [0,1],
                            format_func=lambda x:"No" if x==0 else "Yes (>120)", key="w_fbs")
    if fbs==1:
        st.markdown("<span class='badge br'>Elevated glucose</span>", unsafe_allow_html=True)

    restecg = st.selectbox("Resting ECG", [0,1,2],
                            format_func=lambda x:["Normal","ST-T Abnormal","LVH"][x], key="w_ecg")
    if restecg>0:
        st.markdown("<span class='badge by'>ECG abnormality</span>", unsafe_allow_html=True)

    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, 0.1, key="w_st")
    ob      = ("bg","None") if oldpeak==0 else ("by","Mild") if oldpeak<=2 else ("br","Significant")
    st.markdown(f"<span class='badge {ob[0]}'>ST: {ob[1]}</span>", unsafe_allow_html=True)

    slope   = st.selectbox("ST Slope", [0,1,2],
                            format_func=lambda x:["Upsloping","Flat","Downsloping"][x], key="w_slope")
    slb     = ("bg","Favourable") if slope==0 else ("by","Intermediate") if slope==1 else ("br","Ischaemic")
    st.markdown(f"<span class='badge {slb[0]}'>{slb[1]}</span>", unsafe_allow_html=True)

# ── COL 3 · Clinical ─────────────────────────────────────────────────────
with c3:
    st.markdown("<div class='spill'>💊 Clinical Findings</div>", unsafe_allow_html=True)

    cp    = st.selectbox("Chest Pain", [0,1,2,3],
                          format_func=lambda x:["Typical Angina","Atypical","Non-anginal","Asymptomatic"][x],
                          key="w_cp")
    cpb   = ("br","Classic cardiac") if cp==0 else ("by","Possible") if cp==1 else ("bg","Less likely") if cp==2 else ("by","Evaluate")
    st.markdown(f"<span class='badge {cpb[0]}'>{cpb[1]}</span>", unsafe_allow_html=True)

    exang = st.selectbox("Exercise Angina", [0,1],
                          format_func=lambda x:"No" if x==0 else "Yes", key="w_exang")
    st.markdown(f"<span class='badge {'br' if exang==1 else 'bg'}'>{'Positive' if exang==1 else 'Negative'}</span>",
                unsafe_allow_html=True)

    ca    = st.selectbox("Major Vessels (0–4)", [0,1,2,3,4], key="w_ca")
    cac   = ["bg","by","br","br","br"][ca]
    cat   = ["No blockage","1 blocked","2 blocked","3 blocked","4 blocked"][ca]
    st.markdown(f"<span class='badge {cac}'>{cat}</span>", unsafe_allow_html=True)

    thal  = st.selectbox("Thalassemia", [0,1,2,3],
                          format_func=lambda x:["Normal","Fixed Defect","Reversible","Unknown"][x],
                          key="w_thal")
    thb   = ("bg","Normal") if thal==0 else ("by","Permanent defect") if thal==1 else ("br","Ischaemic") if thal==2 else ("by","Unknown")
    st.markdown(f"<span class='badge {thb[0]}'>{thb[1]}</span>", unsafe_allow_html=True)

# ── COL 4 · Notes + Predict ──────────────────────────────────────────────
with c4:
    st.markdown("<div class='spill'>📝 Notes & Predict</div>", unsafe_allow_html=True)

    notes = st.text_area("Clinician Notes (included in PDF)",
                          placeholder="Clinical notes, medications, history, observations...",
                          height=145, key="w_notes", label_visibility="visible")

    st.markdown("<div style='height:3px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

    st.markdown("""
    <div style="margin-top:5px;padding:4px 6px;background:#FFF3CD;border-radius:5px;
      border-left:3px solid #F39C12;font-size:0.6rem;color:#856404;line-height:1.4;">
      ⚠️ <strong>Educational tool only.</strong>
      Not for medical diagnosis.<br>
      LOW &lt;30% · MEDIUM 30–70% · HIGH &gt;70%
    </div>
    """, unsafe_allow_html=True)

# ── COL 5 · Results + Download ───────────────────────────────────────────
with c5:
    st.markdown("<div class='spill'>📊 Prediction Results</div>", unsafe_allow_html=True)

    # ── Run prediction when button clicked ────────────────────────────────
    if predict_btn:
        inp = pd.DataFrame([{
            'age': age, 'sex': sex, 'cp': cp,
            'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
            'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }])

        if model is not None:
            try:
                pred_val = int(model.predict(inp)[0])
                prob_val = float(model.predict_proba(inp)[0, 1])
            except Exception as e:
                st.error(f"Prediction error: {e}")
                pred_val, prob_val = 0, 0.5
        else:
            # Demo mode — no model file found
            import random
            random.seed(age + sex + cp + ca)
            prob_val = round(random.uniform(0.2, 0.8), 3)
            pred_val = 1 if prob_val > 0.5 else 0
            st.caption("⚠️ Demo mode – model file not found")

        # ── Store results in session state with RESULT_ prefix ─────────────
        # IMPORTANT: keys must NOT match any widget key (w_*).
        # We use "res_*" prefix to avoid all conflicts.
        st.session_state["res_pred"]  = pred_val
        st.session_state["res_prob"]  = prob_val
        st.session_state["res_age"]   = age
        st.session_state["res_sex"]   = sex
        st.session_state["res_cp"]    = cp
        st.session_state["res_bp"]    = trestbps
        st.session_state["res_chol"]  = chol
        st.session_state["res_fbs"]   = fbs
        st.session_state["res_ecg"]   = restecg
        st.session_state["res_hr"]    = thalach
        st.session_state["res_exang"] = exang
        st.session_state["res_st"]    = oldpeak
        st.session_state["res_slope"] = slope
        st.session_state["res_ca"]    = ca
        st.session_state["res_thal"]  = thal
        st.session_state["res_notes"] = notes
        st.session_state["res_pname"] = pname
        st.session_state["res_pdob"]  = pdob
        st.session_state["res_pref"]  = pref

    # ── Display results if available ──────────────────────────────────────
    if "res_prob" in st.session_state:
        prob = st.session_state["res_prob"]
        pred = st.session_state["res_pred"]

        risk  = "LOW" if prob < 0.3 else "MEDIUM" if prob < 0.7 else "HIGH"
        rcls  = {"LOW":"r-low","MEDIUM":"r-med","HIGH":"r-high"}[risk]
        ricon = {"LOW":"✅","MEDIUM":"⚡","HIGH":"⚠️"}[risk]
        rclr  = {"LOW":"#27AE60","MEDIUM":"#F39C12","HIGH":"#E74C3C"}[risk]
        rec   = {
            "LOW":    "Maintain healthy lifestyle & annual check-ups.",
            "MEDIUM": "Schedule GP appointment within 4 weeks.",
            "HIGH":   "Seek urgent cardiology referral immediately."
        }[risk]

        # Risk card
        st.markdown(f"""
        <div class="rcard {rcls}">
          <p class="rt">{ricon} {risk} RISK</p>
          <p class="rp">Probability: <strong>{prob*100:.1f}%</strong></p>
          <p class="rr">{'❤️ Heart Disease Detected' if pred==1 else '💚 No Disease Detected'}</p>
        </div>""", unsafe_allow_html=True)

        # Metrics
        m1, m2 = st.columns(2)
        m1.metric("Confidence", f"{max(prob, 1-prob)*100:.0f}%")
        m2.metric("Result", "Disease" if pred==1 else "Healthy")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'font': {'size': 15, 'color': rclr}, 'suffix': '%'},
            title={'text': "Risk %", 'font': {'size': 9, 'color': '#C44569'}},
            gauge={
                'axis': {'range': [0, 100], 'tickfont': {'size': 7}},
                'bar': {'color': rclr, 'thickness': 0.65},
                'steps': [
                    {'range': [0, 30],   'color': 'rgba(39,174,96,0.12)'},
                    {'range': [30, 70],  'color': 'rgba(243,156,18,0.12)'},
                    {'range': [70, 100], 'color': 'rgba(231,76,60,0.12)'}
                ],
                'threshold': {'line': {'color': '#C44569', 'width': 2}, 'value': 50}
            }
        ))
        fig.update_layout(height=125, margin=dict(l=4,r=4,t=20,b=2),
                          paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        # Recommendation
        st.markdown(
            f"<p style='font-size:0.7rem;color:{rclr};font-weight:700;margin:0;'>💡 {rec}</p>",
            unsafe_allow_html=True
        )

        # Risk factors
        d = {
            'age': st.session_state["res_age"],   'sex': st.session_state["res_sex"],
            'cp':  st.session_state["res_cp"],    'trestbps': st.session_state["res_bp"],
            'chol':st.session_state["res_chol"],  'fbs': st.session_state["res_fbs"],
            'restecg':st.session_state["res_ecg"],'thalach': st.session_state["res_hr"],
            'exang':st.session_state["res_exang"],'oldpeak': st.session_state["res_st"],
            'slope':st.session_state["res_slope"],'ca': st.session_state["res_ca"],
            'thal': st.session_state["res_thal"]
        }
        rfs = []
        if d['age']>55:        rfs.append(f"Age {d['age']} yrs — elevated risk above 55")
        if d['chol']>240:      rfs.append(f"Cholesterol {d['chol']} mg/dL — high (>240)")
        if d['trestbps']>140:  rfs.append(f"BP {d['trestbps']} mmHg — hypertensive")
        if d['fbs']==1:        rfs.append("Fasting blood sugar >120 mg/dL")
        if d['exang']==1:      rfs.append("Exercise-induced angina present")
        if d['oldpeak']>2:     rfs.append(f"ST depression {d['oldpeak']} — significant")
        if d['ca']>0:          rfs.append(f"{d['ca']} major vessel(s) blocked")
        if d['thal']==2:       rfs.append("Reversible thalassemia defect")
        if d['cp']==0:         rfs.append("Typical angina — classic presentation")

        if rfs:
            with st.expander(f"⚠️ {len(rfs)} risk factor(s)", expanded=False):
                for rf in rfs:
                    st.markdown(
                        f"<span style='font-size:0.68rem;'>🔸 {rf}</span>",
                        unsafe_allow_html=True
                    )
        else:
            st.markdown(
                "<span class='badge bg' style='font-size:0.65rem;'>✅ No major risk factors</span>",
                unsafe_allow_html=True
            )

        # ── PDF download button ───────────────────────────────────────────
        html_report = make_pdf(
            d=d,
            pred=pred, prob=prob,
            risk=risk, rec=rec, rfs=rfs,
            notes=st.session_state.get("res_notes", ""),
            pname=st.session_state.get("res_pname", ""),
            pdob=st.session_state.get("res_pdob", ""),
            pref=st.session_state.get("res_pref", "")
        )
        b64 = base64.b64encode(html_report.encode()).decode()
        safe_name = (st.session_state.get("res_pname","Patient") or "Patient").replace(" ", "_")
        fname = f"HeartRisk_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"

        st.markdown(f"""
        <a href="data:text/html;base64,{b64}" download="{fname}"
           style="text-decoration:none;display:block;margin-top:4px;">
          <button style="width:100%;
            background:linear-gradient(135deg,#11998e,#38ef7d);
            color:white;font-size:0.76rem;font-weight:800;
            padding:4px;height:28px;border-radius:7px;border:none;
            cursor:pointer;box-shadow:0 3px 10px rgba(17,153,142,0.38);
            transition:all 0.25s;">
            📄 Download PDF Report
          </button>
        </a>""", unsafe_allow_html=True)

    else:
        # Empty state placeholder
        st.markdown("""
        <div style="text-align:center;padding:28px 8px;color:#ddd;">
          <div style="font-size:2.2rem;">❤️</div>
          <div style="font-size:0.74rem;color:#bbb;margin-top:6px;">
            Fill in patient data<br>and click <strong>Predict Risk</strong>
          </div>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disc">
  ❤️ Heart Disease Risk Predictor &nbsp;·&nbsp;
  Logistic Regression &nbsp;·&nbsp; ROC-AUC 0.9154 &nbsp;·&nbsp;
  UCI Dataset (n=302) &nbsp;·&nbsp; 5-Fold CV &nbsp;·&nbsp;
  <span style="color:#E74C3C;font-weight:700;">
    ⚠️ Educational Tool — Not for Clinical Diagnosis
  </span>
</div>
""", unsafe_allow_html=True)
