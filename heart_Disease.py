import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, date
import io

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg,#FF6B9D,#C44569,#FFA502,#FF6348,#786FA6,#F8B500);
    background-size:400% 400%;
    animation:gradientShift 14s ease infinite;
}
@keyframes gradientShift {
    0%  {background-position:0% 50%}  25%{background-position:50% 100%}
    50% {background-position:100% 50%} 75%{background-position:50% 0%}
    100%{background-position:0% 50%}
}
.main .block-container {
    background:rgba(255,255,255,0.96);
    border-radius:16px;
    padding:0.7rem 1.1rem 0.8rem 1.1rem !important;
    max-width:100% !important;
    box-shadow:0 8px 32px rgba(0,0,0,0.18);
}
#MainMenu,footer,[data-testid="stToolbar"],.stDeployButton,
[data-testid="collapsedControl"],
section[data-testid="stSidebar"]{display:none!important;}

*{font-family:'Segoe UI',Arial,sans-serif!important;}

/* ── Bigger input fonts ── */
.stNumberInput input {
    font-size:.9rem!important; padding:2px 7px!important;
    height:32px!important; border-radius:6px!important;
}
.stNumberInput [data-testid="stNumberInputStepDown"],
.stNumberInput [data-testid="stNumberInputStepUp"] {
    width:24px!important; height:32px!important;
}
.stSelectbox>div>div {
    font-size:.9rem!important; min-height:32px!important;
    padding:2px 8px!important; border-radius:6px!important;
}
.stTextArea textarea {
    font-size:.86rem!important; padding:4px 7px!important;
    border-radius:6px!important; resize:none!important; line-height:1.4!important;
}
.stTextInput input {
    font-size:.86rem!important; padding:2px 7px!important;
    height:30px!important; border-radius:6px!important;
}
.stDateInput input {
    font-size:.86rem!important; padding:2px 7px!important;
    height:30px!important; border-radius:6px!important;
}
/* Labels bigger */
.stNumberInput label,.stSelectbox label,
.stTextArea label,.stTextInput label,.stDateInput label {
    font-size:.8rem!important; margin-bottom:0!important; padding:0!important;
    color:#555!important; font-weight:600!important; line-height:1.3!important;
}
[data-testid="stVerticalBlock"]>div{gap:0!important;}
.element-container{margin:0!important;padding:0!important;}
div[data-testid="column"]{padding:0 3px!important;}

.spill {
    font-size:.76rem; font-weight:800; color:white;
    background:linear-gradient(135deg,#C44569,#FF6B9D);
    border-radius:6px; padding:2px 9px; display:inline-block; margin-bottom:3px;
}
.badge {
    display:inline-block; font-size:.68rem; font-weight:700;
    padding:1px 6px; border-radius:8px; margin:1px 0 3px 0; line-height:1.5;
}
.bg{background:#d4edda;color:#155724;}
.by{background:#fff3cd;color:#856404;}
.br{background:#f8d7da;color:#721c24;}

.rcard {
    border-radius:10px; padding:8px 11px; margin:3px 0;
    border-left:4px solid; animation:popIn .3s ease;
}
@keyframes popIn{0%{opacity:0;transform:scale(.9)}100%{opacity:1;transform:scale(1)}}
.r-low {background:linear-gradient(135deg,#43C6AC,#191654);border-color:#43C6AC;}
.r-med {background:linear-gradient(135deg,#F2994A,#F2C94C);border-color:#F2994A;}
.r-high{background:linear-gradient(135deg,#EB5757,#FF6B9D);border-color:#EB5757;}
.rcard p{color:white!important;margin:0!important;}
.rcard .rt{font-size:1.05rem;font-weight:900;}
.rcard .rp{font-size:.86rem;margin-top:2px!important;}
.rcard .rr{font-size:.74rem;opacity:.9;margin-top:1px!important;}

.stButton>button {
    background:linear-gradient(135deg,#FF6B9D,#C44569)!important;
    color:white!important; font-weight:800!important; font-size:.9rem!important;
    padding:4px 0!important; height:34px!important; border-radius:8px!important;
    border:none!important; width:100%!important;
    box-shadow:0 3px 12px rgba(255,107,157,.42)!important; transition:all .25s!important;
}
.stButton>button:hover {
    transform:translateY(-1px)!important;
    box-shadow:0 5px 18px rgba(255,107,157,.58)!important;
}
[data-testid="metric-container"] {
    background:linear-gradient(135deg,#fafafa,#f0f0f0)!important;
    border-radius:7px!important; padding:3px 8px!important;
    border-left:3px solid #FF6B9D!important;
    box-shadow:0 1px 5px rgba(0,0,0,.07)!important;
}
[data-testid="metric-container"] label{font-size:.7rem!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-size:1rem!important;}

.disc {
    font-size:.72rem; color:white; text-align:center;
    background:linear-gradient(135deg,#C44569,#FF6B9D);
    border-radius:10px;
    padding:8px 16px;
    margin-top:8px;
    box-shadow:0 2px 8px rgba(196,69,105,0.25);
}
hr{margin:3px 0!important;border-color:rgba(196,69,105,.15)!important;}
</style>
""", unsafe_allow_html=True)

# ── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("heart_disease_model.joblib")
    except:
        return None

model = load_model()

# ── PDF generator using ReportLab (true .pdf, no HTML) ──────────────────────
def make_pdf_bytes(d, pred, prob, risk, rec, rfs, notes, pname, pdob, pref):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable)
    from reportlab.graphics.shapes import Rect, String, Drawing

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=15*mm, rightMargin=15*mm,
                             topMargin=12*mm, bottomMargin=12*mm)

    PINK   = colors.HexColor("#C44569")
    LPINK  = colors.HexColor("#FF6B9D")
    RC     = {"LOW":colors.HexColor("#27AE60"),
               "MEDIUM":colors.HexColor("#F39C12"),
               "HIGH":colors.HexColor("#E74C3C")}[risk]
    RBG    = {"LOW":colors.HexColor("#EAFAF1"),
               "MEDIUM":colors.HexColor("#FEF9E7"),
               "HIGH":colors.HexColor("#FDEDEC")}[risk]
    GREY   = colors.HexColor("#888888")
    CARD   = colors.HexColor("#FAFAFA")
    BORD   = colors.HexColor("#EEEEEE")

    def P(text, size=9, color=colors.HexColor("#222222"), bold=False,
          align=TA_LEFT, leading=None):
        fn = "Helvetica-Bold" if bold else "Helvetica"
        return Paragraph(str(text), ParagraphStyle("p",
            fontSize=size, textColor=color, fontName=fn,
            alignment=align, leading=leading or size*1.35))

    W = A4[0] - 30*mm
    story = []

    # ── HEADER ──
    hdr = Table([[
        P("❤️  Heart Disease Risk Assessment Report", 15, colors.white, True),
        P(f"ID: HDR-{datetime.now().strftime('%Y%m%d%H%M%S')}<br/>"
          f"{datetime.now().strftime('%d %B %Y, %H:%M')}<br/>"
          "EDUCATIONAL USE ONLY", 7.5, colors.white, align=TA_RIGHT)
    ],[
        P("Logistic Regression · ROC-AUC: 0.9154 · UCI Heart Disease Dataset",
          8, colors.white),
        ""
    ]], colWidths=[W*0.65, W*0.35])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), PINK),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0),(-1,-1), 9),
        ("BOTTOMPADDING",(0,0),(-1,-1), 7),
        ("LEFTPADDING",  (0,0),(-1,-1), 10),
        ("RIGHTPADDING", (0,0),(-1,-1), 10),
        ("SPAN",         (0,0),(0,1)),
    ]))
    story += [hdr, Spacer(1, 4*mm)]

    # ── DISCLAIMER ──
    disc = Table([[P("⚠  DISCLAIMER: This is an educational ML tool only. It does not "
                     "constitute medical advice, diagnosis, or treatment. Always consult "
                     "a qualified healthcare professional.", 8,
                     colors.HexColor("#856404"))]], colWidths=[W])
    disc.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), colors.HexColor("#FFF3CD")),
        ("BOX",          (0,0),(-1,-1), 0.5, colors.HexColor("#FFEEBA")),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
        ("TOPPADDING",   (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]))
    story += [disc, Spacer(1, 3*mm)]

    # ── PATIENT ROW ──
    pt_parts = []
    if pname: pt_parts.append(f"<b>Patient:</b> {pname}")
    if pdob:  pt_parts.append(f"<b>DOB:</b> {pdob}")
    if pref:  pt_parts.append(f"<b>Clinician:</b> {pref}")
    if pt_parts:
        pt = Table([[P("  |  ".join(pt_parts), 9,
                       colors.HexColor("#333333"))]], colWidths=[W])
        pt.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,-1), colors.HexColor("#F0F4FF")),
            ("BOX",          (0,0),(-1,-1), 0.5, colors.HexColor("#C7D2FE")),
            ("LEFTPADDING",  (0,0),(-1,-1), 8),
            ("TOPPADDING",   (0,0),(-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ]))
        story += [pt, Spacer(1, 3*mm)]

    # ── RISK BANNER ──
    ri = {"LOW":"✓","MEDIUM":"!","HIGH":"⚠"}[risk]
    rb = Table([[
        P(ri, 16, colors.white, True, TA_CENTER),
        [P(f"{risk} RISK", 14, RC, True),
         P(f"Disease probability: <b>{prob*100:.1f}%</b>  |  "
           f"{'Heart Disease Detected' if pred==1 else 'No Heart Disease Detected'}",
           9, colors.HexColor("#555555"))],
        [P("📋  RECOMMENDATION", 8, RC, True),
         Spacer(1, 2),
         P(rec, 9, colors.HexColor("#444444"))]
    ]], colWidths=[11*mm, W*0.53, W*0.35])
    rb.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), RBG),
        ("BOX",          (0,0),(-1,-1), 1.0, RC),
        ("BACKGROUND",   (0,0),(0,0),   RC),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("LINEAFTER",    (1,0),(1,0),   0.5, RC),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
        ("TOPPADDING",   (0,0),(-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
    ]))
    story += [rb, Spacer(1, 3*mm)]

    # ── GAUGE ──
    story.append(P("RISK PROBABILITY GAUGE", 7, GREY, True))
    story.append(Spacer(1, 1*mm))
    gw, gh = W, 10
    g = Drawing(gw, gh + 14)
    g.add(Rect(0,    14, gw*0.30, gh, fillColor=colors.HexColor("#27AE60"), strokeColor=None))
    g.add(Rect(gw*0.30, 14, gw*0.40, gh, fillColor=colors.HexColor("#F39C12"), strokeColor=None))
    g.add(Rect(gw*0.70, 14, gw*0.30, gh, fillColor=colors.HexColor("#E74C3C"), strokeColor=None))
    nx = gw * min(max(prob, 0.01), 0.99)
    g.add(Rect(nx-1.5, 11, 3, gh+5, fillColor=colors.HexColor("#1a1a1a"), strokeColor=None))
    for pct, lbl in [(0,"0%"),(0.25,"25%"),(0.50,"50%"),(0.75,"75%"),(1.0,"100%")]:
        g.add(String(gw*pct, 2, lbl, fontSize=7, fillColor=colors.HexColor("#999999"),
                     textAnchor="middle"))
    story += [g, Spacer(1, 3*mm)]

    # ── DATA CARDS helper ──
    def card(title, rows):
        inner = Table([[P(k, 9, GREY), P(v, 9, colors.HexColor("#222"), True)]
                        for k,v in rows], colWidths=[W*0.22, W*0.25])
        inner.setStyle(TableStyle([
            ("LINEBELOW",    (0,0),(-1,-2), 0.3, colors.HexColor("#F0F0F0")),
            ("TOPPADDING",   (0,0),(-1,-1), 2),
            ("BOTTOMPADDING",(0,0),(-1,-1), 2),
            ("LEFTPADDING",  (0,0),(-1,-1), 0),
            ("RIGHTPADDING", (0,0),(-1,-1), 0),
        ]))
        outer = Table([[P(title, 8, PINK, True)],[inner]], colWidths=[None])
        outer.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,-1), CARD),
            ("BOX",          (0,0),(-1,-1), 0.5, BORD),
            ("LINEBELOW",    (0,0),(0,0),   0.5, colors.HexColor("#F0E0E5")),
            ("LEFTPADDING",  (0,0),(-1,-1), 8),
            ("RIGHTPADDING", (0,0),(-1,-1), 8),
            ("TOPPADDING",   (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ]))
        return outer

    sex_s = "Male" if d["sex"]==1 else "Female"
    cp_s  = ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"][d["cp"]]
    ecg_s = ["Normal","ST-T Abnormality","LV Hypertrophy"][d["restecg"]]
    sl_s  = ["Upsloping","Flat","Downsloping"][d["slope"]]
    th_s  = ["Normal","Fixed Defect","Reversible Defect","Unknown"][d["thal"]]
    fbs_s = "Yes" if d["fbs"]==1 else "No"
    ex_s  = "Yes" if d["exang"]==1 else "No"
    cw = (W - 4*mm) / 2

    row1 = Table([[
        card("👤  DEMOGRAPHICS & VITALS", [
            ("Age",f"{d['age']} years"),("Sex",sex_s),
            ("Resting BP",f"{d['trestbps']} mm Hg"),
            ("Cholesterol",f"{d['chol']} mg/dL"),
            ("Max Heart Rate",f"{d['thalach']} bpm")]),
        card("🏥  BLOOD & ECG", [
            ("Fasting Sugar >120",fbs_s),("Resting ECG",ecg_s),
            ("ST Depression",str(d['oldpeak'])),("ST Slope",sl_s)])
    ]], colWidths=[cw, cw])
    row1.setStyle(TableStyle([
        ("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2),
        ("VALIGN",(0,0),(-1,-1),"TOP")]))
    story += [row1, Spacer(1, 2*mm)]

    rf_items = ([P(f"⚠  {r}", 9, colors.HexColor("#333")) for r in rfs]
                if rfs else [P("✓  No major risk factors identified", 9,
                               colors.HexColor("#27AE60"), True)])
    rf_inner = Table([[item] for item in rf_items], colWidths=[None])
    rf_inner.setStyle(TableStyle([
        ("TOPPADDING",(0,0),(-1,-1),1),("BOTTOMPADDING",(0,0),(-1,-1),1),
        ("LEFTPADDING",(0,0),(-1,-1),0)]))
    rf_card = Table([
        [P("⚠️  RISK FACTORS IDENTIFIED", 8, PINK, True)],
        [rf_inner]
    ], colWidths=[None])
    rf_card.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),CARD),("BOX",(0,0),(-1,-1),0.5,BORD),
        ("LINEBELOW",(0,0),(0,0),0.5,colors.HexColor("#F0E0E5")),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))

    row2 = Table([[
        card("💊  CLINICAL FINDINGS", [
            ("Chest Pain",cp_s),("Exercise Angina",ex_s),
            ("Major Vessels",str(d['ca'])),("Thalassemia",th_s)]),
        rf_card
    ]], colWidths=[cw, cw])
    row2.setStyle(TableStyle([
        ("LEFTPADDING",(0,0),(-1,-1),2),("RIGHTPADDING",(0,0),(-1,-1),2),
        ("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.append(row2)

    # ── CLINICIAN NOTES ──
    if notes.strip():
        story.append(Spacer(1, 3*mm))
        nt = Table([
            [P("📝  CLINICIAN NOTES", 8, colors.HexColor("#E67E22"), True)],
            [P(notes.replace('\n','<br/>'), 9, colors.HexColor("#444444"), leading=13)]
        ], colWidths=[W])
        nt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#FFF8E7")),
            ("BOX",(0,0),(-1,-1),0.8,colors.HexColor("#F39C12")),
            ("LINEBELOW",(0,0),(0,0),0.5,colors.HexColor("#F39C12")),
            ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
            ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        story.append(nt)

    # ── FOOTER ──
    story += [Spacer(1,4*mm), HRFlowable(width=W, thickness=0.5, color=BORD),
              Spacer(1,1*mm)]
    ft = Table([[
        P("Heart Disease Prediction System · Logistic Regression · "
          "UCI (n=302) · 5-Fold CV · Acc: 83.6% · Recall: 84.9%",
          7.5, GREY),
        P("⚠  Educational Tool — Not for Clinical Diagnosis",
          7.5, PINK, True, TA_RIGHT)
    ]], colWidths=[W*0.6, W*0.4])
    ft.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
        ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0)]))
    story.append(ft)

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
  background: linear-gradient(135deg, #C44569 0%, #FF6B9D 60%, #FFA07A 100%);
  border-radius: 12px;
  padding: 12px 20px;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 4px 18px rgba(196,69,105,0.4);
">
  <div style="display:flex;align-items:center;gap:12px;">
    <span style="font-size:2.2rem;line-height:1;">❤️</span>
    <div>
      <div style="font-size:1.5rem;font-weight:900;color:white;
                  letter-spacing:-0.5px;line-height:1.15;
                  text-shadow:0 1px 6px rgba(0,0,0,0.2);">
        Heart Disease Risk Predictor
      </div>
      <div style="font-size:0.75rem;color:rgba(255,255,255,0.88);
                  margin-top:2px;font-weight:500;">
        Logistic Regression &nbsp;·&nbsp; ROC-AUC 0.9154 &nbsp;·&nbsp; UCI Heart Disease Dataset
      </div>
    </div>
  </div>
  <div style="
    background:rgba(255,255,255,0.22);
    border:1.5px solid rgba(255,255,255,0.4);
    border-radius:20px;
    padding:5px 14px;
    font-size:0.73rem;
    color:white;
    font-weight:700;
    white-space:nowrap;
  ">⚠️ Educational Tool Only</div>
</div>
""", unsafe_allow_html=True)

# ── Patient row ───────────────────────────────────────────────────────────────
r0c1, r0c2, r0c3 = st.columns([2, 1.5, 1.5])
pname = r0c1.text_input("Patient Name / ID",  placeholder="e.g. Patient 001", key="w_pname")
pdob  = r0c2.date_input("Date of Birth", value=None,
                         min_value=date(1900,1,1), max_value=date.today(),
                         key="w_pdob", format="DD/MM/YYYY")
pref  = r0c3.text_input("Referring Clinician", placeholder="Dr. ...", key="w_pref")
st.markdown("<hr style='margin:3px 0 4px 0;'/>", unsafe_allow_html=True)

# ── 5 columns ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.15])

# COL 1
with c1:
    st.markdown("<div class='spill'>👤 Demographics & Vitals</div>", unsafe_allow_html=True)
    age      = st.number_input("Age (yrs)", 1, 120, 50, key="w_age")
    ab = ("bg","Low risk") if age<45 else ("by","Moderate") if age<60 else ("br","Elevated")
    st.markdown(f"<span class='badge {ab[0]}'>{ab[1]}</span>", unsafe_allow_html=True)
    sex      = st.selectbox("Sex", [0,1],
                             format_func=lambda x:"♀ Female" if x==0 else "♂ Male", key="w_sex")
    trestbps = st.number_input("Resting BP (mmHg)", 50, 250, 120, key="w_bp")
    bb = ("bg","Normal") if trestbps<120 else ("by","Elevated") if trestbps<140 else ("br","High BP")
    st.markdown(f"<span class='badge {bb[0]}'>{bb[1]}</span>", unsafe_allow_html=True)
    chol     = st.number_input("Cholesterol (mg/dL)", 50, 600, 200, key="w_chol")
    cb = ("bg","Desirable") if chol<200 else ("by","Borderline") if chol<240 else ("br","High")
    st.markdown(f"<span class='badge {cb[0]}'>{cb[1]}</span>", unsafe_allow_html=True)
    thalach  = st.number_input("Max HR (bpm)", 50, 250, 150, key="w_hr")
    pct = int(thalach / max(220 - age, 1) * 100)
    hb = ("bg",f"{pct}%") if pct>=85 else ("by",f"{pct}%") if pct>=70 else ("br",f"{pct}%")
    st.markdown(f"<span class='badge {hb[0]}'>HR {hb[1]} est.max</span>", unsafe_allow_html=True)

# COL 2
with c2:
    st.markdown("<div class='spill'>🏥 Blood & ECG</div>", unsafe_allow_html=True)
    fbs     = st.selectbox("Fasting Sugar >120", [0,1],
                            format_func=lambda x:"No" if x==0 else "Yes", key="w_fbs")
    if fbs==1:
        st.markdown("<span class='badge br'>Elevated glucose</span>", unsafe_allow_html=True)
    restecg = st.selectbox("Resting ECG", [0,1,2],
                            format_func=lambda x:["Normal","ST-T Abnormal","LVH"][x], key="w_ecg")
    if restecg>0:
        st.markdown("<span class='badge by'>ECG abnormality</span>", unsafe_allow_html=True)
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, 0.1, key="w_st")
    ob = ("bg","None") if oldpeak==0 else ("by","Mild") if oldpeak<=2 else ("br","Significant")
    st.markdown(f"<span class='badge {ob[0]}'>ST: {ob[1]}</span>", unsafe_allow_html=True)
    slope   = st.selectbox("ST Slope", [0,1,2],
                            format_func=lambda x:["Upsloping","Flat","Downsloping"][x], key="w_slope")
    slb = ("bg","Favourable") if slope==0 else ("by","Intermediate") if slope==1 else ("br","Ischaemic")
    st.markdown(f"<span class='badge {slb[0]}'>{slb[1]}</span>", unsafe_allow_html=True)

# COL 3
with c3:
    st.markdown("<div class='spill'>💊 Clinical Findings</div>", unsafe_allow_html=True)
    cp    = st.selectbox("Chest Pain", [0,1,2,3],
                          format_func=lambda x:["Typical Angina","Atypical",
                                                "Non-anginal","Asymptomatic"][x], key="w_cp")
    cpb = ("br","Classic cardiac") if cp==0 else ("by","Possible") if cp==1 \
          else ("bg","Less likely") if cp==2 else ("by","Evaluate")
    st.markdown(f"<span class='badge {cpb[0]}'>{cpb[1]}</span>", unsafe_allow_html=True)
    exang = st.selectbox("Exercise Angina", [0,1],
                          format_func=lambda x:"No" if x==0 else "Yes", key="w_exang")
    st.markdown(
        f"<span class='badge {'br' if exang==1 else 'bg'}'>{'Positive' if exang==1 else 'Negative'}</span>",
        unsafe_allow_html=True)
    ca    = st.selectbox("Major Vessels (0–4)", [0,1,2,3,4], key="w_ca")
    st.markdown(f"<span class='badge {['bg','by','br','br','br'][ca]}'>"
                f"{['No blockage','1 blocked','2 blocked','3 blocked','4 blocked'][ca]}</span>",
                unsafe_allow_html=True)
    thal  = st.selectbox("Thalassemia", [0,1,2,3],
                          format_func=lambda x:["Normal","Fixed Defect",
                                                "Reversible","Unknown"][x], key="w_thal")
    thb = ("bg","Normal") if thal==0 else ("by","Permanent") if thal==1 \
          else ("br","Ischaemic") if thal==2 else ("by","Unknown")
    st.markdown(f"<span class='badge {thb[0]}'>{thb[1]}</span>", unsafe_allow_html=True)

# COL 4
with c4:
    st.markdown("<div class='spill'>📝 Notes & Predict</div>", unsafe_allow_html=True)
    notes = st.text_area("Clinician Notes",
                          placeholder="Clinical observations, medications,\nhistory... (included in PDF)",
                          height=155, key="w_notes", label_visibility="visible")
    st.markdown("<div style='height:3px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Risk", type="primary", use_container_width=True)
    st.markdown("""
    <div style="margin-top:5px;padding:5px 8px;background:#FFF3CD;border-radius:6px;
      border-left:3px solid #F39C12;font-size:.74rem;color:#856404;line-height:1.45;">
      ⚠️ <strong>Educational tool only.</strong> Not for medical diagnosis.<br>
      LOW &lt;30% · MEDIUM 30–70% · HIGH &gt;70%
    </div>""", unsafe_allow_html=True)

# COL 5 — Results
with c5:
    st.markdown("<div class='spill'>📊 Prediction Results</div>", unsafe_allow_html=True)

    if predict_btn:
        inp = pd.DataFrame([{
            'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,'chol':chol,
            'fbs':fbs,'restecg':restecg,'thalach':thalach,'exang':exang,
            'oldpeak':oldpeak,'slope':slope,'ca':ca,'thal':thal
        }])
        if model is not None:
            try:
                _pred = int(model.predict(inp)[0])
                _prob = float(model.predict_proba(inp)[0, 1])
            except Exception as e:
                st.error(f"Prediction error: {e}")
                _pred, _prob = 0, 0.5
        else:
            import random
            random.seed(age + sex + cp + ca + int(oldpeak * 10))
            _prob = round(random.uniform(0.2, 0.8), 3)
            _pred = 1 if _prob > 0.5 else 0
            st.caption("⚠️ Demo — model file not found")

        # Store with res_ prefix — never conflicts with w_ widget keys
        for k,v in [("res_pred",_pred),("res_prob",_prob),
                     ("res_age",age),("res_sex",sex),("res_cp",cp),
                     ("res_bp",trestbps),("res_chol",chol),("res_fbs",fbs),
                     ("res_ecg",restecg),("res_hr",thalach),("res_exang",exang),
                     ("res_st",oldpeak),("res_slope",slope),("res_ca",ca),
                     ("res_thal",thal),("res_notes",notes),
                     ("res_pname",pname),("res_pdob",str(pdob) if pdob else ""),
                     ("res_pref",pref)]:
            st.session_state[k] = v

    if "res_prob" in st.session_state:
        prob  = st.session_state["res_prob"]
        pred  = st.session_state["res_pred"]
        risk  = "LOW" if prob<0.3 else "MEDIUM" if prob<0.7 else "HIGH"
        rcls  = {"LOW":"r-low","MEDIUM":"r-med","HIGH":"r-high"}[risk]
        ricon = {"LOW":"✅","MEDIUM":"⚡","HIGH":"⚠️"}[risk]
        rclr  = {"LOW":"#27AE60","MEDIUM":"#F39C12","HIGH":"#E74C3C"}[risk]
        rec   = {"LOW":"Maintain healthy lifestyle & annual check-ups.",
                 "MEDIUM":"Schedule GP appointment within 4 weeks.",
                 "HIGH":"Seek urgent cardiology referral immediately."}[risk]

        st.markdown(f"""
        <div class="rcard {rcls}">
          <p class="rt">{ricon} {risk} RISK</p>
          <p class="rp">Probability: <strong>{prob*100:.1f}%</strong></p>
          <p class="rr">{'❤️ Heart Disease Detected' if pred==1 else '💚 No Disease Detected'}</p>
        </div>""", unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("Confidence", f"{max(prob,1-prob)*100:.0f}%")
        m2.metric("Result", "Disease" if pred==1 else "Healthy")

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob*100,
            number={'font':{'size':16,'color':rclr},'suffix':'%'},
            title={'text':"Risk %",'font':{'size':10,'color':'#C44569'}},
            gauge={'axis':{'range':[0,100],'tickfont':{'size':8}},
                   'bar':{'color':rclr,'thickness':0.65},
                   'steps':[{'range':[0,30],'color':'rgba(39,174,96,0.12)'},
                             {'range':[30,70],'color':'rgba(243,156,18,0.12)'},
                             {'range':[70,100],'color':'rgba(231,76,60,0.12)'}],
                   'threshold':{'line':{'color':'#C44569','width':2},'value':50}}
        ))
        fig.update_layout(height=130, margin=dict(l=4,r=4,t=22,b=2),
                          paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"<p style='font-size:.8rem;color:{rclr};font-weight:700;margin:0;'>💡 {rec}</p>",
                    unsafe_allow_html=True)

        d = {k2:st.session_state[k2] for k2 in
             ["res_age","res_sex","res_cp","res_bp","res_chol","res_fbs",
              "res_ecg","res_hr","res_exang","res_st","res_slope","res_ca","res_thal"]}
        d2 = {'age':d['res_age'],'sex':d['res_sex'],'cp':d['res_cp'],
              'trestbps':d['res_bp'],'chol':d['res_chol'],'fbs':d['res_fbs'],
              'restecg':d['res_ecg'],'thalach':d['res_hr'],'exang':d['res_exang'],
              'oldpeak':d['res_st'],'slope':d['res_slope'],'ca':d['res_ca'],
              'thal':d['res_thal']}
        rfs = []
        if d2['age']      >55:  rfs.append(f"Age {d2['age']} yrs — elevated risk >55")
        if d2['chol']     >240: rfs.append(f"Cholesterol {d2['chol']} mg/dL — high")
        if d2['trestbps'] >140: rfs.append(f"BP {d2['trestbps']} mmHg — hypertensive")
        if d2['fbs']      ==1:  rfs.append("Fasting blood sugar >120 mg/dL")
        if d2['exang']    ==1:  rfs.append("Exercise-induced angina present")
        if d2['oldpeak']  >2:   rfs.append(f"ST depression {d2['oldpeak']} — significant")
        if d2['ca']       >0:   rfs.append(f"{d2['ca']} major vessel(s) blocked")
        if d2['thal']     ==2:  rfs.append("Reversible thalassemia defect")
        if d2['cp']       ==0:  rfs.append("Typical angina — classic cardiac")

        if rfs:
            with st.expander(f"⚠️ {len(rfs)} risk factor(s)", expanded=False):
                for rf in rfs:
                    st.markdown(f"<span style='font-size:.78rem;'>🔸 {rf}</span>",
                                unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge bg' style='font-size:.74rem;'>"
                        "✅ No major risk factors</span>", unsafe_allow_html=True)

        # TRUE PDF download
        pdf_bytes = make_pdf_bytes(
            d=d2, pred=pred, prob=prob, risk=risk, rec=rec, rfs=rfs,
            notes=st.session_state.get("res_notes",""),
            pname=st.session_state.get("res_pname",""),
            pdob=st.session_state.get("res_pdob",""),
            pref=st.session_state.get("res_pref","")
        )
        pn    = (st.session_state.get("res_pname","") or "Patient").replace(" ","_")
        fname = f"HeartRisk_{pn}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div style="text-align:center;padding:30px 8px;">
          <div style="font-size:2.4rem;">❤️</div>
          <div style="font-size:.84rem;color:#bbb;margin-top:8px;">
            Fill in patient data<br>and click <strong>Predict Risk</strong>
          </div>
        </div>""", unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disc">
  ❤️ Heart Disease Risk Predictor &nbsp;·&nbsp;
  Logistic Regression &nbsp;·&nbsp; ROC-AUC 0.9154 &nbsp;·&nbsp;
  UCI Dataset (n=302) &nbsp;·&nbsp; 5-Fold CV &nbsp;·&nbsp;
  <strong>⚠️ Educational Tool — Not for Clinical Diagnosis</strong>
</div>
""", unsafe_allow_html=True)
