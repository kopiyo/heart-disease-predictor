<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Heart Disease Risk Predictor — README</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
  :root {
    --pink:    #C44569;
    --lpink:   #FF6B9D;
    --rose:    #FFA07A;
    --dark:    #1a0a14;
    --card:    #ffffff;
    --muted:   #6b7280;
    --border:  #f0e0e8;
    --text:    #1f1220;
    --accent:  #f3e8ee;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'DM Sans', sans-serif;
    background: #fdf4f7;
    color: var(--text);
    min-height: 100vh;
  }

  /* ── HERO ── */
  .hero {
    background: linear-gradient(135deg, #C44569 0%, #FF6B9D 55%, #FFA07A 100%);
    padding: 56px 40px 48px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  }
  .hero-icon { font-size: 3.5rem; display: block; margin-bottom: 14px; filter: drop-shadow(0 4px 12px rgba(0,0,0,0.2)); }
  .hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(1.8rem, 4vw, 3rem);
    color: white;
    letter-spacing: -0.5px;
    margin-bottom: 12px;
    text-shadow: 0 2px 12px rgba(0,0,0,0.15);
  }
  .hero p {
    color: rgba(255,255,255,0.88);
    font-size: 1rem;
    max-width: 540px;
    margin: 0 auto 22px;
    line-height: 1.65;
  }
  .hero-link {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1.5px solid rgba(255,255,255,0.45);
    color: white;
    text-decoration: none;
    padding: 9px 22px;
    border-radius: 30px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    backdrop-filter: blur(8px);
    transition: background 0.2s;
  }
  .hero-link:hover { background: rgba(255,255,255,0.28); }

  /* ── STATS BAR ── */
  .stats-bar {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0;
    background: white;
    border-bottom: 1px solid var(--border);
    box-shadow: 0 2px 12px rgba(196,69,105,0.08);
  }
  .stat {
    padding: 18px 32px;
    text-align: center;
    border-right: 1px solid var(--border);
    flex: 1;
    min-width: 120px;
  }
  .stat:last-child { border-right: none; }
  .stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem;
    color: var(--pink);
    line-height: 1;
    margin-bottom: 4px;
  }
  .stat-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
  }

  /* ── LAYOUT ── */
  .container {
    max-width: 900px;
    margin: 0 auto;
    padding: 48px 24px;
    display: grid;
    gap: 32px;
  }

  /* ── SECTION CARD ── */
  .card {
    background: white;
    border-radius: 16px;
    border: 1px solid var(--border);
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(196,69,105,0.06);
  }
  .card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 16px 24px;
    background: linear-gradient(135deg, #fdf0f4 0%, #fef9f0 100%);
    border-bottom: 1px solid var(--border);
  }
  .card-header-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, var(--pink), var(--lpink));
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
  }
  .card-header h2 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: var(--text);
    font-weight: 400;
  }
  .card-body { padding: 24px; }

  /* ── PROSE ── */
  .prose {
    font-size: 0.92rem;
    line-height: 1.75;
    color: #374151;
  }

  /* ── TABLE ── */
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.87rem;
  }
  thead tr {
    background: linear-gradient(135deg, #fdf0f4, #fef9f0);
  }
  th {
    text-align: left;
    padding: 10px 14px;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: var(--pink);
    font-weight: 700;
    border-bottom: 2px solid var(--border);
  }
  td {
    padding: 10px 14px;
    border-bottom: 1px solid #faf0f4;
    color: #374151;
    vertical-align: top;
  }
  tr:last-child td { border-bottom: none; }
  tbody tr:hover { background: #fdf8fa; }
  td:first-child { font-weight: 600; color: var(--text); }
  td code, th code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    background: var(--accent);
    padding: 1px 6px;
    border-radius: 4px;
    color: var(--pink);
  }
  td.num {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    color: var(--pink);
    font-size: 0.95rem;
  }

  /* ── NOTE BOX ── */
  .note {
    margin-top: 16px;
    padding: 12px 16px;
    background: #fdf0f4;
    border-left: 3px solid var(--lpink);
    border-radius: 0 8px 8px 0;
    font-size: 0.84rem;
    color: #555;
    line-height: 1.65;
  }

  /* ── FEATURE LIST ── */
  .feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 10px;
  }
  .feature-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 12px;
    background: #fdf8fa;
    border-radius: 8px;
    border: 1px solid var(--border);
  }
  .feat-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    color: white;
    background: linear-gradient(135deg, var(--pink), var(--lpink));
    border-radius: 4px;
    padding: 2px 6px;
    flex-shrink: 0;
    margin-top: 1px;
  }
  .feat-name { font-weight: 700; font-size: 0.85rem; color: var(--text); }
  .feat-desc { font-size: 0.78rem; color: var(--muted); margin-top: 2px; line-height: 1.4; }

  /* ── OUTPUT LIST ── */
  .output-list { display: grid; gap: 10px; }
  .output-item {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: #fdf8fa;
  }
  .output-icon { font-size: 1.5rem; flex-shrink: 0; }
  .output-label { font-weight: 700; font-size: 0.88rem; color: var(--text); }
  .output-desc { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }

  /* ── CODE BLOCK ── */
  .code-wrap { position: relative; margin-top: 4px; }
  pre {
    background: #1e1028;
    border-radius: 10px;
    padding: 18px 20px;
    overflow-x: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    color: #e2c8f0;
  }
  pre .c  { color: #9ca3af; } /* comment */
  pre .k  { color: #f9a8d4; } /* keyword */
  pre .s  { color: #86efac; } /* string */
  pre .p  { color: #fbbf24; } /* path */
  .copy-btn {
    position: absolute;
    top: 10px; right: 10px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15);
    color: #ccc;
    font-size: 0.72rem;
    padding: 4px 10px;
    border-radius: 5px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    transition: all 0.2s;
  }
  .copy-btn:hover { background: rgba(255,255,255,0.18); color: white; }
  .copy-btn.ok { background: rgba(39,174,96,0.3); color: #6ee7b7; border-color: rgba(39,174,96,0.4); }

  /* ── STEP ── */
  .step {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 14px;
  }
  .step-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    background: #fdf0f4;
    border-bottom: 1px solid var(--border);
  }
  .step-num {
    width: 24px; height: 24px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--pink), var(--lpink));
    color: white;
    font-size: 0.75rem;
    font-weight: 800;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  .step-title { font-weight: 700; font-size: 0.88rem; color: var(--text); }
  .step pre { border-radius: 0; }

  /* ── DISCLAIMER ── */
  .disclaimer {
    background: #fff8f0;
    border: 1px solid #fbbf24;
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 0.86rem;
    color: #92400e;
    line-height: 1.7;
    margin-bottom: 20px;
  }
  .author-card {
    background: linear-gradient(135deg, #fdf0f4, #f0f4ff);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 22px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .author-avatar {
    width: 48px; height: 48px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--pink), var(--lpink));
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    flex-shrink: 0;
  }
  .author-name { font-weight: 700; font-size: 1rem; color: var(--text); margin-bottom: 3px; }
  .author-dept { font-size: 0.82rem; color: var(--muted); line-height: 1.5; }

  /* ── FOOTER ── */
  footer {
    text-align: center;
    padding: 24px;
    font-size: 0.78rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
    background: white;
  }
  footer strong { color: var(--pink); }
</style>
</head>
<body>

<!-- HERO -->
<div class="hero">
  <span class="hero-icon">❤️</span>
  <h1>Heart Disease Risk Predictor</h1>
  <p>A machine learning web application for predicting coronary heart disease risk from clinical patient data. Built with Streamlit and deployed on Streamlit Cloud.</p>
  <a class="hero-link" href="https://heart-disease-prediction-app.streamlit.app" target="_blank">
    🔗 heart-disease-prediction-app.streamlit.app
  </a>
</div>

<!-- STATS BAR -->
<div class="stats-bar">
  <div class="stat"><div class="stat-val">SVC</div><div class="stat-label">Algorithm</div></div>
  <div class="stat"><div class="stat-val">0.9481</div><div class="stat-label">ROC-AUC</div></div>
  <div class="stat"><div class="stat-val">94.12%</div><div class="stat-label">Recall</div></div>
  <div class="stat"><div class="stat-val">91.43%</div><div class="stat-label">F1-Score</div></div>
  <div class="stat"><div class="stat-val">918</div><div class="stat-label">Records</div></div>
</div>

<div class="container">

  <!-- OVERVIEW -->
  <div class="card">
    <div class="card-header">
      <div class="card-header-icon">📌</div>
      <h2>Overview</h2>
    </div>
    <div class="card-body">
      <p class="prose">This application takes 13 clinical measurements as input and returns a real-time heart disease risk prediction — classified as <strong>Low</strong>, <strong>Medium</strong>, or <strong>High</strong> risk — along with a predicted probability, a visual speedometer gauge, and a downloadable PDF patient report.</p>
      <p class="prose" style="margin-top:12px;">It was developed as part of a research project comparing multiple machine learning classifiers on the UCI Heart Disease dataset, demonstrating how a trained ML model can be made accessible to non-technical users through a clean web interface.</p>
    </div>
  </div>

  <!-- MODEL -->
  <div class="card">
    <div class="card-header">
      <div class="card-header-icon">🧠</div>
      <h2>Model</h2>
    </div>
    <div class="card-body">
      <table>
        <thead><tr><th>Detail</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Algorithm</td><td>Support Vector Classification (SVC, tuned)</td></tr>
          <tr><td>Dataset</td><td>Cleveland + Statlog + Hungarian (n = 918)</td></tr>
          <tr><td>Train / Test Split</td><td>80% / 20%, stratified</td></tr>
          <tr><td>Cross-Validation</td><td>5-fold GridSearchCV</td></tr>
          <tr><td>ROC-AUC</td><td class="num">0.9481</td></tr>
          <tr><td>Recall</td><td class="num">94.12%</td></tr>
          <tr><td>F1-Score</td><td class="num">91.43%</td></tr>
        </tbody>
      </table>
      <div class="note">💡 Four classifiers were compared during development: Logistic Regression, Random Forest, Gradient Boosting, and SVC. SVC with tuned hyperparameters achieved the best overall performance and was selected for deployment.</div>
    </div>
  </div>

  <!-- INPUT FEATURES -->
  <div class="card">
    <div class="card-header">
      <div class="card-header-icon">📋</div>
      <h2>Input Features</h2>
    </div>
    <div class="card-body">
      <p class="prose" style="margin-bottom:16px;">The app accepts <strong>13 clinical features</strong> collected during a standard patient consultation.</p>
      <div class="feature-grid">
        <div class="feature-item"><span class="feat-num">01</span><div><div class="feat-name">Age</div><div class="feat-desc">Patient age in years</div></div></div>
        <div class="feature-item"><span class="feat-num">02</span><div><div class="feat-name">Sex</div><div class="feat-desc">Male / Female</div></div></div>
        <div class="feature-item"><span class="feat-num">03</span><div><div class="feat-name">Chest Pain Type</div><div class="feat-desc">Typical angina, atypical, non-anginal, asymptomatic</div></div></div>
        <div class="feature-item"><span class="feat-num">04</span><div><div class="feat-name">Resting BP</div><div class="feat-desc">Resting blood pressure (mmHg)</div></div></div>
        <div class="feature-item"><span class="feat-num">05</span><div><div class="feat-name">Cholesterol</div><div class="feat-desc">Serum cholesterol (mg/dL)</div></div></div>
        <div class="feature-item"><span class="feat-num">06</span><div><div class="feat-name">Fasting Blood Sugar</div><div class="feat-desc">Whether fasting glucose &gt; 120 mg/dL</div></div></div>
        <div class="feature-item"><span class="feat-num">07</span><div><div class="feat-name">Resting ECG</div><div class="feat-desc">Normal, ST-T abnormality, or LV hypertrophy</div></div></div>
        <div class="feature-item"><span class="feat-num">08</span><div><div class="feat-name">Max Heart Rate</div><div class="feat-desc">Maximum heart rate achieved during exercise (bpm)</div></div></div>
        <div class="feature-item"><span class="feat-num">09</span><div><div class="feat-name">Exercise Angina</div><div class="feat-desc">Whether exercise induced angina</div></div></div>
        <div class="feature-item"><span class="feat-num">10</span><div><div class="feat-name">ST Depression</div><div class="feat-desc">ST depression induced by exercise relative to rest</div></div></div>
        <div class="feature-item"><span class="feat-num">11</span><div><div class="feat-name">ST Slope</div><div class="feat-desc">Slope of the peak exercise ST segment</div></div></div>
        <div class="feature-item"><span class="feat-num">12</span><div><div class="feat-name">Major Vessels</div><div class="feat-desc">Number of major vessels coloured by fluoroscopy (0–4)</div></div></div>
        <div class="feature-item"><span class="feat-num">13</span><div><div class="feat-name">Thalassemia</div><div class="feat-desc">Normal, fixed defect, reversible defect, or unknown</div></div></div>
      </div>
    </div>
  </div>

  <!-- OUTPUT -->
  <div class="card">
    <div class="card-header">
      <div class="card-header-icon">📊</div>
      <h2>Output</h2>
    </div>
    <div class="card-body">
      <div class="output-list">
        <div class="output-item"><span class="output-icon">🟢</span><div><div class="output-label">Risk Classification</div><div class="output-desc">LOW (&lt; 30%) · MEDIUM (30–70%) · HIGH (&gt; 70%)</div></div></div>
        <div class="output-item"><span class="output-icon">📈</span><div><div class="output-label">Predicted Probability</div><div class="output-desc">Displayed as a percentage on every prediction</div></div></div>
        <div class="output-item"><span class="output-icon">⏱️</span><div><div class="output-label">Speedometer Gauge</div><div class="output-desc">Interactive visual risk indicator built with Plotly</div></div></div>
        <div class="output-item"><span class="output-icon">⚠️</span><div><div class="output-label">Flagged Risk Factors</div><div class="output-desc">Automatically derived from the entered clinical values</div></div></div>
        <div class="output-item"><span class="output-icon">💡</span><div><div class="output-label">Clinical Recommendation</div><div class="output-desc">Personalised advice based on the risk tier result</div></div></div>
        <div class="output-item"><span class="output-icon">📄</span><div><div class="output-label">Downloadable PDF Report</div><div class="output-desc">Patient data, risk result, gauge, and clinician notes — generated with ReportLab</div></div></div>
      </div>
    </div>
  </div>

  <!-- TECH STACK -->
  <div class="card">
    <div class="card-header">
      <div class="card-header-icon">🛠️</div>
      <h2>Tech Stack</h2>
    </div>
    <div class="card-body">
      <table>
        <thead><tr><th>Component</th><th>Technology</th></tr></thead>
        <tbody>
          <tr><td>Web Framework</td><td>Streamlit</td></tr>
          <tr><td>ML / Preprocessing</td><td>scikit-learn, pandas, numpy</td></tr>
          <tr><td>Visualisation</td><td>Plotly (gauge chart)</td></tr>
          <tr><td>PDF Generation</td><td>ReportLab</td></tr>
          <tr><td>Model Serialisation</td><td>joblib</td></tr>
          <tr><td>Deployment</td><td>Streamlit Cloud</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- PROJECT STRUCTURE -->
  <div class="card">
    <div class="card-header">
      <div class="card-header-icon">🗂️</div>
      <h2>Project Structure</h2>
    </div>
    <div class="card-body">
      <div class="code-wrap">
        <pre>├── <span class="p">heart_Disease.py</span>               <span class="c"># Main Streamlit application</span>
├── <span class="p">heart_disease_model.joblib</span>     <span class="c"># Trained and serialised SVC pipeline</span>
├── <span class="p">heart.csv</span>                      <span class="c"># Source dataset (UCI Heart Disease)</span>
├── <span class="p">Heart_Disease_corrected.ipynb</span>  <span class="c"># Full analysis and model training notebook</span>
├── <span class="p">requirements.txt</span>               <span class="c"># Python dependencies</span>
└── <span class="p">README.md</span></pre>
        <button class="copy-btn" onclick="copyCode(this, '├── heart_Disease.py               # Main Streamlit application\n├── heart_disease_model.joblib     # Trained and serialised SVC pipeline\n├── heart.csv                      # Source dataset (UCI Heart Disease)\n├── Heart_Disease_corrected.ipynb  # Full analysis and model training notebook\n├── requirements.txt               # Python dependencies\n└── README.md')">Copy</button>
      </div>
    </div>
  </div>

  <!-- RUNNING LOCALLY -->
  <div class="card">
    <div class="card-header">
      <div class="card-header-icon">⚙️</div>
      <h2>Running Locally</h2>
    </div>
    <div class="card-body">
      <div class="step">
        <div class="step-header"><div class="step-num">1</div><div class="step-title">Clone the repository</div></div>
        <div class="code-wrap">
          <pre><span class="k">git clone</span> https://github.com/your-username/heart-disease-prediction-app.git
<span class="k">cd</span> heart-disease-prediction-app</pre>
          <button class="copy-btn" onclick="copyCode(this, 'git clone https://github.com/your-username/heart-disease-prediction-app.git\ncd heart-disease-prediction-app')">Copy</button>
        </div>
      </div>
      <div class="step">
        <div class="step-header"><div class="step-num">2</div><div class="step-title">Install dependencies</div></div>
        <div class="code-wrap">
          <pre><span class="k">pip install</span> -r requirements.txt</pre>
          <button class="copy-btn" onclick="copyCode(this, 'pip install -r requirements.txt')">Copy</button>
        </div>
      </div>
      <div class="step">
        <div class="step-header"><div class="step-num">3</div><div class="step-title">Run the app</div></div>
        <div class="code-wrap">
          <pre><span class="k">streamlit run</span> heart_Disease.py</pre>
          <button class="copy-btn" onclick="copyCode(this, 'streamlit run heart_Disease.py')">Copy</button>
        </div>
      </div>

      <p style="font-size:0.84rem;font-weight:700;color:var(--text);margin:20px 0 8px;">requirements.txt</p>
      <div class="code-wrap">
        <pre><span class="s">streamlit
pandas
numpy
scikit-learn
plotly
joblib
reportlab</span></pre>
        <button class="copy-btn" onclick="copyCode(this, 'streamlit\npandas\nnumpy\nscikit-learn\nplotly\njoblib\nreportlab')">Copy</button>
      </div>
    </div>
  </div>

  <!-- DISCLAIMER & AUTHOR -->
  <div class="card">
    <div class="card-header">
      <div class="card-header-icon">⚕️</div>
      <h2>Disclaimer &amp; Author</h2>
    </div>
    <div class="card-body">
      <div class="disclaimer">
        ⚠️ <strong>Educational Tool Only.</strong> This application does not constitute medical advice, diagnosis, or treatment. Predictions are based on a machine learning model trained on a research dataset and should never be used as a substitute for professional clinical judgement. Always consult a qualified healthcare professional for any medical concerns.
      </div>
      <div class="author-card">
        <div class="author-avatar">👩‍💻</div>
        <div>
          <div class="author-name">Diana Atieno Opiyo</div>
          <div class="author-dept">Department of Computing, Grand Valley State University<br/>Allendale, MI 49401, USA</div>
        </div>
      </div>
    </div>
  </div>

</div>

<footer>
  ❤️ Heart Disease Risk Predictor &nbsp;·&nbsp;
  <strong>SVC (Tuned)</strong> &nbsp;·&nbsp;
  ROC-AUC 0.9481 &nbsp;·&nbsp; Recall 94.12% &nbsp;·&nbsp;
  UCI Heart Disease Dataset &nbsp;·&nbsp;
  <strong>Educational Tool — Not for Clinical Diagnosis</strong>
</footer>

<script>
function copyCode(btn, text) {
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = '✓ Copied';
    btn.classList.add('ok');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('ok'); }, 2000);
  });
}
</script>
</body>
</html>
