# ❤️ Heart Disease Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heart-disease-risk-predictor-app.streamlit.app/)

> Cardiovascular risk assessment using Machine Learning

Interactive web application predicting heart disease risk using Logistic Regression. Achieves **91.5% ROC-AUC** on 302-patient UCI dataset with real-time risk assessment and clinical recommendations.

## Features

- 🎯 Real-time cardiovascular risk prediction
- 📈 83.6% accuracy, 91.5% ROC-AUC
- 💡 Clinical recommendations by risk level
- 📊 Interactive visualizations
- 📄 Downloadable assessment reports
- 🎨 Modern animated interface

## Quick Start

### Online
Visit the [live demo](https://heart-disease-risk-predictor-app.streamlit.app/)

### Local
```bash
git clone https://github.com/kopiyo/heart-disease-predictor.git
cd heart-disease-predictor
pip install -r requirements.txt
streamlit run heart_Disease.py
```

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 83.6% |
| ROC-AUC | 91.5% |
| Precision | 84.8% |
| Recall | 84.8% |

**Model**: Logistic Regression with Pipeline preprocessing  
**Dataset**: 302 unique patients (UCI Heart Disease)  
**Validation**: 5-fold cross-validation

## Tech Stack

- Streamlit • Scikit-learn • Pandas • Plotly

## 📁 Files

- `heart_Disease.py` - Main application
- `Heart_Disease.ipynb` - Model training
- `heart_disease_model.joblib` - Trained model
- `heart.csv` - Dataset

## ⚠️ Disclaimer

Educational tool only. NOT for medical diagnosis. Consult healthcare professionals for medical decisions.

## Author

**[Diana Opiyo]**  
[GitHub](https://github.com/kopiyo) • [LinkedIn](https://www.linkedin.com/in/diana-opiyo-680b98309/)

---

⭐ Star this repo if you find it useful!
