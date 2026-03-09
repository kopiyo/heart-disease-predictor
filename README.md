**Heart Disease Risk Predictor**

A machine learning web application for predicting coronary heart disease (CHD) risk from clinical patient data.

**Overview**

This application takes 13 clinical measurements as input and returns a real-time heart disease risk prediction — classified as Low, Medium, or High risk — along with:

A predicted probability percentage
An interactive speedometer gauge
A downloadable PDF patient report

It was developed as part of a research project comparing multiple machine learning classifiers on the UCI Heart Disease dataset, demonstrating how a trained ML model can be made accessible to non-technical users through a clean web interface.

**Model Performance**
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/c5baf220-deca-4c1d-a71b-e38fa840fd53" />


**Four classifiers were compared during development:**
Logistic Regression, Random Forest, Gradient Boosting, and SVC. SVC with tuned hyperparameters achieved the best overall performance and was selected for deployment.


**Input Features**
The app accepts the following 13 clinical features collected during a standard patient consultation:
<img width="450" height="400" alt="image" src="https://github.com/user-attachments/assets/d6ace23b-ee3e-4cbe-bf34-9003a778d247" />


**Application Output**
<img width="777" height="365" alt="image" src="https://github.com/user-attachments/assets/e1fab426-87fe-40f5-ae9d-52f5f8127921" />



**Project Structure**

├── heart_Disease.py               # Main Streamlit application
├── heart_disease_model.joblib     # Trained and serialised SVC pipeline
├── heart_statlog_cleveland_hungary_final.xls                      # Source dataset (UCI Heart Disease)
├── Heart_Disease_v2.ipynb  # Full analysis and model training notebook
├── requirements.txt               # Python dependencies
└── README.md

**Disclaimer**

⚠️ This application is an educational tool only. It does not constitute medical advice, diagnosis, or treatment. Predictions are based on a machine learning model trained on a research dataset and should not be used as a substitute for professional clinical judgement. Always consult a qualified healthcare professional for any medical concerns.


**👩‍💻 Author**
**Diana Atieno Opiyo**
**Department of Computing, Grand Valley State University**
**Allendale, MI 49401, USA**
