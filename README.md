# Credit Card Fraud Detection Using SMOTE

## Project Description
This project implements a machine learning based credit card fraud detection system.  
Due to extreme class imbalance in fraud datasets, SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the data and improve fraud detection performance.

The system supports batch prediction through CSV upload using a Streamlit web application.

---

## Key Features
- Handles imbalanced data using SMOTE
- Batch prediction using CSV upload
- Fraud probability calculation
- Final prediction as Fraud or Legit
- Downloadable prediction results
- Web application using Streamlit

---

## Machine Learning Details
- Model: Trained classification model (Logistic Regression / Random Forest)
- Imbalance Handling: SMOTE
- Scaling: StandardScaler (Amount & Time)
- Output: Fraud Probability and Prediction

---

## Project Structure
Credit-Card-Fraud-Detection-Using-SMOTE/
│
├── app.py
├── requirements.txt
├── README.md
├── audit_logs.csv
│
├── models/
│   ├── fraud_model.pkl
│   ├── scaler_amount.pkl
│   ├── scaler_time.pkl
│   └── feature_order.pkl

## Dataset Information
The dataset contains anonymized credit card transactions with the following columns:
- Time
- Amount
- V1 to V28 (PCA features)
- Class (0 = Legit, 1 = Fraud)

## Batch Prediction Output
After uploading a CSV file, the application generates:
- Fraud_Probability
- Prediction (Fraud / Legit)
The results can be viewed on the web interface and downloaded as a CSV file.

## How to Run the Project

### Step 1: Clone Repository
git clone https://github.com/SumitChoudhary003/Credit-Card-Fraud-Detection-Using-SMOTE.git
cd Credit-Card-Fraud-Detection-Using-SMOTE
### Step 2: Install Dependencies
pip install -r requirements.txt
### Step 3: Run Application
Streamlit run app.py

🌐 Live Demo
🚀 Live App: https://credit-card-fraud-detection-using-smote-r23jnpqwqdcawjxke2dyex.streamlit.app/

📌 Technologies Used
Python
Pandas, NumPy
Scikit-learn
SMOTE (imbalanced-learn)
Streamlit
Joblib

🎯 Use Case
Financial fraud detection systems
Banking & payment platforms
Academic and placement-ready ML project

👨‍💻 Author
Sumit Choudhary
B.Tech CSE | Machine Learning Enthusiast
🔗 GitHub: https://github.com/SumitChoudhary003 
