import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import os
import joblib

# Load scalers
scaler_amount = joblib.load("models/scaler_amount.pkl")
scaler_time = joblib.load("models/scaler_time.pkl")
feature_order = joblib.load("models/feature_order.pkl")


# ===============================
# Config
# ===============================
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

LOG_FILE = "audit_logs.csv"

# ===============================
# Utils
# ===============================
def save_audit_log(data):
    if os.path.exists(LOG_FILE):
        old = pd.read_csv(LOG_FILE)
        data = pd.concat([old, data], ignore_index=True)
    data.to_csv(LOG_FILE, index=False)

# ===============================
# Load Models
# ===============================
@st.cache_resource
def load_models():
    rf = joblib.load("models/random_forest_model.pkl")
    xgb = joblib.load("models/xgboost_model.pkl")
    return rf, xgb

rf_model, xgb_model = load_models()

# ===============================
# Sidebar
# ===============================
st.sidebar.title("⚙ Settings")

model_option = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"])

threshold = st.sidebar.slider(
    "Fraud Decision Threshold", 0.1, 0.9, 0.5, 0.05
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Performance")

if model_option == "Random Forest":
    st.sidebar.write("ROC-AUC: 0.968")
    st.sidebar.write("Fraud Recall: 82%")
    model = rf_model
else:
    st.sidebar.write("ROC-AUC: 0.975")
    st.sidebar.write("Fraud Recall: 87%")
    model = xgb_model

# ===============================
# Main UI
# ===============================
st.title("💳 Credit Card Fraud Detection System")
st.markdown("---")

# ===============================
# Feature Input
# ===============================
st.subheader("🔢 Enter Transaction Features")

important_features = {
    10: "V10 (High impact)",
    12: "V12",
    14: "V14 (Very High impact)",
    17: "V17"
}

input_data = [0.0] * 30
c1, c2 = st.columns(2)

for i, (idx, label) in enumerate(important_features.items()):
    with c1 if i % 2 == 0 else c2:
        input_data[idx - 1] = st.slider(
            label, -10.0, 10.0, 0.0, 0.1
        )

input_array = np.array(input_data).reshape(1, -1)

# ===============================
# Prediction Button
# ===============================
if st.button("🔍 Predict Transaction", use_container_width=True):

    prob = model.predict_proba(input_array)[0][1]
    prediction = int(prob >= threshold)

    # Save to session (IMPORTANT)
    st.session_state["prob"] = prob
    st.session_state["prediction"] = prediction

    # Risk logic
    if prediction == 1:
        risk_level = "HIGH"
    elif prob < threshold:
        risk_level = "LOW"
    else:
        risk_level = "MEDIUM"

    st.session_state["risk_level"] = risk_level

# ===============================
# Show Result (persist)
# ===============================
if "prob" in st.session_state:

    prob = st.session_state["prob"]
    prediction = st.session_state["prediction"]
    risk_level = st.session_state["risk_level"]

    st.markdown("### 🧠 Prediction Result")
    st.metric("Fraud Probability", f"{prob*100:.2f}%")

    if prediction == 1:
        st.error("🔴 Risk Level: HIGH")
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success(f"🟢 Risk Level: {risk_level}")
        st.success("✅ Legitimate Transaction")

    # ===============================
    # Save Audit Log
    # ===============================
    if st.checkbox("📝 Save Prediction Log", key="save_log"):
        log = pd.DataFrame([{
            "timestamp": datetime.now(),
            "model": model_option,
            "fraud_probability": round(prob, 4),
            "threshold": threshold,
            "risk_level": risk_level,
            "prediction": "Fraud" if prediction == 1 else "Legitimate"
        }])

        save_audit_log(log)
        st.success("Audit log saved successfully")

# ===============================
# Audit Dashboard
# ===============================
st.markdown("---")
st.subheader("📊 Audit Log Dashboard")

if os.path.exists(LOG_FILE):
    logs = pd.read_csv(LOG_FILE)
    logs = logs.reset_index(drop=True)
    logs = logs.astype(str)
    st.dataframe(logs, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Predictions", len(logs))
    with col2:
        fraud_rate= (logs["prediction"] == "Fraud").mean()*100
        st.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")
    st.download_button(
        "Download Audit Logs",
        logs.to_csv(index=False),
        file_name= "audit_logs.csv",
        mime= "text/csv"
    )        
else:
    st.info("No audit logs found yet.")


        # ===============================
# BATCH PREDICTION (FINAL)
# ===============================
st.subheader("📦 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader(
    "Upload CSV file with credit card transactions",
    type="csv",
    key = "batch_csv_uploader"
)

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Original CSV shape:", batch_data.shape)

        # ---------- preprocessing ----------
        if "Amount" in batch_data.columns and "Time" in batch_data.columns:

            batch_data["scaled_amount"] = scaler_amount.transform(
                batch_data[["Amount"]]
            )

            batch_data["scaled_time"] = scaler_time.transform(
                batch_data[["Time"]]
            )

            drop_cols = ["Amount", "Time"]
        else:
            drop_cols = []

        # drop Class if present
        if "Class" in batch_data.columns:
            drop_cols.append("Class")

        batch_data = batch_data.drop(columns=drop_cols)

        # keep same feature order as training
        batch_data = batch_data[feature_order]

        # ---------- prediction ----------
        probs = model.predict_proba(batch_data)[:, 1]
        preds = (probs >= threshold).astype(int)

        batch_data["Fraud_Probability"] = probs.round(4)
        batch_data["Prediction"] = batch_data["Fraud_Probability"].apply(
            lambda x: "Fraud" if x >= threshold else "Legit"
        )

        st.success("✅ Batch prediction successful")
        st.dataframe(batch_data.head(20), use_container_width=True)

        st.download_button(
            "⬇ Download Results",
            data=batch_data.to_csv(index=False),
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error during batch prediction: {e}")