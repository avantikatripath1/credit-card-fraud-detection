import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
rf = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("💳 Credit Card Fraud Detection Dashboard")

st.markdown("Machine Learning prototype using **Random Forest**")

# ------------------------
# Transaction Input
# ------------------------



st.sidebar.header("Transaction Input")

# Fraud demo example values
fraud_example = {
    "V1": -2.5, "V2": 2.3, "V3": -1.8, "V4": 3.2,
    "V5": -1.5, "V6": -0.8, "V7": -2.1, "V8": 1.7,
    "V9": -2.3, "V10": -3.1, "V11": 2.4, "V12": -2.7,
    "V13": 0, "V14": -4.2, "V15": 0, "V16": -1.9,
    "V17": -2.8, "V18": -1.3, "V19": 0, "V20": 0,
    "V21": 1.5, "V22": 0, "V23": -0.5, "V24": 0,
    "V25": 0, "V26": 0, "V27": 0.3, "V28": 0.1,
    "scaled_amount": 0,
    "scaled_time": 0
}

if st.sidebar.button("🚨 Load Fraud Example"):
    st.session_state.update(fraud_example)

features = {}

for i in range(1, 29):
    features[f"V{i}"] = st.sidebar.number_input(
        f"V{i}",
        value=st.session_state.get(f"V{i}", 0.0)
    )

features["scaled_amount"] = st.sidebar.number_input(
    "Scaled Amount",
    value=st.session_state.get("scaled_amount", 0.0)
)

features["scaled_time"] = st.sidebar.number_input(
    "Scaled Time",
    value=st.session_state.get("scaled_time", 0.0)
)

# ------------------------
# Prediction
# ------------------------

st.subheader("Prediction")

# Convert sidebar inputs into dataframe
input_df = pd.DataFrame([features])

if st.button("Predict Transaction"):

    prediction = rf.predict(input_df)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Normal Transaction")

# ------------------------
# Feature Importance
# ------------------------

st.subheader("Feature Importance")

importances = rf.feature_importances_
feature_names = input_df.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots()

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature",
    ax=ax
)

st.pyplot(fig)

# ------------------------
# About Section
# ------------------------

st.subheader("Model Information")

st.write("""
Model Used: **Random Forest Classifier**

Performance:
- Precision: 0.88
- Recall: 0.85
- F1 Score: 0.86
- ROC AUC: 0.98
""")