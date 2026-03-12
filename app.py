import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime


# ----------------------------
# Load trained model
# ----------------------------

rf = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ----------------------------
# INITIALIZE STATE
# ----------------------------

if 'timeline' not in st.session_state:
    st.session_state.timeline = []

# initialize widget state if missing
for i in range(1,29):
    if f"V{i}" not in st.session_state:
        st.session_state[f"V{i}"] = 0.0

if "scaled_amount" not in st.session_state:
    st.session_state["scaled_amount"] = 0.0

if "scaled_time" not in st.session_state:
    st.session_state["scaled_time"] = 0.0


# ----------------------------
# PATTERN GENERATOR
# ----------------------------

def update_pattern(is_fraud=False):

    if is_fraud:

        # Reset all V features
        for i in range(1,29):
            st.session_state[f"V{i}"] = np.random.normal(0,0.3)

        # Inject stronger fraud signatures on important PCA components
        st.session_state["V14"] = np.random.normal(-8, 1.5)
        st.session_state["V12"] = np.random.normal(-6, 1.2)
        st.session_state["V10"] = np.random.normal(-5, 1.2)
        st.session_state["V17"] = np.random.normal(-7, 1.5)
        st.session_state["V4"] = np.random.normal(3, 0.8)
        st.session_state["V11"] = np.random.normal(4, 0.8)

        st.session_state["scaled_amount"] = np.random.uniform(1.5,3.0)
        st.session_state["scaled_time"] = np.random.uniform(0.3,0.8)

    else:

        # Generate a realistic normal transaction pattern
        for i in range(1,29):
            st.session_state[f"V{i}"] = np.random.normal(0,0.5)

        st.session_state["scaled_amount"] = np.random.uniform(0.1,0.5)
        st.session_state["scaled_time"] = np.random.uniform(0.1,0.5)

    # --- Run prediction for timeline log ---
    cols = [f"V{i}" for i in range(1,29)] + ["scaled_amount","scaled_time"]

    vals = {c: st.session_state[c] for c in cols}

    df = pd.DataFrame([vals])

    prediction = rf.predict(df)[0]

    status = "High Risk 🚨" if prediction == 1 else "Normal"
    color = "red" if prediction == 1 else "green"

    st.session_state.timeline.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Amount": f"${st.session_state['scaled_amount']*100:.2f}",
        "Status": status,
        "Color": color
    })



# ----------------------------
# LAYOUT
# ----------------------------

col_left, col_main, col_right = st.columns([1,2,1.2])


# ----------------------------
# LEFT PANEL
# ----------------------------

with col_left:

    st.header("Quick Controls")

    st.button(
        "✅ Load Normal Pattern",
        on_click=update_pattern,
        args=(False,),
        use_container_width=True
    )

    st.button(
        "🚨 Load Fraud Pattern",
        on_click=update_pattern,
        args=(True,),
        use_container_width=True
    )

    st.caption("Transactions are logged automatically upon click.")

    st.divider()

    st.caption(
        """
        **Transparency Note**

        This model uses a dataset where the original transaction attributes were
        anonymized using **Principal Component Analysis (PCA)**.

        Features appear as **V1–V28**, which are mathematical components rather
        than real-world variables like merchant or location.  
        The model therefore detects fraud based on **patterns in this PCA feature space**.
        """
    )


# ----------------------------
# RIGHT PANEL (FEATURES)
# ----------------------------

with col_right:

    st.header("Feature PCA Panel")

    st.caption(
        "These PCA features are anonymized components of the original transaction data. "
        "Manual adjustments are for experimentation and may not represent realistic transactions."
    )

    # 👇 Collapsible advanced controls
    with st.expander("Advanced: Manual PCA Feature Controls"):

        st.number_input(
            "Scaled Amount",
            key="scaled_amount",
            step=0.1
        )

        st.number_input(
            "Scaled Time",
            key="scaled_time",
            step=0.1
        )

        st.divider()

        for i in range(1,29):

            st.number_input(
                f"V{i}",
                key=f"V{i}",
                step=0.1
            )

# ----------------------------
# CENTER PANEL
# ----------------------------

with col_main:

    st.title("💳 Credit Card Fraud Dashboard")

    cols = [f"V{i}" for i in range(1,29)] + ["scaled_amount","scaled_time"]

    vals = {c: st.session_state[c] for c in cols}

    input_df = pd.DataFrame([vals])

    prediction = rf.predict(input_df)[0]
    prob = rf.predict_proba(input_df)[0][1]

    st.subheader("Current Prediction")

    st.write(f"Fraud Probability: **{prob*100:.2f}%**")
    st.caption("Adjust PCA features on the right to see how the model reacts.")
    if prediction == 1:
        st.error("🚨 FRAUD DETECTED")
    else:
        st.success("✅ NORMAL TRANSACTION")

    # ----------------------------
    # TIMELINE
    # ----------------------------

    st.divider()
    st.subheader("🕒 Transaction Timeline")

    if not st.session_state.timeline:
        st.caption("No system logs yet.")

    for item in reversed(st.session_state.timeline):

        st.markdown(
            f"**{item['Time']}** ———— "
            f"<span style='color:{item['Color']}'>{item['Status']}</span> "
            f"(Approx. {item['Amount']})",
            unsafe_allow_html=True
        )



    
