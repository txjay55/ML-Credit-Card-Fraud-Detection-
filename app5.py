import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------
model = joblib.load("fraud_model.pkl")
fraud_sample = joblib.load("fraud_sample.pkl")
normal_sample = joblib.load("normal_sample.pkl")

# -------------------- BRANDING --------------------
st.markdown("""
# 💳 Credit Card Fraud Detection System  
### 🔎 AI-Powered Risk Monitoring Dashboard
""")

st.markdown("---")

# -------------------- FEATURE NAMES --------------------
feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# -------------------- SESSION STATE --------------------
if "inputs" not in st.session_state:
    st.session_state.inputs = [0.0] * 30

# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs([
    "🧾 Transaction Analysis",
    "📊 Risk Dashboard",
    "🧠 Model Insights"
])

# =========================================================
# 🧾 TAB 1 – INPUT PAGE
# =========================================================
with tab1:

    st.subheader("Enter Transaction Features")

    col_demo1, col_demo2 = st.columns(2)

    with col_demo1:
        if st.button("🚨 Load Fraud Case", use_container_width=True):
            st.session_state.inputs = fraud_sample.tolist()

    with col_demo2:
        if st.button("✅ Load Normal Case", use_container_width=True):
            st.session_state.inputs = normal_sample.tolist()

    st.markdown("---")

    col_left, col_right = st.columns(2)
    inputs = []

    with col_left:
        for i in range(0, 15):
            val = st.number_input(feature_names[i], value=st.session_state.inputs[i])
            inputs.append(val)

    with col_right:
        for i in range(15, 30):
            val = st.number_input(feature_names[i], value=st.session_state.inputs[i])
            inputs.append(val)

    st.markdown("---")

    if st.button("🚀 Analyze Transaction", use_container_width=True):

        features = np.array([inputs])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        st.session_state.prediction = prediction
        st.session_state.probability = probability

        st.success("✅ Analysis Complete → Check Risk Dashboard Tab")

# =========================================================
# 📊 TAB 2 – DASHBOARD
# =========================================================
with tab2:

    st.subheader("Fraud Risk Dashboard")

    if "probability" not in st.session_state:
        st.info("Run a transaction analysis first.")
    else:
        prediction = st.session_state.prediction
        probability = st.session_state.probability

        # ---------------- KPI CARDS ----------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Fraud Probability", f"{probability:.2%}")

        with col2:
            st.metric("Prediction", "🚨 Fraud" if prediction == 1 else "✅ Normal")

        with col3:
            risk_level = (
                "High Risk" if probability > 0.7 else
                "Moderate Risk" if probability > 0.3 else
                "Low Risk"
            )
            st.metric("Risk Level", risk_level)

        st.markdown("---")

        # ---------------- ALERT ----------------
        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Normal Transaction")

        # ---------------- GAUGE ----------------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Fraud Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"},
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # ---------------- INTERPRETATION ----------------
        if probability > 0.7:
            st.warning("⚠️ High Fraud Risk – Immediate Action Required")
        elif probability > 0.3:
            st.info("ℹ️ Moderate Risk – Monitor Transaction")
        else:
            st.success("✔ Low Risk – Transaction Appears Safe")

# =========================================================
# 🧠 TAB 3 – MODEL INFO
# =========================================================
with tab3:

    st.subheader("Model Insights")

    st.markdown("""
**Model Used:** Random Forest Classifier  

**Dataset:** Kaggle Credit Card Fraud Detection  

**Challenges:**  
- Highly imbalanced data  
- Fraud < 1%  

**Techniques Applied:**  
✔ Feature Scaling  
✔ Logistic Regression (Baseline)  
✔ Random Forest  
✔ Threshold Tuning  
✔ ROC-AUC Evaluation  

**Key Metrics Achieved:**  
- Fraud Precision ≈ 0.96  
- Fraud Recall ≈ 0.80+  
- ROC-AUC ≈ 0.91  
""")

    st.success("🚀 Streamlit Dashboard Deployment")
