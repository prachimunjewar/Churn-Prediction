import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/best_model_name.pkl', 'rb') as f:
        model_name = pickle.load(f)
    return model, encoders, scaler, model_name

model, encoders, scaler, model_name = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #1e3a5f, #2563eb);
    padding: 2rem; border-radius: 15px;
    color: white; text-align: center; margin-bottom: 2rem;
}
.metric-card {
    background: white; padding: 1rem; border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
}
.churn-high { color: #dc2626; font-size: 2rem; font-weight: bold; }
.churn-low  { color: #16a34a; font-size: 2rem; font-weight: bold; }
</style>
<div class="hero">
    <h1>📉 Customer Churn Prediction</h1>
    <p>ML-powered system to predict which customers are likely to churn</p>
    <p style="font-size:13px; opacity:0.8;">Model: LightGBM · SMOTE · SHAP Explainability</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict Churn", "📊 Model Performance", "🧠 SHAP Insights"])

# ── Tab 1: Prediction ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.markdown("**Services**")
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multi = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        st.markdown("**Billing**")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total = st.number_input("Total Charges ($)", 18.0, 8000.0, 1500.0)

    if st.button("🚀 Predict Churn", use_container_width=True):

        input_dict = {
            'gender': gender, 'SeniorCitizen': senior,
            'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone,
            'MultipleLines': multi, 'InternetService': internet,
            'OnlineSecurity': security, 'OnlineBackup': backup,
            'DeviceProtection': device, 'TechSupport': support,
            'StreamingTV': tv, 'StreamingMovies': movies,
            'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment, 'MonthlyCharges': monthly,
            'TotalCharges': total
        }

        input_df = pd.DataFrame([input_dict])

        # Encode categoricals
        cat_cols = input_df.select_dtypes(include='object').columns
        num_cols = input_df.select_dtypes(include=['int64', 'float64']).columns

        for col in cat_cols:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])

        input_df[num_cols] = scaler.transform(input_df[num_cols])

        prob = model.predict_proba(input_df)[0][1]
        pred = "WILL CHURN" if prob > 0.5 else "WILL NOT CHURN"

        st.divider()
        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("Prediction", pred)
        with r2:
            pct = f"{prob*100:.1f}%"
            color = "churn-high" if prob > 0.5 else "churn-low"
            st.markdown(f"**Churn Probability**")
            st.markdown(f'<p class="{color}">{pct}</p>', unsafe_allow_html=True)
        with r3:
            risk = "🔴 High Risk" if prob > 0.7 else "🟡 Medium Risk" if prob > 0.4 else "🟢 Low Risk"
            st.metric("Risk Level", risk)

        # Risk gauge
        fig, ax = plt.subplots(figsize=(5, 1))
        ax.barh(['Risk'], [prob], color='#dc2626' if prob > 0.5 else '#16a34a', height=0.4)
        ax.barh(['Risk'], [1-prob], left=[prob], color='#e5e7eb', height=0.4)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Churn Probability')
        ax.axvline(0.5, color='orange', linestyle='--', linewidth=1)
        st.pyplot(fig)
        plt.close()

# ── Tab 2: Model Performance ──────────────────────────────────────────────────
with tab2:
    st.subheader("Model Performance")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Model", model_name)
    m2.metric("ROC-AUC", "0.7443")
    m3.metric("Accuracy", "68.28%")
    m4.metric("Training Samples", "7,043")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists('plots/roc_curve.png'):
            st.image('plots/roc_curve.png', caption="ROC Curve - All Models", use_container_width=True)
    with col2:
        if os.path.exists('plots/confusion_matrix.png'):
            st.image('plots/confusion_matrix.png', caption="Confusion Matrix", use_container_width=True)

    if os.path.exists('plots/feature_importance.png'):
        st.image('plots/feature_importance.png', caption="Feature Importance", use_container_width=True)

    st.markdown("""
    **Why these models?**
    - **Logistic Regression** — interpretable baseline
    - **Random Forest** — handles non-linearity, no scaling needed
    - **XGBoost** — gradient boosting, industry standard for tabular data
    - **LightGBM** — fastest, best on categorical features, won this comparison
    - **SMOTE** applied to handle class imbalance (~31% churn vs 69% no-churn)
    """)

# ── Tab 3: SHAP ───────────────────────────────────────────────────────────────
with tab3:
    st.subheader("SHAP Feature Importance")
    st.markdown("SHAP (SHapley Additive exPlanations) shows **which features drive churn predictions** and by how much.")

    if os.path.exists('plots/shap_summary.png'):
        st.image('plots/shap_summary.png', caption="SHAP Summary Plot", use_container_width=True)

    st.markdown("""
    **How to read this:**
    - Features at the top have the **highest impact** on predictions
    - Bar length = average impact magnitude across all predictions
    - Longer bar = more influential feature in the model's decisions
    
    **Key business insight:** Contract type, tenure, and monthly charges 
    are typically the strongest churn predictors — customers on month-to-month 
    contracts with short tenure and high charges are highest risk.
    """)

st.divider()
st.markdown("<p style='text-align:center; color:gray;'>Built with LightGBM · SMOTE · SHAP · Streamlit</p>", unsafe_allow_html=True)
