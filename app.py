import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load saved objects
# -------------------------------
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# -------------------------------
# App configuration
# -------------------------------
st.set_page_config(
    page_title="Regulation-Aware Credit Assessment",
    layout="centered"
)

st.title("💳 Regulation-Aware Credit Assessment System")
st.markdown(
    """
    This application demonstrates a **regulation-aware machine learning model**
    for credit risk assessment.  
    The model is designed to be **accurate, explainable, and ethically aligned**.
    """
)

st.warning(
    "⚠️ This application is for **academic demonstration purposes only** "
    "and does not represent a real-world lending system."
)

# -------------------------------
# User Inputs
# -------------------------------
st.header("📥 Input Credit Information")

limit_bal = st.number_input(
    "Credit Limit",
    min_value=0,
    value=50000,
    step=5000
)

avg_pay_status = st.slider(
    "Average Repayment Status (Lower is better)",
    min_value=-2.0,
    max_value=8.0,
    value=0.0,
    step=0.5
)

avg_bill_amt = st.number_input(
    "Average Bill Amount",
    min_value=0.0,
    value=6000.0,
    step=500.0
)

avg_pay_amt = st.number_input(
    "Average Payment Amount",
    min_value=0.0,
    value=5000.0,
    step=500.0
)

education = st.selectbox(
    "Education Level",
    ["Graduate", "University", "High School", "Other"]
)

marriage = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Other"]
)

# -------------------------------
# Encode categorical variables
# -------------------------------
input_data = {
    "LIMIT_BAL": limit_bal,
    "AVG_PAY_STATUS": avg_pay_status,
    "AVG_BILL_AMT": avg_bill_amt,
    "AVG_PAY_AMT": avg_pay_amt
}

# Education encoding
input_data["EDUCATION_2"] = 1 if education == "University" else 0
input_data["EDUCATION_3"] = 1 if education == "High School" else 0
input_data["EDUCATION_4"] = 1 if education == "Other" else 0

# Marriage encoding
input_data["MARRIAGE_1"] = 1 if marriage == "Married" else 0
input_data["MARRIAGE_2"] = 1 if marriage == "Single" else 0

# -------------------------------
# Create input DataFrame
# -------------------------------
input_df = pd.DataFrame([input_data])

# Align with training features
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Assess Credit Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.header("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Credit Risk (Likely Default)")
    else:
        st.success("✅ Low Credit Risk (Unlikely Default)")

    st.metric(
        label="Probability of Default",
        value=f"{probability:.2%}"
    )

    st.subheader("🧠 Explanation Summary")
    st.markdown(
        """
        - **Repayment behavior** has the strongest influence on the decision  
        - **Higher credit limits and payments** reduce default risk  
        - The model follows **logical and explainable rules** enforced during training
        """
    )

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(
    "Developed as part of an MSc project on Regulation-Aware Machine Learning for Credit Assessment."
)
