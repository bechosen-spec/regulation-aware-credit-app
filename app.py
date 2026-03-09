# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # -------------------------------
# # Load saved objects
# # -------------------------------
# model = joblib.load("credit_model.pkl")
# scaler = joblib.load("scaler.pkl")
# feature_names = joblib.load("feature_names.pkl")

# # -------------------------------
# # App configuration
# # -------------------------------
# st.set_page_config(
#     page_title="Regulation-Aware Credit Assessment",
#     layout="centered"
# )

# st.title("💳 Regulation-Aware Credit Assessment System")
# st.markdown(
#     """
#     This application demonstrates a **regulation-aware machine learning model**
#     for credit risk assessment.  
#     The model is designed to be **accurate, explainable, and ethically aligned**.
#     """
# )

# st.warning(
#     "⚠️ This application is for **academic demonstration purposes only** "
#     "and does not represent a real-world lending system."
# )

# # -------------------------------
# # User Inputs
# # -------------------------------
# st.header("📥 Input Credit Information")

# limit_bal = st.number_input(
#     "Credit Limit",
#     min_value=0,
#     value=50000,
#     step=5000
# )

# avg_pay_status = st.slider(
#     "Average Repayment Status (Lower is better)",
#     min_value=-2.0,
#     max_value=8.0,
#     value=0.0,
#     step=0.5
# )

# avg_bill_amt = st.number_input(
#     "Average Bill Amount",
#     min_value=0.0,
#     value=6000.0,
#     step=500.0
# )

# avg_pay_amt = st.number_input(
#     "Average Payment Amount",
#     min_value=0.0,
#     value=5000.0,
#     step=500.0
# )

# education = st.selectbox(
#     "Education Level",
#     ["Graduate", "University", "High School", "Other"]
# )

# marriage = st.selectbox(
#     "Marital Status",
#     ["Married", "Single", "Other"]
# )

# # -------------------------------
# # Encode categorical variables
# # -------------------------------
# input_data = {
#     "LIMIT_BAL": limit_bal,
#     "AVG_PAY_STATUS": avg_pay_status,
#     "AVG_BILL_AMT": avg_bill_amt,
#     "AVG_PAY_AMT": avg_pay_amt
# }

# # Education encoding
# input_data["EDUCATION_2"] = 1 if education == "University" else 0
# input_data["EDUCATION_3"] = 1 if education == "High School" else 0
# input_data["EDUCATION_4"] = 1 if education == "Other" else 0

# # Marriage encoding
# input_data["MARRIAGE_1"] = 1 if marriage == "Married" else 0
# input_data["MARRIAGE_2"] = 1 if marriage == "Single" else 0

# # -------------------------------
# # Create input DataFrame
# # -------------------------------
# input_df = pd.DataFrame([input_data])

# # Align with training features
# input_df = input_df.reindex(columns=feature_names, fill_value=0)

# # -------------------------------
# # Prediction
# # -------------------------------
# if st.button("🔍 Assess Credit Risk"):
#     prediction = model.predict(input_df)[0]
#     probability = model.predict_proba(input_df)[0][1]

#     st.header("📊 Prediction Result")

#     if prediction == 1:
#         st.error("⚠️ High Credit Risk (Likely Default)")
#     else:
#         st.success("✅ Low Credit Risk (Unlikely Default)")

#     st.metric(
#         label="Probability of Default",
#         value=f"{probability:.2%}"
#     )

#     st.subheader("🧠 Explanation Summary")
#     st.markdown(
#         """
#         - **Repayment behavior** has the strongest influence on the decision  
#         - **Higher credit limits and payments** reduce default risk  
#         - The model follows **logical and explainable rules** enforced during training
#         """
#     )

# # -------------------------------
# # Footer
# # -------------------------------
# st.markdown("---")
# st.caption(
#     "Developed as part of an MSc project on Regulation-Aware Machine Learning for Credit Assessment."
# )



import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import datetime
from pathlib import Path

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Credit Risk Assessment System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM STYLING (LIGHT + DARK MODE FRIENDLY)
# -----------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 1.2rem;
    }

    .hero-card {
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(120, 120, 120, 0.18);
        background: linear-gradient(
            135deg,
            rgba(99, 102, 241, 0.16),
            rgba(59, 130, 246, 0.12)
        );
        backdrop-filter: blur(8px);
        margin-bottom: 1.25rem;
    }

    .info-card {
        padding: 1.15rem 1.2rem;
        border-radius: 18px;
        border: 1px solid rgba(120, 120, 120, 0.16);
        background-color: rgba(120, 120, 120, 0.05);
        margin-bottom: 1rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.04);
    }

    .result-card {
        padding: 1.25rem;
        border-radius: 18px;
        border: 1px solid rgba(120, 120, 120, 0.16);
        background-color: rgba(120, 120, 120, 0.05);
        margin-top: 0.75rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.04);
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .soft-text {
        opacity: 0.88;
        font-size: 0.96rem;
        line-height: 1.65;
    }

    .footer-note {
        text-align: center;
        opacity: 0.72;
        font-size: 0.9rem;
        padding-top: 0.6rem;
    }

    div[data-testid="stMetric"] {
        border-radius: 16px;
        padding: 0.8rem;
        border: 1px solid rgba(120, 120, 120, 0.15);
        background-color: rgba(120, 120, 120, 0.05);
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }

    div[data-testid="stSidebar"] {
        border-right: 1px solid rgba(120, 120, 120, 0.12);
    }

    .mini-note {
        font-size: 0.88rem;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# DATABASE CONNECTION
# -----------------------------
conn = sqlite3.connect("credit_risk.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    credit_limit REAL,
    avg_pay_status REAL,
    avg_bill_amt REAL,
    avg_pay_amt REAL,
    education TEXT,
    marriage TEXT,
    risk TEXT,
    probability REAL,
    created_at TEXT
)
""")
conn.commit()

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("credit_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# -----------------------------
# HELPERS
# -----------------------------
def save_assessment(
    credit_limit,
    avg_pay_status,
    avg_bill_amt,
    avg_pay_amt,
    education,
    marriage,
    risk,
    probability
):
    cursor.execute("""
    INSERT INTO assessments (
        credit_limit,
        avg_pay_status,
        avg_bill_amt,
        avg_pay_amt,
        education,
        marriage,
        risk,
        probability,
        created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        credit_limit,
        avg_pay_status,
        avg_bill_amt,
        avg_pay_amt,
        education,
        marriage,
        risk,
        probability,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()


def prepare_input(credit_limit, avg_pay_status, avg_bill_amt, avg_pay_amt, education, marriage):
    input_data = {
        "LIMIT_BAL": credit_limit,
        "AVG_PAY_STATUS": avg_pay_status,
        "AVG_BILL_AMT": avg_bill_amt,
        "AVG_PAY_AMT": avg_pay_amt
    }

    input_data["EDUCATION_2"] = 1 if education == "University" else 0
    input_data["EDUCATION_3"] = 1 if education == "High School" else 0
    input_data["EDUCATION_4"] = 1 if education == "Other" else 0

    input_data["MARRIAGE_1"] = 1 if marriage == "Married" else 0
    input_data["MARRIAGE_2"] = 1 if marriage == "Single" else 0

    df = pd.DataFrame([input_data])
    df = df.reindex(columns=feature_names, fill_value=0)
    return df


def clean_text_value(x):
    if pd.isna(x):
        return x
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        return x.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return x


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(clean_text_value)
    return df


def load_records():
    records = pd.read_sql_query(
        "SELECT * FROM assessments ORDER BY created_at DESC",
        conn
    )
    return clean_dataframe(records)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("## 💳 Navigation")
    menu = st.radio(
        "Go to",
        [
            "Home",
            "Credit Risk Assessment",
            "View Assessment Records",
            "System Information"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Quick Stats")

    total_records = pd.read_sql_query(
        "SELECT COUNT(*) AS total FROM assessments", conn
    )["total"].iloc[0]

    st.metric("Stored Assessments", int(total_records))
    st.metric("Models Used", 5)
    st.metric("Dataset Records", "30,000")

    st.markdown("---")
    st.markdown(
        '<div class="mini-note">Hybrid regulation-aware credit assessment dashboard.</div>',
        unsafe_allow_html=True
    )

# -----------------------------
# HOME PAGE
# -----------------------------
if menu == "Home":
    st.markdown("""
    <div class="hero-card">
        <h1 style="margin-bottom:0.35rem;">Regulation-Aware Credit Risk Assessment System</h1>
        <p style="font-size:1rem; margin-bottom:0;">
            A predictive platform for evaluating credit default probability using machine learning,
            structured financial indicators, and compliance-oriented design principles.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Models Implemented", "5")
    col2.metric("Dataset Size", "30,000")
    col3.metric("Prediction Output", "Binary Risk")

    st.markdown("")

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("""
        <div class="info-card">
            <div class="section-title">System Overview</div>
            <div class="soft-text">
                This platform analyzes key financial indicators such as credit limit,
                repayment behavior, billing pattern, and payment history to estimate
                the likelihood of credit default.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <div class="section-title">Core Capabilities</div>
            <div class="soft-text">
                • Credit risk prediction<br>
                • Assessment record storage<br>
                • Hybrid regulation-aware modeling<br>
                • Structured decision support interface
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <div class="section-title">System Value</div>
            <div class="soft-text">
                The application supports risk evaluation, preserves assessment history,
                and provides a user-friendly environment for decision support in regulated
                credit assessment contexts.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        image_path = Path("image.png")
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.info("Add an image named 'image.png' to your project folder to display the homepage illustration.")

# -----------------------------
# CREDIT RISK ASSESSMENT
# -----------------------------
elif menu == "Credit Risk Assessment":
    st.title("Credit Risk Assessment")
    st.caption("Provide customer financial information to estimate default probability.")

    st.markdown("""
    <div class="info-card">
        <div class="section-title">Assessment Form</div>
        <div class="soft-text">
            Fill in the relevant credit indicators below. The system will process the
            values, generate a probability score, classify the risk level, and store
            the result in the database.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("assessment_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            credit_limit = st.number_input(
                "Credit Limit",
                min_value=0,
                max_value=1000000,
                value=50000,
                step=5000,
                help="Total credit amount assigned to the customer."
            )

            avg_pay_status = st.slider(
                "Average Repayment Status",
                min_value=-2.0,
                max_value=8.0,
                value=0.0,
                step=0.5,
                help="Lower values indicate better repayment behavior."
            )

            avg_bill_amt = st.number_input(
                "Average Bill Amount",
                min_value=0.0,
                max_value=1000000.0,
                value=6000.0,
                step=500.0,
                help="Average monthly billed amount."
            )

        with col2:
            avg_pay_amt = st.number_input(
                "Average Payment Amount",
                min_value=0.0,
                max_value=1000000.0,
                value=5000.0,
                step=500.0,
                help="Average amount paid by the customer."
            )

            education = st.selectbox(
                "Education Level",
                ["Graduate", "University", "High School", "Other"],
                help="Educational background category."
            )

            marriage = st.selectbox(
                "Marital Status",
                ["Married", "Single", "Other"],
                help="Marital status category."
            )

        submitted = st.form_submit_button("Run Credit Assessment", use_container_width=True)

    if submitted:
        df = prepare_input(
            credit_limit,
            avg_pay_status,
            avg_bill_amt,
            avg_pay_amt,
            education,
            marriage
        )

        prediction = model.predict(df)[0]
        probability = float(model.predict_proba(df)[0][1])

        risk = "High Risk" if prediction == 1 else "Low Risk"
        decision_text = "Likely Default" if prediction == 1 else "Unlikely Default"

        save_assessment(
            credit_limit,
            avg_pay_status,
            avg_bill_amt,
            avg_pay_amt,
            education,
            marriage,
            risk,
            probability
        )

        st.markdown("### Assessment Result")

        res1, res2, res3 = st.columns(3)
        res1.metric("Risk Category", risk)
        res2.metric("Default Probability", f"{probability:.2%}")
        res3.metric("Prediction", decision_text)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        if prediction == 1:
            st.error("The customer is classified as high risk based on the provided financial indicators.")
        else:
            st.success("The customer is classified as low risk based on the provided financial indicators.")

        if probability >= 0.70:
            st.warning("This result suggests a strong probability of default and may require closer financial review.")
        elif probability >= 0.40:
            st.info("This result suggests moderate risk and may require additional evaluation.")
        else:
            st.info("This result suggests relatively stable repayment behavior under the current input values.")

        with st.expander("Interpretation Summary", expanded=True):
            st.write("""
            The system bases its decision primarily on repayment behavior, credit utilization,
            and payment history. Higher repayment delays generally increase default risk,
            while stronger payment behavior and healthier credit patterns reduce predicted risk.
            """)

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# VIEW RECORDS
# -----------------------------
elif menu == "View Assessment Records":
    st.title("Assessment Records")
    st.caption("Review, filter, and export stored assessment history.")

    records = load_records()

    if records.empty:
        st.warning("No assessment records found.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            risk_filter = st.selectbox("Filter by Risk", ["All", "High Risk", "Low Risk"])

        with col2:
            search_term = st.text_input("Search by Education or Marital Status")

        filtered_records = records.copy()

        if risk_filter != "All":
            filtered_records = filtered_records[filtered_records["risk"] == risk_filter]

        if search_term:
            filtered_records = filtered_records[
                filtered_records["education"].astype(str).str.contains(search_term, case=False, na=False) |
                filtered_records["marriage"].astype(str).str.contains(search_term, case=False, na=False)
            ]

        filtered_records = clean_dataframe(filtered_records)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records", len(filtered_records))
        m2.metric("High Risk Cases", int((filtered_records["risk"] == "High Risk").sum()))
        m3.metric("Low Risk Cases", int((filtered_records["risk"] == "Low Risk").sum()))

        st.dataframe(
            filtered_records,
            use_container_width=True,
            height=430,
            hide_index=True
        )

        csv = filtered_records.to_csv(index=False).encode("utf-8", errors="ignore")
        st.download_button(
            "Download Records as CSV",
            data=csv,
            file_name="credit_assessment_records.csv",
            mime="text/csv",
            use_container_width=True
        )

# -----------------------------
# SYSTEM INFORMATION
# -----------------------------
elif menu == "System Information":
    st.title("System Information")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="info-card">
            <div class="section-title">System Summary</div>
            <div class="soft-text">
                <b>System Name:</b> Regulation-Aware Credit Risk Assessment System<br><br>
                <b>Purpose:</b> Predict credit default probability using a machine learning
                framework that supports structured financial decision-making.<br><br>
                <b>Dataset:</b> UCI Credit Card Default Dataset<br>
                <b>Dataset Size:</b> 30,000 records
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <div class="section-title">Technologies Used</div>
            <div class="soft-text">
                • Python<br>
                • Pandas<br>
                • Scikit-learn<br>
                • XGBoost<br>
                • Streamlit<br>
                • SQLite<br>
                • Joblib
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="section-title">Implemented Models</div>
        <div class="soft-text">
            1. Logistic Regression<br>
            2. Neural Network<br>
            3. Baseline XGBoost<br>
            4. Fairness-Constrained Model<br>
            5. Hybrid Regulation-Aware Model
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    '<div class="footer-note">PhD Research System – Credit Risk Assessment</div>',
    unsafe_allow_html=True
)