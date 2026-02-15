import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ------------------------------------------------------
# Load saved model + columns
# ------------------------------------------------------
model = pickle.load(open("decision_tree_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

# Load encoders if used
try:
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
except:
    label_encoders = None


# ------------------------------------------------------
# Prediction Function
# ------------------------------------------------------
def predict_loan_status(input_data: dict):

    df = pd.DataFrame([input_data])

    # Apply label encoders
    if label_encoders is not None:
        for col, encoder in label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])

    # Add missing columns
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Arrange column order
    df = df[model_columns]

    pred = model.predict(df)[0]
    return "Approved" if pred == 1 else "Rejected"


# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------
st.set_page_config(page_title="Loan Approval Predictor", layout="wide")
st.title("🏦 Loan Approval Prediction App")
st.write("Fill the details below to predict loan approval.")

col1, col2, col3 = st.columns(3)

with col1:
    Gender = st.selectbox("Gender", [0, 1])
    Married = st.selectbox("Married", [0, 1])
    Dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    Education = st.selectbox("Education (0 = Grad, 1 = Not Grad)", [0, 1])
    Self_Employed = st.selectbox("Self Employed", [0, 1])

with col2:
    ApplicantIncome = st.number_input("Applicant Income (scaled)", value=0.0, format="%.4f")
    CoapplicantIncome = st.number_input("Coapplicant Income (scaled)", value=0.0, format="%.4f")
    LoanAmount = st.number_input("Loan Amount (scaled)", value=0.0, format="%.4f")
    Loan_Amount_Term = st.selectbox("Loan Amount Term", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480])
    Credit_History = st.selectbox("Credit History", [0.0, 1.0])

with col3:
    Property_Area_Rural = st.selectbox("Rural (1 = Yes, 0 = No)", [0, 1])
    Property_Area_Semiurban = st.selectbox("Semiurban (1 = Yes, 0 = No)", [0, 1])
    Property_Area_Urban = st.selectbox("Urban (1 = Yes, 0 = No)", [0, 1])
    TotalIncome = st.number_input("Total Income", value=3000)
    EMI = st.number_input("Monthly EMI", value=100)
    BalanceIncome = st.number_input("Balance Income", value=2000)

# Organize input
input_data = {
    "Gender": Gender,
    "Married": Married,
    "Dependents": Dependents,
    "Education": Education,
    "Self_Employed": Self_Employed,
    "ApplicantIncome": ApplicantIncome,
    "CoapplicantIncome": CoapplicantIncome,
    "LoanAmount": LoanAmount,
    "Loan_Amount_Term": Loan_Amount_Term,
    "Credit_History": Credit_History,
    "Property_Area_Rural": Property_Area_Rural,
    "Property_Area_Semiurban": Property_Area_Semiurban,
    "Property_Area_Urban": Property_Area_Urban,
    "TotalIncome": TotalIncome,
    "EMI": EMI,
    "BalanceIncome": BalanceIncome,
}

if st.button("Predict Loan Status"):
    result = predict_loan_status(input_data)
    if result == "Approved":
        st.success("🎉 Loan Status: APPROVED")
    else:
        st.error("❌ Loan Status: REJECTED")
