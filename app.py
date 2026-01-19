import streamlit as st

import numpy as np
import joblib

model=joblib.load("loan_approval_model.pkl")
scaler=joblib.load("scaler.pkl")

st.title("LOAN APPROVAL PREDICTION")

no_of_dependents=st.number_input("number of dependents",min_value=0,max_value=10)
income_annum=st.number_input("annual income",min_value=0,)
loan_amount=st.number_input("loan amount",min_value=0)
loan_term=st.number_input("loan term(in year)",min_value=1)
cibil_score=st.number_input("CIBIL score",min_value=300,max_value=900)
residential_assets_value = st.number_input("Residential Asset Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Asset Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Asset Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)


education=st.selectbox("education",["Graduated","not Graduated"])
self_employed=st.selectbox("self employed",["yes","no"])

education_val= 1 if education == "not Graduated" else 0
self_employed_val= 1 if self_employed == "yes" else 0

input_data = np.array([[
    no_of_dependents,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
    education_val,
    self_employed_val
]])


input_scaled=scaler.transform(input_data)

if st.button("Predict Loan Status"):
    probability = model.predict_proba(input_scaled)[0][1]

    if probability >= 0.6:  # stricter approval
        st.success(f"✅ Loan Approved (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ Loan Rejected (Confidence: {probability:.2f})")