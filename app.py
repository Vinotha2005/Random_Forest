# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np  # ✅ Added import

# -----------------------------
# Load model and components
# -----------------------------
model = pickle.load(open("loan_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pt = pickle.load(open("power_transformer.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🏦 Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction App")
st.write("This app predicts loan approval likelihood using a trained Random Forest model.")

# -----------------------------
# Input Fields
# -----------------------------
col1, col2 = st.columns(2)

person_age = col1.number_input("Applicant Age", 18, 100, 30)
person_income = col2.number_input("Annual Income ($)", 1000, 200000, 50000)
person_emp_exp = col1.number_input("Years of Employment", 0, 40, 5)
loan_amnt = col2.number_input("Loan Amount ($)", 500, 50000, 10000)
loan_int_rate = col1.number_input("Interest Rate (%)", 1.0, 30.0, 12.5)
loan_percent_income = col2.number_input("Loan % of Income", 0.0, 1.0, 0.2)
cb_person_cred_hist_length = col1.number_input("Credit History (Years)", 1, 50, 5)
credit_score = col2.number_input("Credit Score", 300, 850, 700)

person_gender = col1.selectbox("Gender", ["male", "female"])
person_education = col2.selectbox("Education Level", ["high_school", "bachelor", "master", "doctorate", "other"])
person_home_ownership = col1.selectbox("Home Ownership", ["rent", "own", "mortgage", "other"])
loan_intent = col2.selectbox("Loan Intent", ["personal", "education", "medical", "venture", "homeimprovement", "debtconsolidation"])
previous_loan_defaults_on_file = col1.selectbox("Previous Loan Defaults", ["y", "n"])

# -----------------------------
# Prepare input DataFrame
# -----------------------------
input_data = {
    "person_age": person_age,
    "person_income": person_income,
    "person_emp_exp": person_emp_exp,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "credit_score": credit_score,
    "person_gender": person_gender,
    "person_education": person_education,
    "person_home_ownership": person_home_ownership,
    "loan_intent": loan_intent,
    "previous_loan_defaults_on_file": previous_loan_defaults_on_file
}

df_input = pd.DataFrame([input_data])

# -----------------------------
# Encode categoricals safely
# -----------------------------
for col, le in encoders.items():
    if col in df_input.columns:
        df_input[col] = df_input[col].astype(str).str.lower()
        unseen = set(df_input[col]) - set(le.classes_)
        if unseen:
            # Extend known classes for unseen label
            le.classes_ = np.append(le.classes_, list(unseen))
        df_input[col] = le.transform(df_input[col])

# -----------------------------
# Apply transformations
# -----------------------------
numeric_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']

df_input[numeric_cols] = pt.transform(df_input[numeric_cols])
df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

# -----------------------------
# Ensure column order matches training
# -----------------------------
df_input = df_input.reindex(columns=feature_columns, fill_value=0)
proba = model.predict_proba(df_input)[0][1]  # probability of approval (1)
threshold = 0.4  # you can tune this, default 0.5
if proba >= threshold:
    result = f"✅ Loan Approved (Probability: {proba:.2f})"
else:
    result = f"❌ Loan Rejected (Probability: {proba:.2f})"
st.success(result)

# Predict
# -----------------------------
if st.button("Predict Loan Approval 💡"):
    prediction = model.predict(df_input)[0]
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
    st.success(result)
    st.write(f"**Model Output:** {prediction}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn")
