# app.py
import streamlit as st
import pandas as pd
import pickle

# Load the model and encoders
with open("model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    label_encoders = data["label_encoders"]
    columns = data["columns"]

st.title("🏦 Loan Approval Prediction App")
st.markdown("Enter applicant details below to predict loan approval status:")

# Create input fields dynamically
user_input = {}
for col in columns:
    user_input[col] = st.text_input(f"Enter {col}")

if st.button("Predict Loan Status"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Apply label encoding (for categorical columns)
    for col, le in label_encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = le.transform(input_df[col])
            except ValueError:
                st.warning(f"⚠️ Unknown category entered for '{col}'. Using default value 0.")
                input_df[col] = 0

    # Ensure same column order and numeric types
    input_df = input_df.reindex(columns=columns, fill_value=0)
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Make prediction
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Prediction: {'Approved ✅' if prediction == 1 else 'Rejected ❌'}")
