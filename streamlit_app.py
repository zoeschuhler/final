import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Predictive Model App", layout="centered")

st.title("Predictive Model App")
st.write("Enter the information below to generate a prediction.")

@st.cache_resource
def load_model():
    with open("my_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

st.subheader("Input Values")

# Replace these with the exact variables your model was trained on
fico_score = st.number_input("FICO Score", min_value=300, max_value=850, value=700)
monthly_gross_income = st.number_input("Monthly Gross Income", min_value=0, value=5000)
requested_loan_amount = st.number_input("Requested Loan Amount", min_value=0, value=10000)

input_df = pd.DataFrame({
    "FICO_score": [fico_score],
    "Monthly_Gross_Income": [monthly_gross_income],
    "Requested_Loan_Amount": [requested_loan_amount]
})

st.write("Model input:")
st.dataframe(input_df)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction")
    st.write(prediction)
