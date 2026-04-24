import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Loan Approval Prediction App",
    page_icon="💰",
    layout="centered"
)

st.title("Loan Approval Prediction App")
st.write(
    "This app uses the trained Logistic Regression model from the BUS 458 final project "
    "to predict whether a loan application is likely to be approved."
)

@st.cache_resource
def load_model():
    with open("my_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

if hasattr(model, "feature_names_in_"):
    expected_columns = list(model.feature_names_in_)
else:
    expected_columns = None

st.header("Enter Loan Applicant Information")

requested_loan_amount = st.number_input(
    "Requested Loan Amount",
    min_value=0,
    value=10000,
    step=500
)

fico_score = st.number_input(
    "FICO Score",
    min_value=300,
    max_value=850,
    value=700,
    step=1
)

monthly_gross_income = st.number_input(
    "Monthly Gross Income",
    min_value=0,
    value=5000,
    step=100
)

monthly_housing_payment = st.number_input(
    "Monthly Housing Payment",
    min_value=0,
    value=1500,
    step=100
)

reason = st.selectbox(
    "Loan Reason",
    [
        "credit_card_refinancing",
        "home_improvement",
        "major_purchase",
        "cover_an_unexpected_cost",
        "debt_conslidation",
        "other"
    ]
)

fico_score_group = st.selectbox(
    "FICO Score Group",
    [
        "fair",
        "good",
        "very_good",
        "excellent"
    ]
)

employment_status = st.selectbox(
    "Employment Status",
    [
        "employed",
        "self_employed",
        "unemployed",
        "retired"
    ]
)

employment_sector = st.selectbox(
    "Employment Sector",
    [
        "Unknown",
        "education",
        "finance",
        "health_care",
        "information_technology",
        "manufacturing",
        "retail",
        "transportation",
        "other"
    ]
)

lender = st.selectbox(
    "Lender",
    [
        "A",
        "B",
        "C"
    ]
)

ever_bankrupt_or_foreclose = st.selectbox(
    "Ever Bankrupt or Foreclosed?",
    [
        "no",
        "yes"
    ]
)

raw_input = pd.DataFrame({
    "Requested_Loan_Amount": [requested_loan_amount],
    "FICO_score": [fico_score],
    "Monthly_Gross_Income": [monthly_gross_income],
    "Monthly_Housing_Payment": [monthly_housing_payment],
    "Reason": [reason],
    "Fico_Score_group": [fico_score_group],
    "Employment_Status": [employment_status],
    "Employment_Sector": [employment_sector],
    "Lender": [lender],
    "Ever_Bankrupt_or_Foreclose": [ever_bankrupt_or_foreclose]
})

encoded_input = pd.get_dummies(raw_input, drop_first=True)

if expected_columns is not None:
    model_input = encoded_input.reindex(columns=expected_columns, fill_value=0)
else:
    model_input = encoded_input

if st.button("Predict Loan Approval"):
    try:
        prediction = model.predict(model_input)[0]

        if hasattr(model, "predict_proba"):
            approval_probability = model.predict_proba(model_input)[0][1]
        else:
            approval_probability = None

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("Prediction: Approved")
        else:
            st.error("Prediction: Denied")

        if approval_probability is not None:
            st.metric("Estimated Approval Probability", f"{approval_probability:.1%}")

        st.caption(
            "This prediction is based on the saved model file and should be used for educational purposes."
        )

    except Exception as e:
        st.error("The app could not generate a prediction.")
        st.write("Error details:")
        st.code(str(e))
