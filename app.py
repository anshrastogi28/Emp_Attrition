import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('Employee_Attrition_Model.pkl')
scaler = joblib.load('Employee_Attrition_Scaler.pkl')

#interface for user input
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼"
)
st.title("👨‍💼Employee Attrition Prediction")
st.write("This app predicts whether an employee is likely to leave the company based on their attributes.")

age = st.number_input("Age", min_value=18, max_value=65, value=30)
distance_from_home = st.number_input("Distance from Home (in km)", min_value=0, max_value=2000, value=10)
monthly_income = st.number_input("Monthly Income", min_value=10000, max_value=1000000, value=15000)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=1)
job_satisfaction = st.selectbox("Job Satisfaction (1-4)", options=[1, 2, 3, 4], index=2)
overtime = st.selectbox("Overtime", options=["Yes", "No"], index=1)
num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=20, value=1)
training_hours = st.number_input("Training Hours", min_value=0, max_value=100, value=10)

# Convert categorical input to numerical
overtime = 1 if overtime == "Yes" else 0

# Create a DataFrame with the user input
input_data = pd.DataFrame({
    'age': [age],
    'distance_from_home': [distance_from_home],
    'monthly_income': [monthly_income],
    'years_at_company': [years_at_company],
    'job_satisfaction': [job_satisfaction],
    'overtime': [overtime],
    'num_companies_worked': [num_companies_worked],
    'training_hours': [training_hours]
})

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make a prediction
prediction = model.predict(input_data_scaled)

# Display the prediction result
if st.button("Predict"):
    if prediction[0] == 1:
        st.error("The employee is likely to leave the company.")
    else:
        st.success("The employee is likely to stay with the company.")
        
