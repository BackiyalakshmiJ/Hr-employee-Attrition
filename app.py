# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -------------------------
# Load model
# -------------------------
with open("cgb_hr_attrition_best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scalers and label encoders
with open("scaler.pkl", "rb") as f:
    scaler_data = pickle.load(f)
scaler = scaler_data["scaler"]
numeric_cols = scaler_data["numeric_cols"]

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# -------------------------
# Streamlit UI
# -------------------------
st.title("HR Employee Attrition Prediction")
st.markdown("""
Predict whether an employee is likely to **leave the company** or **stay**.
""")

# Input form
with st.form(key="attrition_form"):
    st.subheader("Employee Details")
    
    # Numeric inputs
    Age = st.number_input("Age", min_value=18, max_value=65, value=30)
    DailyRate = st.number_input("Daily Rate", min_value=100, max_value=2000, value=500)
    DistanceFromHome = st.number_input("Distance From Home", min_value=1, max_value=50, value=10)
    Education = st.number_input("Education Level (1-5)", min_value=1, max_value=5, value=3)
    EnvironmentSatisfaction = st.number_input("Environment Satisfaction (1-4)", min_value=1, max_value=4, value=3)
    HourlyRate = st.number_input("Hourly Rate", min_value=30, max_value=150, value=60)
    JobInvolvement = st.number_input("Job Involvement (1-4)", min_value=1, max_value=4, value=3)
    JobLevel = st.number_input("Job Level (1-5)", min_value=1, max_value=5, value=2)
    JobSatisfaction = st.number_input("Job Satisfaction (1-4)", min_value=1, max_value=4, value=3)
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
    NumCompaniesWorked = st.number_input("Number of Companies Worked", min_value=0, max_value=20, value=1)
    PercentSalaryHike = st.number_input("Percent Salary Hike", min_value=0, max_value=50, value=15)
    PerformanceRating = st.number_input("Performance Rating (1-4)", min_value=1, max_value=4, value=3)
    RelationshipSatisfaction = st.number_input("Relationship Satisfaction (1-4)", min_value=1, max_value=4, value=3)
    StockOptionLevel = st.number_input("Stock Option Level (0-3)", min_value=0, max_value=3, value=1)
    TotalWorkingYears = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)
    TrainingTimesLastYear = st.number_input("Training Times Last Year", min_value=0, max_value=20, value=2)
    WorkLifeBalance = st.number_input("Work Life Balance (1-4)", min_value=1, max_value=4, value=3)
    YearsAtCompany = st.number_input("Years at Company", min_value=0, max_value=40, value=3)
    YearsInCurrentRole = st.number_input("Years in Current Role", min_value=0, max_value=20, value=2)
    YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)
    YearsWithCurrManager = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=2)
    
    # Categorical inputs
    BusinessTravel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    EducationField = st.selectbox("Education Field", ["Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    JobRole = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    OverTime = st.selectbox("OverTime", ["Yes", "No"])
    
    submit_btn = st.form_submit_button("Predict Attrition")

if submit_btn:
    # Create dataframe from input
    input_dict = {
        'Age': Age, 'DailyRate': DailyRate, 'DistanceFromHome': DistanceFromHome,
        'Education': Education, 'EnvironmentSatisfaction': EnvironmentSatisfaction,
        'HourlyRate': HourlyRate, 'JobInvolvement': JobInvolvement, 'JobLevel': JobLevel,
        'JobSatisfaction': JobSatisfaction, 'MonthlyIncome': MonthlyIncome,
        'NumCompaniesWorked': NumCompaniesWorked, 'PercentSalaryHike': PercentSalaryHike,
        'PerformanceRating': PerformanceRating, 'RelationshipSatisfaction': RelationshipSatisfaction,
        'StockOptionLevel': StockOptionLevel, 'TotalWorkingYears': TotalWorkingYears,
        'TrainingTimesLastYear': TrainingTimesLastYear, 'WorkLifeBalance': WorkLifeBalance,
        'YearsAtCompany': YearsAtCompany, 'YearsInCurrentRole': YearsInCurrentRole,
        'YearsSinceLastPromotion': YearsSinceLastPromotion, 'YearsWithCurrManager': YearsWithCurrManager,
        'BusinessTravel': BusinessTravel, 'Department': Department, 'EducationField': EducationField,
        'Gender': Gender, 'JobRole': JobRole, 'MaritalStatus': MaritalStatus, 'OverTime': OverTime
    }
    input_df = pd.DataFrame([input_dict])
    
    # Encode categorical columns
    for col in label_encoders:
        if col in input_df.columns:
            input_df[col] = label_encoders[col].transform(input_df[col])
    
    # Scale numeric columns
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # Predict
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    st.subheader("Prediction Result")
    st.write("✅ Employee likely to **leave**" if pred==1 else "✅ Employee likely to **stay**")
    st.write(f"Attrition Probability: {prob:.2f}")
