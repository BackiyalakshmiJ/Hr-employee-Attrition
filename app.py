import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------
# Load Model, Scaler, and Encoders
# -------------------------
try:
    with open("best_hr_attrition_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}. Please make sure all required files are in the same folder as app.py.")
    st.stop()

# -------------------------
# App Header
# -------------------------
st.title("HR Employee Attrition Prediction")
st.write("Fill in employee details to predict whether they are likely to leave the company.")

# -------------------------
# Input form
# -------------------------
st.sidebar.header("Employee Details")

def user_input_features():
    Age = st.sidebar.number_input("Age", 18, 60, 30)
    BusinessTravel = st.sidebar.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    Department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    DistanceFromHome = st.sidebar.number_input("Distance From Home", 1, 50, 10)
    Education = st.sidebar.selectbox("Education", [1, 2, 3, 4, 5])
    EducationField = st.sidebar.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    JobRole = st.sidebar.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
                                                "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
    MaritalStatus = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    OverTime = st.sidebar.selectbox("OverTime", ["Yes", "No"])

    data = {
        "Age": Age,
        "BusinessTravel": BusinessTravel,
        "Department": Department,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EducationField": EducationField,
        "Gender": Gender,
        "JobRole": JobRole,
        "MaritalStatus": MaritalStatus,
        "OverTime": OverTime
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# -------------------------
# Preprocess the input
# -------------------------
try:
    # Define mappings same as used during training
    business_travel_map = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    department_map = {"Sales": 0, "Research & Development": 1, "Human Resources": 2}
    educationfield_map = {
        "Life Sciences": 0, "Medical": 1, "Marketing": 2,
        "Technical Degree": 3, "Other": 4, "Human Resources": 5
    }
    gender_map = {"Male": 0, "Female": 1}
    jobrole_map = {
        "Sales Executive": 0, "Research Scientist": 1, "Laboratory Technician": 2,
        "Manufacturing Director": 3, "Healthcare Representative": 4,
        "Manager": 5, "Sales Representative": 6, "Research Director": 7,
        "Human Resources": 8
    }
    maritalstatus_map = {"Single": 0, "Married": 1, "Divorced": 2}
    overtime_map = {"Yes": 1, "No": 0}

    input_df["BusinessTravel"] = input_df["BusinessTravel"].map(business_travel_map)
    input_df["Department"] = input_df["Department"].map(department_map)
    input_df["EducationField"] = input_df["EducationField"].map(educationfield_map)
    input_df["Gender"] = input_df["Gender"].map(gender_map)
    input_df["JobRole"] = input_df["JobRole"].map(jobrole_map)
    input_df["MaritalStatus"] = input_df["MaritalStatus"].map(maritalstatus_map)
    input_df["OverTime"] = input_df["OverTime"].map(overtime_map)

    # Scale the numeric data
    input_df_scaled = scaler.transform(input_df)

except Exception as e:
    st.error(f"Error in preprocessing: {e}")
    st.stop()

# -------------------------
# Display input
# -------------------------
st.subheader("Employee Input:")
st.write(input_df)

# -------------------------
# Prediction
# -------------------------
prediction = model.predict(input_df_scaled)
prediction_proba = model.predict_proba(input_df_scaled)[:, 1]

st.subheader("Prediction:")
st.write("Likely to Leave" if prediction[0] == 1 else "Likely to Stay")

st.subheader("Prediction Probability:")
st.write(f"{prediction_proba[0]*100:.2f}%")
