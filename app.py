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
        scaler_data = pickle.load(f)
        scaler = scaler_data["scaler"]
        numeric_cols = scaler_data["numeric_cols"]

    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}. Make sure all required files are in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# -------------------------
# App Header
# -------------------------
st.title("HR Employee Attrition Prediction")
st.write("Fill in employee details to predict whether they are likely to leave the company.")

# -------------------------
# Input Form
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
# Encode categorical columns
# -------------------------
try:
    for col in encoders:
        if col in input_df.columns:
            le = encoders[col]
            valid_classes = le.classes_
            val = input_df.at[0, col]
            if val in valid_classes:
                input_df[col] = le.transform([val])[0]
            else:
                # Handle unseen category by mapping to most frequent or default class
                input_df[col] = le.transform([valid_classes[0]])[0]
except Exception as e:
    st.error(f"Error in preprocessing: {e}")
    st.stop()

# -------------------------
# Ensure only numeric columns are scaled
# -------------------------
try:
    # Check all columns to be scaled are numeric
    for col in numeric_cols:
        if col not in input_df.columns:
            input_df[col] = 0  # Default value if missing
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
except Exception as e:
    st.error(f"Error scaling data: {e}")
    st.stop()

# -------------------------
# Display Input
# -------------------------
st.subheader("Employee Input:")
st.write(input_df)

# -------------------------
# Prediction
# -------------------------
try:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]
except Exception as e:
    st.error(f"Error making prediction: {e}")
    st.stop()

st.subheader("Prediction:")
st.write("Likely to Leave" if prediction[0] == 1 else "Likely to Stay")

st.subheader("Prediction Probability:")
st.write(f"{prediction_proba[0]*100:.2f}%")
