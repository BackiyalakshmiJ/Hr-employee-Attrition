# app.py
import streamlit as st
import pandas as pd
import pickle

# -------------------------
# Load Preprocessing Pipeline & Trained Model
# -------------------------
try:
    with open("preprocessing.pkl", "rb") as f:
        preprocessing = pickle.load(f)
except FileNotFoundError:
    st.error("Preprocessing file not found: preprocessing.pkl. Make sure it's in the same folder as app.py.")
    st.stop()

try:
    with open("best_hr_attrition_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found: best_hr_attrition_model.pkl. Make sure it's in the same folder as app.py.")
    st.stop()

st.title("HR Employee Attrition Prediction")
st.write("Predict whether an employee is likely to leave the company.")

# -------------------------
# Input form for single employee
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
# Preprocess input
# -------------------------
try:
    input_processed = preprocessing.transform(input_df)
except Exception as e:
    st.error(f"Error during preprocessing: {e}")
    st.stop()

# -------------------------
# Prediction
# -------------------------
try:
    prediction = model.predict(input_processed)
    prediction_proba = model.predict_proba(input_processed)[:, 1]
except Exception as e:
    st.error(f"Error during prediction: {e}")
    st.stop()

# -------------------------
# Display Results
# -------------------------
st.subheader("Employee Input:")
st.write(input_df)

st.subheader("Prediction:")
st.write("Likely to Leave" if prediction[0]==1 else "Likely to Stay")

st.subheader("Prediction Probability:")
st.write(f"{prediction_proba[0]*100:.2f}%")
