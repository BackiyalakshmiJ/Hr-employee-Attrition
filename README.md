# HR Employee Attrition Prediction App

This is a **Streamlit web app** for predicting employee attrition using a CatBoost model. Users can either input employee details manually or upload a CSV file to predict attrition for multiple employees.

---

## 🛠️ Files

- `app.py` : Streamlit app code.
- `cgb_hr_attrition_best_model.pkl` : Trained CatBoost model.
- `HR_Attrition_Processed.csv` : Preprocessed dataset used for training.
- `preprocessing_tools.pkl` : Preprocessing pipeline (scaling + encoding).
- `requirements.txt` : Python dependencies.
- `.gitignore` : Git ignore rules.

---

## 🚀 Deployment on Streamlit Cloud

1. Fork or clone this repo.  
2. Ensure all `.pkl` and CSV files are included.  
3. Go to [Streamlit Cloud](https://streamlit.io/cloud).  
4. Click **New App** → Connect your GitHub repo → Deploy.  
5. Streamlit will automatically install dependencies from `requirements.txt`.  

---

## 💡 Features

- Predict attrition probability for a single employee.  
- Display results with probability.  
- Optional: upload CSV for batch predictions (can be added in next iteration).  

---

## ⚙️ Usage

1. Open the deployed app.  
2. Fill in employee details in the form.  
3. Click **Predict Attrition**.  
4. View results with probability of leaving or staying.
