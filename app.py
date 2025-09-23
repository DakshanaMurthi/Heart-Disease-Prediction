import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import warnings

warnings.filterwarnings('ignore') # Suppress warnings

# --- 1. Load Data, Models, and Encoders ---
@st.cache_data # Cache the data loading for efficiency
def load_data_and_encoders():
    df = pd.read_csv('heart(final_data).csv')

    # Initialize and fit LabelEncoders for all categorical columns
    le_sex = LabelEncoder()
    le_chestpaintype = LabelEncoder()
    le_restingecg = LabelEncoder()
    le_exerciseangina = LabelEncoder()
    le_heartdisease = LabelEncoder()

    # Fit encoders on the full dataset to ensure all possible categories are learned
    # This is crucial for consistent encoding during prediction
    le_sex.fit(df['Sex'])
    le_chestpaintype.fit(df['ChestPainType'])
    le_restingecg.fit(df['RestingECG'])
    le_exerciseangina.fit(df['ExerciseAngina'])
    le_heartdisease.fit(df['HeartDisease'])

    # Store the numerical codes for 'PRESENCE' and 'ABSENCE'
    presence_code = le_heartdisease.transform(['PRESENCE'])[0]
    absence_code = le_heartdisease.transform(['ABSENCE'])[0]

    return df, le_sex, le_chestpaintype, le_restingecg, le_exerciseangina, le_heartdisease, presence_code, absence_code

df, le_sex, le_chestpaintype, le_restingecg, le_exerciseangina, le_heartdisease, PRESENCE_CODE, ABSENCE_CODE = load_data_and_encoders()

# Load the trained models and scaler
@st.cache_resource # Cache the model loading (as models are static after loading)
def load_models():
    try:
        nb_model = pickle.load(open('naive_bayes_model.pkl', 'rb'))
    except FileNotFoundError:
        st.error("Naive Bayes model file 'naive_bayes_model.pkl' not found. Please ensure your training script saves it correctly.")
        nb_model = None

    try:
        knn_model = pickle.load(open('knn_model.pkl', 'rb'))
        scaler_model = pickle.load(open('scaler.pkl', 'rb'))
    except FileNotFoundError:
        st.error("KNN model or scaler file ('knn_model.pkl' or 'scaler.pkl') not found. Please ensure your training script saves them correctly.")
        knn_model = None
        scaler_model = None

    return nb_model, knn_model, scaler_model

NB_model, KNN_model, scaler_model = load_models()


# --- NEW: Definitions for categorical features ---
# Based on image_9ae8a9.jpg
chest_pain_definitions = {
    "Typical Angina": "Pain during exertion or stress.",
     "Atypical Angina": "Unusual chest discomfort.",
    "Non-Anginal Pain": "Pain not related to the heart.",
    "Asymptomatic": "No chest pain."
}

# Based on image_9ae8c9.jpg
resting_ecg_definitions = {
  "Normal": "Normal heart electrical activity.",
    "ST-T wave abnormality": "Possible heart ischemia.",
    "Left ventricular hypertrophy": "Thickened heart muscle."
}

# --- 2. Prediction Function ---
def predict_heart_disease(model, scaler, model_name,
                          age, sex_str, chest_pain_type_str, resting_bp, cholesterol,
                          resting_ecg_str, max_hr, exercise_angina_str, fasting_bs_num_val):
    try:
        encoded_sex = le_sex.transform([sex_str])[0]
        encoded_chest_pain_type = le_chestpaintype.transform([chest_pain_type_str])[0]
        encoded_resting_ecg = le_restingecg.transform([resting_ecg_str])[0]
        encoded_exercise_angina = le_exerciseangina.transform([exercise_angina_str])[0]
    except ValueError as e:
        st.error(f"Error encoding categorical feature: {e}. Please ensure input values are valid.")
        return

    patient_features = [
        age,
        encoded_sex,
        encoded_chest_pain_type,
        resting_bp,
        cholesterol,
        encoded_resting_ecg,
        max_hr,
        encoded_exercise_angina,
        fasting_bs_num_val
    ]

    patient_data_array = np.array(patient_features).reshape(1, -1)

    if scaler:
        patient_data_array = scaler.transform(patient_data_array)

    prediction = model.predict(patient_data_array)[0]

    st.subheader("Prediction Result:")
    if prediction == PRESENCE_CODE:
        st.error(f"Prediction ({model_name}): The Patient **has heart disease** ({le_heartdisease.inverse_transform([prediction])[0]}). Please consult a doctor.")
    else:
        st.success(f"Prediction ({model_name}): The Patient **does NOT have heart disease** ({le_heartdisease.inverse_transform([prediction])[0]}).")

        advice_given = False
        if cholesterol > 200:
            if resting_bp > 120:
                st.warning("🚨 **Warning:** Both blood pressure and cholesterol are high. Please consult a doctor.")
            elif resting_bp < 120:
                st.warning("⚠️ **Caution:** Blood pressure is low, but cholesterol is high. Please consult a doctor.")
            elif resting_bp == 120:
                st.warning("⚠️ **Caution:** Blood pressure is normal but cholesterol is high. Please consult a doctor.")
            advice_given = True
        elif resting_bp > 120:
            st.warning("🚨 **Warning:** Blood pressure is high. Please consult a doctor.")
            advice_given = True

        if not advice_given:
            st.info("✅ **All readings appear normal.**")

# --- 3. Streamlit UI ---
# Corrected: Removed the 'icon' parameter as it's not supported in older Streamlit versions.
st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter patient details below to predict the likelihood of heart disease.")

# User input sections
st.header("Patient Information")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=40)

    sex = st.selectbox("Sex", options=le_sex.classes_)

    chest_pain_type = st.selectbox("Chest Pain Type", options=le_chestpaintype.classes_, help="Type of chest pain experienced.")
    if chest_pain_type in chest_pain_definitions:
        st.info(f"**{chest_pain_type}:** {chest_pain_definitions[chest_pain_type]}") # Display definition

    resting_bp = st.number_input("Resting Blood Pressure (mm/Hg)", min_value=70, max_value=200, value=140, help="Systolic blood pressure at rest.")

with col2:
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=289, help="Serum cholesterol level.")

    resting_ecg = st.selectbox("Resting ECG Results", options=le_restingecg.classes_, help="Results of electrocardiogram at rest.")
    if resting_ecg in resting_ecg_definitions:
        st.info(f"**{resting_ecg}:** {resting_ecg_definitions[resting_ecg]}") # Display definition

    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=172, help="Highest heart rate achieved during exercise.")

    exercise_angina = st.selectbox("Exercise Induced Angina", options=le_exerciseangina.classes_, help="Chest pain or discomfort during exercise.")

    fasting_bs_num = st.number_input("Fasting Blood Sugar (mg/dl)", min_value=50, max_value=300, value=108, help="Fasting blood sugar level (>120 mg/dL indicates high).")


st.markdown("---")
st.header("Model Selection & Prediction")

selected_model = st.selectbox("Choose Model for Prediction", options=["Naive Bayes", "KNN"])

if st.button("Get Prediction"):
    if selected_model == "Naive Bayes":
        if NB_model:
            predict_heart_disease(NB_model, None, "Naive Bayes",
                                  age, sex, chest_pain_type, resting_bp, cholesterol,
                                  resting_ecg, max_hr, exercise_angina, fasting_bs_num)
        else:
            st.error("Naive Bayes model not loaded. Check console for errors during model loading.")
    elif selected_model == "KNN":
        if KNN_model and scaler_model:
            predict_heart_disease(KNN_model, scaler_model, "KNN",
                                  age, sex, chest_pain_type, resting_bp, cholesterol,
                                  resting_ecg, max_hr, exercise_angina, fasting_bs_num)
        else:
            st.error("KNN model or scaler not loaded. Check console for errors during model loading.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit")

#cd "C:\Users\DAKSHANAMURTHI\OneDrive\Desktop\bomb_squrd"
#dir
#streamlit run app.py
