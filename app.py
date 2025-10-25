# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np

# with open("best_model.pkl", "rb") as f:
#     model = pickle.load(f)

# st.title(" Health Risk Prediction")
# st.write("Enter the baby's details below to predict the health status.")

# # --- Binary inputs (Yes/No) ---
# gender = st.radio("Gender", ["Male", "Female"])
# immunizations_done = st.radio("Immunizations Done?", ["Yes", "No"])
# reflexes_normal = st.radio("Reflexes Normal?", ["Yes", "No"])

# feeding_type = st.selectbox(
#     "Feeding Type", 
#     ["Breastfeeding", "Formula", "Mixed"]
# )

# # Convert to numerical
# gender = 1 if gender == "Male" else 0
# immunizations_done = 1 if immunizations_done == "Yes" else 0
# reflexes_normal = 1 if reflexes_normal == "Yes" else 0

# feeding_type_Breastfeeding = 1 if feeding_type == "Breastfeeding" else 0
# feeding_type_Formula = 1 if feeding_type == "Formula" else 0
# feeding_type_Mixed = 1 if feeding_type == "Mixed" else 0

# # --- Numeric inputs with realistic ranges ---
# gestational_age_weeks = st.number_input("Gestational Age (weeks)", 34.0, 45.0, 37.0, 0.1)
# birth_weight_kg = st.number_input("Birth Weight (kg)", 1.5, 5.0, 3.0, 0.01)
# birth_length_cm = st.number_input("Birth Length (cm)", 43.0, 56.0, 50.0, 0.1)
# birth_head_circumference_cm = st.number_input("Birth Head Circumference (cm)", 30.0, 40.0, 33.5, 0.1)
# age_days = st.number_input("Age (days)", 1.0, 30.0, 20.0, 1.0)
# weight_kg = st.number_input("Current Weight (kg)", 1.5, 6.5, 3.4, 0.01)
# length_cm = st.number_input("Current Length (cm)", 43.0, 56.0, 50.0, 0.1)
# head_circumference_cm = st.number_input("Current Head Circumference (cm)", 30.0, 41.0, 34.0, 0.1)
# temperature_c = st.number_input("Temperature (°C)", 35.0, 40.0, 37.0, 0.1)
# heart_rate_bpm = st.number_input("Heart Rate (bpm)", 90.0, 180.0, 140.0, 1.0)
# respiratory_rate_bpm = st.number_input("Respiratory Rate (breaths/min)", 20.0, 65.0, 35.0, 1.0)
# oxygen_saturation = st.number_input("Oxygen Saturation (%)", 80.0, 100.0, 98.0, 0.1)
# feeding_frequency_per_day = st.number_input("Feeding Frequency (times/day)", 1.0, 12.0, 8.0, 1.0)
# urine_output_count = st.number_input("Urine Output Count", 0.0, 10.0, 8.0, 1.0)
# stool_count = st.number_input("Stool Count", 0.0, 6.0, 5.0, 1.0)
# jaundice_level_mg_dl = st.number_input("Jaundice Level (mg/dL)", 0.0, 25.0, 2.0, 0.1)
# apgar_score = st.number_input("APGAR Score", 4.0, 10.0, 8.0, 1.0)

# # --- Combine features into a DataFrame ---
# features = pd.DataFrame([[
#     gender, gestational_age_weeks, birth_weight_kg, birth_length_cm, 
#     birth_head_circumference_cm, age_days, weight_kg, length_cm, 
#     head_circumference_cm, temperature_c, heart_rate_bpm, respiratory_rate_bpm, 
#     oxygen_saturation, feeding_frequency_per_day, urine_output_count, stool_count, 
#     jaundice_level_mg_dl, apgar_score, immunizations_done, reflexes_normal,
#     feeding_type_Breastfeeding, feeding_type_Formula, feeding_type_Mixed
# ]],
#  columns=[
#     'gender', 'gestational_age_weeks', 'birth_weight_kg', 'birth_length_cm',
#     'birth_head_circumference_cm', 'age_days', 'weight_kg', 'length_cm',
#     'head_circumference_cm', 'temperature_c', 'heart_rate_bpm', 
#     'respiratory_rate_bpm', 'oxygen_saturation', 'feeding_frequency_per_day', 
#     'urine_output_count', 'stool_count', 'jaundice_level_mg_dl', 'apgar_score',
#     'immunizations_done', 'reflexes_normal', 'feeding_type_Breastfeeding',
#     'feeding_type_Formula', 'feeding_type_Mixed'
# ])

# Threshold = 0.21

# if st.button(" Predict"):
#     probs = model.predict_proba(features)[0]
#     prediction = 0 if probs[0] >= Threshold else 1

#     if prediction == 0:
#         st.error(f" Prediction: At Risk \n\n Probability: {probs[0]*100:.2f}%")
#     else:
#         st.success(f" Prediction: Healthy \n\n Probability: {probs[1]*100:.2f}%")

#     st.write("### Detailed Probabilities")
#     st.write(f"- **At Risk:** {probs[0]*100:.2f}%")
#     st.write(f"- **Healthy:** {probs[1]*100:.2f}%")


import streamlit as st
import pickle
import pandas as pd
import numpy as np

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title(" Baby Health Risk Prediction")
st.markdown("""
This app predicts whether a baby is **At Risk** or **Healthy**  
based on medical parameters.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Input Features")
st.sidebar.write("Enter the baby's details to predict the health status.")

# --- Binary inputs (Yes/No) ---
gender = st.sidebar.radio("Gender", ["Male", "Female"])
gender = 0 if gender == "Male" else 1

immunizations_done = st.sidebar.radio("Immunizations Done?", ["Yes", "No"])
immunizations_done = 1 if immunizations_done == "Yes" else 0

reflexes_normal = st.sidebar.radio("Reflexes ", ["Normal", "Not Normal"])
reflexes_normal = 1 if reflexes_normal == "Normal" else 0

feeding_type = st.sidebar.selectbox("Feeding Type", ["Breastfeeding", "Formula", "Mixed"])

# Convert to numerical
feeding_type_Breastfeeding = 1 if feeding_type == "Breastfeeding" else 0
feeding_type_Formula = 1 if feeding_type == "Formula" else 0
feeding_type_Mixed = 1 if feeding_type == "Mixed" else 0

# ---  inputs  ---
gestational_age_weeks = st.sidebar.number_input("Gestational Age (weeks)", 34.0, 45.0, 37.0, 0.1)
birth_weight_kg = st.sidebar.number_input("Birth Weight (kg)", 1.5, 5.0, 3.0, 0.01)
birth_length_cm = st.sidebar.number_input("Birth Length (cm)", 43.0, 56.0, 50.0, 0.1)
birth_head_circumference_cm = st.sidebar.number_input("Birth Head Circumference (cm)", 30.0, 40.0, 33.5, 0.1)
age_days = st.sidebar.number_input("Age (days)", 1.0, 30.0, 20.0, 1.0)
weight_kg = st.sidebar.number_input("Current Weight (kg)", 1.5, 6.5, 3.4, 0.01)
length_cm = st.sidebar.number_input("Current Length (cm)", 43.0, 56.0, 50.0, 0.1)
head_circumference_cm = st.sidebar.number_input("Current Head Circumference (cm)", 30.0, 41.0, 34.0, 0.1)
temperature_c = st.sidebar.number_input("Temperature (°C)", 35.0, 40.0, 37.0, 0.1)
heart_rate_bpm = st.sidebar.number_input("Heart Rate (bpm)", 90, 180, 140, 1)
respiratory_rate_bpm = st.sidebar.number_input("Respiratory Rate (breaths/min)", 20, 65, 35, 1)
oxygen_saturation = st.sidebar.number_input("Oxygen Saturation (%)", 80, 100, 98, 1)
feeding_frequency_per_day = st.sidebar.number_input("Feeding Frequency (times/day)", 1, 12, 8, 1)
urine_output_count = st.sidebar.number_input("Urine Output Count", 0, 10, 8, 1)
stool_count = st.sidebar.number_input("Stool Count", 0, 6, 5, 1)
jaundice_level_mg_dl = st.sidebar.number_input("Jaundice Level (mg/dL)", 0.0, 25.0, 2.0, 0.1)
apgar_score = st.sidebar.number_input("APGAR Score", 4.0, 10.0, 8.0, 1.0)

# --- features into DataFrame ---
features = pd.DataFrame([[
    gender, gestational_age_weeks, birth_weight_kg, birth_length_cm,
    birth_head_circumference_cm, age_days, weight_kg, length_cm,
    head_circumference_cm, temperature_c, heart_rate_bpm, respiratory_rate_bpm,
    oxygen_saturation, feeding_frequency_per_day, urine_output_count, stool_count,
    jaundice_level_mg_dl, apgar_score, immunizations_done, reflexes_normal,
    feeding_type_Breastfeeding, feeding_type_Formula, feeding_type_Mixed
]], 
    columns=[
    'gender', 'gestational_age_weeks', 'birth_weight_kg', 'birth_length_cm',
    'birth_head_circumference_cm', 'age_days', 'weight_kg', 'length_cm',
    'head_circumference_cm', 'temperature_c', 'heart_rate_bpm',
    'respiratory_rate_bpm', 'oxygen_saturation', 'feeding_frequency_per_day',
    'urine_output_count', 'stool_count', 'jaundice_level_mg_dl', 'apgar_score',
    'immunizations_done', 'reflexes_normal', 'feeding_type_Breastfeeding',
    'feeding_type_Formula', 'feeding_type_Mixed']
)

# --- Prediction logic ---
Threshold = 0.5

if st.button("Predict"):
    probs = model.predict_proba(features)[0]
    prediction = 0 if probs[0] >= Threshold else 1

    if prediction == 0:
        st.error(f"Prediction: At Risk \n\n Probability: {probs[0]*100:.2f}%")
    else:
        st.success(f"Prediction: Healthy \n\n Probability: {probs[1]*100:.2f}%")

    st.write("### Detailed Probabilities")
    st.write(f"- **At Risk:** {probs[0]*100:.2f}%")
    st.write(f"- **Healthy:** {probs[1]*100:.2f}%")
    
    # Probability bar chart
    prob_df = pd.DataFrame({
        "Class": ["At Risk", "Healthy"],
        "Probability (%)": [probs[0]*100, probs[1]*100]
    })
    st.write("### Probability Chart")
    st.bar_chart(prob_df.set_index("Class"))
