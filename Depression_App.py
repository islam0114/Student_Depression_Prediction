import numpy as np
import streamlit as st
import pandas as pd
import pickle

Data = pickle.load(open('milestone_4 (deployment)/xgboost.sav', 'rb'))

st.set_page_config(page_title="Student Depression", page_icon="🧠", layout="wide")
st.title("Student Depression Prediction App")
st.markdown('Predict whether a student is depressed based on various attributes.')

# Input Form
st.markdown("---")
st.subheader("📝 Enter Student Data:")
Academic_Pressure = st.number_input("📚 Academic Pressure (0 ==> 5)", min_value=0.0, max_value=5.0,step=0.01)

Study_Satisfaction = st.number_input("😌 Study Satisfaction (0 ==> 5)", min_value=0.0, max_value=5.0,step=0.01)

mapping_1 = {'Healthy': 0, 'Moderate': 1, "Unhealthy": 2}
mapping_2 = {'No': 0, 'Yes': 1}

Dietary_Habits_choice = st.selectbox('🍽️ Dietary Habits', list(mapping_1.keys()))
Have_you_ever_had_suicidal_thoughts_choice = st.selectbox('🌧️ Have you ever had suicidal thoughts ?', list(mapping_2.keys()))

Dietary_Habits = mapping_1[Dietary_Habits_choice]
Have_you_ever_had_suicidal_thoughts = mapping_2[Have_you_ever_had_suicidal_thoughts_choice]

Work_Study_Hours = st.number_input("🕒 Study Hours (0 ==> 12)", min_value=0.0, max_value=12.0,step=0.01)

Financial_Stress = st.number_input("💰 Financial Stress (0 ==> 5)", min_value=0.0, max_value=5.0,step=0.01)

mapping_3 = {'No': 0, 'Yes': 1}
Bullying = st.selectbox('😢 Bullying', list(mapping_3.keys()))
Bullying = mapping_3[Bullying]

PHQ_9 = st.number_input("📋 PHQ-9 (0 ==> 27)", min_value=0.0, max_value=27.0,step=0.01,help = "is a 9-item questionnaire used to assess the severity of depression in individuals.")
st.markdown("to do PHQ-9 test: [Link](https://psychology-tools.com/test/phq-9)")

Cortisol_Level = st.number_input("😰 Cortisol Level (0 ==> 10)", min_value=0.0, max_value=10.0,step=0.01,help = "is a hormone produced by the adrenal glands, often referred to as the 'stress hormone.' It helps in regulating metabolism, controlling blood sugar levels, reducing inflammation, and managing stress.")

df = pd.DataFrame({
    "Academic_Pressure": [Academic_Pressure],
    "Study_Satisfaction": [Study_Satisfaction],
    "Dietary_Habits": [Dietary_Habits],
    "Have_you_ever_had_suicidal_thoughts": [Have_you_ever_had_suicidal_thoughts],
    "Work_Study_Hours": [Work_Study_Hours],
    "Financial_Stress": [Financial_Stress],
    "Bullying": [Bullying],
    "PHQ_9": [PHQ_9],
    "Cortisol_Level": [Cortisol_Level]
}, index=[0])

df = df.rename(columns={
    'Academic_Pressure': 'Academic Pressure',
    'Study_Satisfaction': 'Study Satisfaction',
    'Dietary_Habits': 'Dietary Habits',
    'Have_you_ever_had_suicidal_thoughts': 'Have you ever had suicidal thoughts ?',
    'Work_Study_Hours': 'Work/Study Hours',
    'Financial_Stress': 'Financial Stress',
    'PHQ_9': 'PHQ-9',
})

correct_order = [
    'Have you ever had suicidal thoughts ?',
    'PHQ-9',
    'Academic Pressure',
    'Bullying',
    'Cortisol_Level',
    'Financial Stress',
    'Study Satisfaction',
    'Dietary Habits',
    'Work/Study Hours'
]

df = df[correct_order]

con = st.button("🔍 predict depression")
if con:
    result = Data.predict(df)
    if result == 1:
        st.write("The student is likely to be depressed. 😢")
    else:
        st.write("The student is not likely to be depressed. 😌")
