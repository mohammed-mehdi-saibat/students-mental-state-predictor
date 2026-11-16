import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Locate project files and allow loading custom preprocessing functions
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
model_path = project_root / "models" / "logistic_regression_airflow.joblib"
metrics_path = project_root / "reports" / "metrics" / "airflow_evaluation.json"

st.set_page_config(page_title="SC1 Mental Health Risk", layout="centered")
st.title("Student Mental Health Risk — SC1")
st.info("Educational demonstration only. Not a clinical diagnosis, screening, or student decision tool. Use fictional inputs.")

if not model_path.exists():
    st.error(f"Model not found: {model_path}")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(model_path)


model = load_model()

# Display saved evaluation metrics
if metrics_path.exists():
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    st.subheader("Model performance")
    col1, col2 = st.columns(2)
    col1.metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
    col2.metric("Test F1 macro", f"{metrics['test_f1_macro']:.1%}")

    st.subheader("Confusion matrix")
    labels = metrics["labels"]
    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels],
    )
    st.dataframe(matrix, use_container_width=True)

# Try a fictional student profile
st.subheader("Try a prediction")
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=17, max_value=29, value=21)
    gender = st.selectbox("Gender", ["Female", "Male", "Non-binary"])
    year_of_study = st.number_input("Year of study", min_value=1, max_value=5, value=2)
    sleep_hours = st.number_input("Sleep hours per day", min_value=0.0, max_value=24.0, value=7.0)
    study_hours_per_week = st.number_input("Study hours per week", min_value=0.0, value=20.0)
    gpa = st.number_input("GPA (out of 4)", min_value=0.0, max_value=4.0, value=2.6)
    social_media_hours = st.number_input("Social media hours per day", min_value=0.0, max_value=24.0, value=6.0)
    screen_time_hours = st.number_input("Screen time hours per day", min_value=0.0, max_value=24.0, value=8.5)
    physical_activity = st.selectbox("Physical activity", ["Low", "Moderate", "High"])
    financial_stress = st.slider("Financial stress", 1.0, 10.0, 6.0)
    family_support = st.slider("Family support", 1.0, 10.0, 6.5)
    attendance_rate = st.slider("Attendance rate (%)", 0.0, 100.0, 85.0)
    part_time_job = st.selectbox("Part-time job", ["No", "Yes"])
    diet_quality = st.selectbox("Diet quality", ["Poor", "Average", "Good"])
    previous_mental_health_history = st.selectbox("Previous mental health history", ["No", "Yes"])

    submitted = st.form_submit_button("Predict (educational demo)")

if submitted:
    student = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "year_of_study": year_of_study,
        "sleep_hours": sleep_hours,
        "study_hours_per_week": study_hours_per_week,
        "gpa": gpa,
        "social_media_hours": social_media_hours,
        "screen_time_hours": screen_time_hours,
        "physical_activity": physical_activity,
        "financial_stress": financial_stress,
        "family_support": family_support,
        "attendance_rate": attendance_rate,
        "part_time_job": part_time_job,
        "diet_quality": diet_quality,
        "previous_mental_health_history": previous_mental_health_history,
    }])

    prediction = model.predict(student)[0]
    st.write(f"**Predicted dataset class: {prediction}**")
    st.caption("This label is a model output for demonstration, not a psychological assessment.")
