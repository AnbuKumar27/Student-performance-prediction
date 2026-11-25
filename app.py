"""Simple Streamlit interface for the student performance demo."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_performance_model.joblib"
FEATURES = [
    "study_hours_per_day",
    "attendance_percent",
    "previous_score",
    "assignment_score",
]

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Prediction")
st.write(
    "Enter academic information to get a demonstration prediction. "
    "This model uses an illustrative synthetic dataset; it must not be used "
    "to make real academic decisions about students."
)

if not MODEL_PATH.exists():
    st.warning("The model file is missing. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

col1, col2 = st.columns(2)
with col1:
    study = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
with col2:
    previous = st.slider("Previous exam score", 0, 100, 65)
    assignment = st.slider("Assignment score", 0, 100, 70)

if st.button("Predict performance", type="primary"):
    sample = pd.DataFrame([{
        "study_hours_per_day": study,
        "attendance_percent": attendance,
        "previous_score": previous,
        "assignment_score": assignment,
    }], columns=FEATURES)
    result = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    classes = model.named_steps["classifier"].classes_

    st.subheader(f"Predicted category: {result}")
    st.write("Model confidence scores (not guaranteed probabilities of real-world success):")
    st.bar_chart(pd.Series(probabilities, index=classes))
    if result == "High":
        st.success("The model predicts a high performance category.")
    elif result == "Medium":
        st.info("The model predicts a medium performance category.")
    else:
        st.warning("The model predicts a low performance category. Consider offering supportive resources.")
    st.caption("This is a learning prototype, not a validated educational assessment tool.")
