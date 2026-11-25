"""
Train and evaluate the Student Performance Prediction model.
Dataset is illustrative/synthetic and intended for a classroom demonstration.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset.csv"
MODEL_PATH = ROOT / "student_performance_model.joblib"

FEATURES = [
    "study_hours_per_day",
    "attendance_percent",
    "previous_score",
    "assignment_score",
]

def main():
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data["performance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (labels Low, Medium, High):")
    print(confusion_matrix(y_test, predictions, labels=["Low", "Medium", "High"]))

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained model to: {MODEL_PATH.name}")

if __name__ == "__main__":
    main()
