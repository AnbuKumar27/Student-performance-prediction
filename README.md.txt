# Student Performance Prediction Using Machine Learning

A beginner-friendly college mini-project that trains a Random Forest
classifier to predict a student's illustrative performance category
(Low, Medium, High) from study hours, attendance, previous score and
assignment score.

> **Important:** `dataset.csv` is synthetic demonstration data, not real
> student records. The model is not validated for real academic
> decisions. Predictions should never be used to label, punish, admit or
> exclude students.

## Features

-   Loads and explores a tabular CSV dataset.
-   Splits data into training and test sets.
-   Trains a Random Forest classification pipeline.
-   Prints accuracy, classification report and confusion matrix.
-   Saves the trained model.
-   Provides a simple Streamlit web interface for trying sample inputs.

## Project files

-   `dataset.csv` --- illustrative synthetic dataset.
-   `train_model.py` --- model training and evaluation.
-   `app.py` --- Streamlit user interface.
-   `requirements.txt` --- Python dependencies.
-   `project_report.pdf` --- college project report.

## Run locally

1.  Install Python 3.10 or newer.
2.  Open a terminal in this project folder.
3.  (Recommended) Create and activate a virtual environment:
    -   Windows: `python -m venv .venv` then `.venv\Scripts\activate`
    -   macOS/Linux: `python3 -m venv .venv` then
        `source .venv/bin/activate`
4.  Install packages: `pip install -r requirements.txt`
5.  Train the model: `python train_model.py`
6.  Start the interface: `streamlit run app.py`
7.  Open the local address shown in the terminal (usually
    `http://localhost:8501`).

## How it works

The model uses four input features: - Study hours per day - Attendance
percentage - Previous exam score - Assignment score

A Random Forest classifier learns patterns from the example rows and
predicts one of three categories. The data labels were generated using a
simple illustrative scoring rule; therefore, high test accuracy only
shows that the model can learn patterns in this artificial dataset. It
does not demonstrate real-world predictive validity.

## GitHub submission

1.  Sign in to GitHub and create a new public repository named
    `Student-Performance-Prediction`.

2.  Extract this ZIP file on your computer.

3.  Upload all project files to the repository (or use Git):

    ``` bash
    git init
    git add .
    git commit -m "Add student performance prediction ML project"
    git branch -M main
    git remote add origin https://github.com/YOUR-USERNAME/Student-Performance-Prediction.git
    git push -u origin main
    ```

4.  Replace `YOUR-USERNAME` with your GitHub username.

5.  Copy the repository page URL and submit it to your faculty.

## Suggested viva points

-   **Machine Learning:** A method that allows a computer to learn
    patterns from data.
-   **Supervised learning:** Learning from examples that include known
    labels.
-   **Classification:** Predicting a category rather than a continuous
    number.
-   **Random Forest:** An ensemble of decision trees whose predictions
    are combined.
-   **Train/test split:** Keeping some records aside to evaluate
    performance on unseen examples.
-   **Accuracy:** The fraction of test predictions that are correct; it
    should be considered alongside other metrics.

## Future enhancements

Use a properly consented, representative real dataset; assess bias and
privacy; compare multiple algorithms; use cross-validation; and evaluate
the model with educators before any real-world use.
