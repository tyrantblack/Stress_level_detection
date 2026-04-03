import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Stress AI Dashboard", layout="wide")
st.title("🧠 AI Stress Prediction Dashboard")

section = st.sidebar.radio("Go to:", ["Upload & EDA", "Model Performance", "Prediction", "Batch Testing"])

# ------------------ GLOBAL FEATURES ------------------
FEATURES = [
    "Study_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day"
]

# ================== PIPELINE FUNCTIONS ==================

def clean_data(df):
    df.columns = df.columns.str.strip()
    return df.dropna(subset=FEATURES + ["Stress_Level"])

def encode_target(df):
    le = LabelEncoder()
    df["Stress_Level"] = le.fit_transform(df["Stress_Level"])
    return df, le

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, report, cm

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

def plot_roc(y_true, y_score, classes):
    y_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots()
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, label=f"Class {i} (AUC={auc(fpr,tpr):.2f})")
    ax.legend()
    st.pyplot(fig)

def fix_test_columns(df):
    df.columns = df.columns.str.strip().str.lower()

    mapping = {
        "study hours per day": "Study_Hours_Per_Day",
        "sleep hours per day": "Sleep_Hours_Per_Day",
        "physical activity hours per day": "Physical_Activity_Hours_Per_Day",
        "stress level": "Stress_Level"
    }

    df.rename(columns=mapping, inplace=True)
    return df

# ================== MAIN ==================

uploaded_file = st.file_uploader("Upload Training CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # Pipeline
    df = clean_data(df)
    df, le = encode_target(df)

    X = df[FEATURES]
    y = df["Stress_Level"]

    model, X_test, y_test = train_model(X, y)

    # ------------------ EDA ------------------
    if section == "Upload & EDA":
        st.dataframe(df.head())

    # ------------------ MODEL PERFORMANCE ------------------
    if section == "Model Performance":

        y_pred = model.predict(X_test)

        acc, report, cm = evaluate_model(y_test, y_pred)

        st.subheader("Model Performance")
        st.write(f"Accuracy: {round(acc,2)}")
        st.text(report)

        plot_confusion_matrix(cm)

        y_score = model.predict_proba(X_test)
        plot_roc(y_test, y_score, np.unique(y))

    # ------------------ SINGLE PREDICTION ------------------
    if section == "Prediction":

        study = st.number_input("Study Hours", 0.0, 24.0, 5.0)
        sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
        activity = st.number_input("Physical Activity", 0.0, 10.0, 1.0)

        if st.button("Predict"):
            pred = model.predict([[study, sleep, activity]])
            result = le.inverse_transform(pred)[0]

            st.success(f"Predicted Stress: {result}")

    # ------------------ BATCH TEST ------------------
    if section == "Batch Testing":

        test_file = st.file_uploader("Upload Excel", type=["xlsx"])

        if test_file:

            new_df = pd.read_excel(test_file)
            new_df = fix_test_columns(new_df)

            missing = [c for c in FEATURES if c not in new_df.columns]

            if missing:
                st.error(f"Missing columns: {missing}")
                st.stop()

            X_new = new_df[FEATURES]

            preds = model.predict(X_new)
            new_df["Predicted_Stress"] = le.inverse_transform(preds)

            st.subheader("Predictions")
            st.dataframe(new_df.head())

            # -------- METRICS --------
            if "Stress_Level" in new_df.columns:

                try:
                    y_true = le.transform(new_df["Stress_Level"])

                    acc, report, cm = evaluate_model(y_true, preds)

                    st.subheader("Batch Performance")
                    st.write(f"Accuracy: {round(acc,2)}")
                    st.text(report)

                    plot_confusion_matrix(cm)

                    y_score = model.predict_proba(X_new)
                    plot_roc(y_true, y_score, np.unique(y))

                except Exception as e:
                    st.error(e)

            # Download
            output = BytesIO()
            new_df.to_excel(output, index=False)
            output.seek(0)

            st.download_button("Download Results", output, "results.xlsx")

else:
    st.info("Upload dataset to start")
