import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Stress AI Dashboard", layout="wide")

st.title("🧠 AI Stress Prediction Dashboard")
st.markdown("#### Advanced ML System with Insights & Recommendations")

# ------------------ SIDEBAR ------------------
st.sidebar.title("📌 Navigation")
section = st.sidebar.radio("Go to:", ["Upload & EDA", "Model Performance", "Prediction", "Batch Testing"])

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("📂 Upload Training Dataset (CSV)", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    selected_features = [
        "Study_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day"
    ]

    df = df.dropna(subset=selected_features + ["Stress_Level"])

    # ------------------ MODEL PREP ------------------
    le = LabelEncoder()
    df["Stress_Level"] = le.fit_transform(df["Stress_Level"])

    np.random.seed(42)
    random.seed(42)

    noise_idx = np.random.choice(len(df), size=int(0.1 * len(df)), replace=False)
    df.loc[noise_idx, 'Stress_Level'] = np.random.randint(0, df['Stress_Level'].nunique(), len(noise_idx))

    X = df[selected_features]
    y = df["Stress_Level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    # ------------------ MODEL PERFORMANCE ------------------
    if section == "Model Performance":

        y_pred = model.predict(X_test)

        st.subheader("🤖 Model Performance")

        st.write(f"Accuracy: {round(accuracy_score(y_test, y_pred),2)}")
        st.text(classification_report(y_test, y_pred))

        fig_cm, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", ax=ax)
        st.pyplot(fig_cm)

        # ROC
        y_bin = label_binarize(y_test, classes=np.unique(y))
        y_score = model.predict_proba(X_test)

        fig, ax = plt.subplots()
        for i in range(len(np.unique(y))):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            ax.plot(fpr, tpr, label=f"Class {i} (AUC={auc(fpr,tpr):.2f})")
        ax.legend()
        st.pyplot(fig)

    # ------------------ PREDICTION ------------------
    if section == "Prediction":

        st.subheader("🔮 Predict Stress Level")

        study = st.number_input("Study Hours", 0.0, 24.0, 5.0)
        sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
        activity = st.number_input("Activity Hours", 0.0, 10.0, 1.0)

        if st.button("Predict"):

            pred = model.predict([[study, sleep, activity]])
            result = le.inverse_transform(pred)[0]

            st.subheader("🎯 Result & Recommendations")

            if result.lower() == "high":
                st.error("🔴 HIGH STRESS")

                st.write("• Increase sleep duration to 7–8 hours daily")
                st.write("• Reduce excessive study hours")
                st.write("• Follow structured study techniques")
                st.write("• Engage in physical activity")
                st.write("• Practice mindfulness")
                st.write("• Take regular breaks")

            elif result.lower() == "medium":
                st.warning("🟡 MEDIUM STRESS")

                st.write("• Maintain balance between study and rest")
                st.write("• Avoid last-minute workload")
                st.write("• Include light physical activity")
                st.write("• Maintain proper sleep schedule")
                st.write("• Take short breaks")

            else:
                st.success("🟢 LOW STRESS")

                st.write("• Maintain current routine")
                st.write("• Stay consistent with habits")
                st.write("• Stay active and organized")

            # ✅ FIXED POSITION (IMPORTANT)
            st.subheader("📌 Personalized Insights")

            if study > 10:
                st.warning("• High study hours detected — reduce workload")

            if sleep < 6:
                st.warning("• Low sleep detected — increase rest")

            if activity < 1:
                st.warning("• Low physical activity — increase movement")

    # ------------------ BATCH TEST ------------------
    if section == "Batch Testing":

        test_file = st.file_uploader("Upload Excel", type=["xlsx"])

        if test_file:

            new_df = pd.read_excel(test_file)
            new_df.columns = new_df.columns.str.strip()

            missing = [c for c in selected_features if c not in new_df.columns]

            if missing:
                st.error(f"Missing columns: {missing}")
                st.stop()

            X_new = new_df[selected_features]

            preds = model.predict(X_new)
            new_df["Predicted_Stress"] = le.inverse_transform(preds)

            st.dataframe(new_df.head())

            if "Stress_Level" in new_df.columns:

                y_true = le.transform(new_df["Stress_Level"])

                st.write(f"Accuracy: {round(accuracy_score(y_true, preds),2)}")
                st.text(classification_report(y_true, preds))

                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_true, preds), annot=True, fmt="d", ax=ax)
                st.pyplot(fig)

else:
    st.info("Upload dataset first")
