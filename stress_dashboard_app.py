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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Stress AI Dashboard", layout="wide")

st.title("🧠 Advanced Stress Prediction Dashboard")
st.markdown("### Production-Level ML with SMOTE + Noise + RF")

# ------------------ UPLOAD ------------------
uploaded_file = st.file_uploader("📂 Upload Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ------------------ SELECT FEATURES ------------------
    selected_features = [
        "Study_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day"
    ]

    # ------------------ CLEAN ------------------
    df = df.dropna(subset=selected_features + ["Stress_Level"])

    # ------------------ EDA ------------------
    st.subheader("📊 Exploratory Data Analysis")

    st.markdown("### 📌 Summary Statistics")
    st.dataframe(df.describe())

    st.markdown("### 📈 Feature Distributions")
    col1, col2, col3 = st.columns(3)

    fig1, ax1 = plt.subplots(figsize=(3,3))
    sns.histplot(df[selected_features[0]], kde=True, ax=ax1)
    ax1.set_title("Study Hours")
    col1.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(3,3))
    sns.histplot(df[selected_features[1]], kde=True, ax=ax2)
    ax2.set_title("Sleep Hours")
    col2.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(3,3))
    sns.histplot(df[selected_features[2]], kde=True, ax=ax3)
    ax3.set_title("Activity")
    col3.pyplot(fig3)

    st.markdown("### 📊 Stress Distribution")
    fig4, ax4 = plt.subplots(figsize=(4,3))
    sns.countplot(x=df["Stress_Level"], ax=ax4)
    st.pyplot(fig4)

    st.markdown("### 📦 Feature vs Stress")
    c1, c2, c3 = st.columns(3)

    fig5, ax5 = plt.subplots(figsize=(3,3))
    sns.boxplot(x=df["Stress_Level"], y=df[selected_features[0]], ax=ax5)
    c1.pyplot(fig5)

    fig6, ax6 = plt.subplots(figsize=(3,3))
    sns.boxplot(x=df["Stress_Level"], y=df[selected_features[1]], ax=ax6)
    c2.pyplot(fig6)

    fig7, ax7 = plt.subplots(figsize=(3,3))
    sns.boxplot(x=df["Stress_Level"], y=df[selected_features[2]], ax=ax7)
    c3.pyplot(fig7)

    # ------------------ ENCODE ------------------
    le = LabelEncoder()
    df["Stress_Level"] = le.fit_transform(df["Stress_Level"])

    # ------------------ ADD NOISE ------------------
    np.random.seed(42)
    random.seed(42)

    noise_idx = np.random.choice(len(df), size=int(0.1 * len(df)), replace=False)
    df.loc[noise_idx, 'Stress_Level'] = np.random.randint(
        0, df['Stress_Level'].nunique(), len(noise_idx)
    )

    # ------------------ SPLIT ------------------
    X = df[selected_features]
    y = df["Stress_Level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------ SMOTE ------------------
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    # ------------------ CLASS WEIGHTS ------------------
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # ------------------ MODEL ------------------
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )

    model.fit(X_train, y_train)

    # ------------------ PERFORMANCE ------------------
    st.subheader("🤖 Model Performance")

    y_pred = model.predict(X_test)

    st.write(f"Accuracy: {round(accuracy_score(y_test, y_pred),3)}")
    st.text(classification_report(y_test, y_pred))

    # ------------------ CONFUSION MATRIX ------------------
    st.subheader("📊 Confusion Matrix")

    fig_cm, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig_cm)

    # ------------------ ROC ------------------
    st.subheader("📈 ROC Curve")

    y_test_bin = label_binarize(y_test, classes=[0,1,2])
    y_score = model.predict_proba(X_test)

    fig_roc, ax = plt.subplots(figsize=(4,3))
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, label=f"Class {i}")

    ax.legend()
    st.pyplot(fig_roc)

    # ------------------ SPEARMAN ------------------
    st.subheader("📊 Spearman Correlation")

    corr = df[selected_features + ["Stress_Level"]].corr(method="spearman")

    fig_corr, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)

    # ------------------ DOWNLOAD ------------------
    def save_fig(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return buf

    st.download_button("Download Heatmap", save_fig(fig_corr), "heatmap.png")

# ------------------ PREDICTION ------------------
st.subheader("🔮 Predict Stress")

c1, c2, c3 = st.columns(3)

study = c1.number_input("Study Hours", 0.0, 24.0, 5.0)
sleep = c2.number_input("Sleep Hours", 0.0, 24.0, 7.0)
activity = c3.number_input("Activity Hours", 0.0, 10.0, 1.0)

total = study + sleep + activity
st.write(f"⏱ Total Time Used: {total} hrs")

if total > 24:
    st.error("⚠️ Total exceeds 24 hours!")
else:
    st.info(f"Remaining Free Time: {24-total} hrs")

if st.button("Predict"):

    pred = model.predict([[study, sleep, activity]])
    result = le.inverse_transform(pred)[0]

    st.subheader("🎯 Result & Recommendations")

    # ------------------ HIGH ------------------
    if result.lower() == "high":
        st.error("🔴 HIGH STRESS")

        st.markdown("### 💡 Recommended Actions:")
        st.write("1. Increase sleep to **7–8 hours daily**")
        st.write("2. Reduce continuous study sessions (Pomodoro technique)")
        st.write("3. Add **30–45 mins physical activity**")
        st.write("4. Practice meditation or breathing exercises")
        st.write("5. Prioritize tasks and avoid overload")

    # ------------------ MEDIUM ------------------
    elif result.lower() == "medium":
        st.warning("🟡 MEDIUM STRESS")

        st.markdown("### 💡 Recommended Actions:")
        st.write("1. Maintain balanced study schedule")
        st.write("2. Keep consistent sleep routine (6–8 hrs)")
        st.write("3. Include light exercise (walking/stretching)")
        st.write("4. Plan tasks to avoid last-minute stress")

    # ------------------ LOW ------------------
    else:
        st.success("🟢 LOW STRESS")

        st.markdown("### 💡 Recommended Actions:")
        st.write("1. Continue current healthy routine")
        st.write("2. Maintain sleep and activity levels")
        st.write("3. Stay socially and mentally active")

    # ------------------ PERSONALIZED INSIGHTS ------------------
    st.markdown("### 📌 Personalized Tips")

    if sleep < 6:
        st.write("👉 You are sleeping less — increase sleep")

    if study > 10:
        st.write("👉 Study hours are high — reduce overload")

    if activity < 1:
        st.write("👉 Add physical activity to reduce stress")

    # ------------------ BATCH ------------------
    st.subheader("📂 Batch Prediction")

    test_file = st.file_uploader("Upload Excel", type=["xlsx"])

    if test_file:
        new_df = pd.read_excel(test_file, header=None)

        new_df.columns = [
            "Study_Hours_Per_Day",
            "Extracurricular_Hours_Per_Day",
            "Sleep_Hours_Per_Day",
            "Social_Hours_Per_Day",
            "Physical_Activity_Hours_Per_Day",
            "Stress_Level"
        ]

        preds = model.predict(new_df[selected_features])
        new_df["Predicted"] = le.inverse_transform(preds)

        st.dataframe(new_df.head())

        output = BytesIO()
        new_df.to_excel(output, index=False)
        output.seek(0)

        st.download_button("Download Results", output, "predictions.xlsx")
