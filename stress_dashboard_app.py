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

    # ------------------ EDA SECTION ------------------
st.subheader("📊 Exploratory Data Analysis (EDA)")

# ---------- BASIC STATS ----------
st.markdown("### 📌 Summary Statistics")
st.dataframe(df.describe())

# ---------- SMALL STRUCTURED VISUALS ----------
st.markdown("### 📈 Feature Distributions")

col1, col2, col3 = st.columns(3)

# Study Hours
fig1, ax1 = plt.subplots(figsize=(3,3)) 
sns.histplot(df[selected_features[0]], kde=True, ax=ax1)
ax1.set_title("Study Hours")
col1.pyplot(fig1)

# Sleep Hours
fig2, ax2 = plt.subplots(figsize=(3,3))
sns.histplot(df[selected_features[1]], kde=True, ax=ax2)
ax2.set_title("Sleep Hours")
col2.pyplot(fig2)

# Activity
fig3, ax3 = plt.subplots(figsize=(3,3))
sns.histplot(df[selected_features[2]], kde=True, ax=ax3)
ax3.set_title("Physical Activity")
col3.pyplot(fig3)


# ---------- STRESS DISTRIBUTION ----------
st.markdown("### 📊 Stress Level Distribution")

fig4, ax4 = plt.subplots(figsize=(4,3))
sns.countplot(x=df["Stress_Level"], ax=ax4)
ax4.set_title("Stress Distribution")
st.pyplot(fig4)


# ---------- BOXPLOTS (STRUCTURED) ----------
st.markdown("### 📦 Feature vs Stress")

c1, c2, c3 = st.columns(3)

fig5, ax5 = plt.subplots(figsize=(3,3))
sns.boxplot(x=df["Stress_Level"], y=df[selected_features[0]], ax=ax5)
ax5.set_title("Study vs Stress")
c1.pyplot(fig5)

fig6, ax6 = plt.subplots(figsize=(3,3))
sns.boxplot(x=df["Stress_Level"], y=df[selected_features[1]], ax=ax6)
ax6.set_title("Sleep vs Stress")
c2.pyplot(fig6)

fig7, ax7 = plt.subplots(figsize=(3,3))
sns.boxplot(x=df["Stress_Level"], y=df[selected_features[2]], ax=ax7)
ax7.set_title("Activity vs Stress")
c3.pyplot(fig7)

    # ------------------ SELECT FEATURES ------------------
    selected_features = [
        "Study_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day"
    ]

    df = df.dropna(subset=selected_features + ["Stress_Level"])

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

    # ------------------ PREDICTION ------------------
    y_pred = model.predict(X_test)

    # ------------------ PERFORMANCE ------------------
    st.subheader("🤖 Model Performance")

    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {round(acc, 3)}")

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # ------------------ CONFUSION MATRIX ------------------
    st.subheader("📊 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig_cm)

    # ------------------ ROC CURVE ------------------
    st.subheader("📈 ROC Curve")

    y_test_bin = label_binarize(y_test, classes=[0,1,2])
    y_score = model.predict_proba(X_test)

    fig_roc, ax = plt.subplots()

    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'Class {i} (AUC={roc_auc:.2f})')

    ax.plot([0,1],[0,1],'--')
    ax.legend()
    st.pyplot(fig_roc)

    # ------------------ SPEARMAN ------------------
    st.subheader("📊 Spearman Correlation")

    df_corr = df.copy()
    corr = df_corr[selected_features + ["Stress_Level"]].corr(method="spearman")

    fig_corr, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)

    # ------------------ DOWNLOAD IMAGE ------------------
    def save_fig(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return buf

    st.download_button("📥 Download Heatmap", save_fig(fig_corr), "heatmap.png")

    # ------------------ PREDICT UI ------------------
    st.subheader("🔮 Predict Stress")

    c1, c2, c3 = st.columns(3)

    study = c1.number_input("Study Hours", 0.0, 24.0, 5.0)
    sleep = c2.number_input("Sleep Hours", 0.0, 24.0, 7.0)
    activity = c3.number_input("Activity Hours", 0.0, 10.0, 1.0)

    if st.button("Predict"):

        pred = model.predict([[study, sleep, activity]])
        result = le.inverse_transform(pred)[0]

        st.subheader("🎯 Result")

        if result.lower() == "high":
            st.error("🔴 HIGH STRESS")
            st.write("• Increase sleep (7–8 hrs)")
            st.write("• Reduce study overload")
            st.write("• Add exercise")
            st.write("• Practice mindfulness")

        elif result.lower() == "medium":
            st.warning("🟡 MEDIUM STRESS")
            st.write("• Maintain balance")
            st.write("• Avoid last-minute work")
            st.write("• Light physical activity")

        else:
            st.success("🟢 LOW STRESS")
            st.write("• Maintain routine")
            st.write("• Stay active and consistent")

    # ------------------ EXTERNAL FILE PREDICTION ------------------
    st.subheader("📂 Batch Prediction")

    test_file = st.file_uploader("Upload Excel for Prediction", type=["xlsx"])

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

        X_new = new_df[selected_features]

        preds = model.predict(X_new)
        new_df["Predicted_Stress"] = le.inverse_transform(preds)

        st.dataframe(new_df.head())

        output = BytesIO()
        new_df.to_excel(output, index=False)
        output.seek(0)

        st.download_button("📥 Download Predictions", output, "predictions.xlsx")
