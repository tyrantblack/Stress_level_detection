import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Stress Prediction Dashboard", layout="wide")

st.title("🧠 AI Stress Prediction Dashboard")
st.markdown("### Professional Student Stress Analysis System")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ------------------ FEATURE SELECTION ------------------
    selected_features = [
        "Study_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day"
    ]

    # ------------------ METRICS ------------------
    st.subheader("📌 Key Insights")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Study Hours", round(df[selected_features[0]].mean(), 2))
    col2.metric("Avg Sleep Hours", round(df[selected_features[1]].mean(), 2))
    col3.metric("Avg Activity", round(df[selected_features[2]].mean(), 2))

    st.metric("Most Common Stress Level", df["Stress_Level"].mode()[0])

    # ------------------ SPEARMAN CORRELATION ------------------
    st.subheader("📊 Spearman Correlation Analysis")

    corr = df[selected_features + ["Stress_Level"]].corr(method="spearman")

    fig_corr, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)

    # ------------------ IMAGE DOWNLOAD ------------------
    def save_fig(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        return buf

    st.download_button(
        label="📥 Download Correlation Heatmap",
        data=save_fig(fig_corr),
        file_name="spearman_heatmap.png",
        mime="image/png"
    )

    # ------------------ BOX PLOTS ------------------
    st.subheader("📈 Feature vs Stress Analysis")

    fig_box, ax = plt.subplots(1, 3, figsize=(15, 5))

    sns.boxplot(x="Stress_Level", y=selected_features[0], data=df, ax=ax[0])
    ax[0].set_title("Study vs Stress")

    sns.boxplot(x="Stress_Level", y=selected_features[1], data=df, ax=ax[1])
    ax[1].set_title("Sleep vs Stress")

    sns.boxplot(x="Stress_Level", y=selected_features[2], data=df, ax=ax[2])
    ax[2].set_title("Activity vs Stress")

    st.pyplot(fig_box)

    st.download_button(
        label="📥 Download Boxplots",
        data=save_fig(fig_box),
        file_name="boxplots.png",
        mime="image/png"
    )

    # ------------------ MODEL ------------------
    le = LabelEncoder()
    df["Stress_Level"] = le.fit_transform(df["Stress_Level"])

    X = df[selected_features]
    y = df["Stress_Level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ------------------ PERFORMANCE ------------------
    st.subheader("🤖 Model Performance")

    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {round(acc, 2)}")

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # ------------------ PREDICTION UI ------------------
    st.subheader("🔮 Predict Stress Level")

    col1, col2, col3 = st.columns(3)

    study = col1.number_input("Study Hours", 0.0, 24.0, 5.0)
    sleep = col2.number_input("Sleep Hours", 0.0, 24.0, 7.0)
    activity = col3.number_input("Physical Activity Hours", 0.0, 10.0, 1.0)

    total = study + sleep + activity
    st.write(f"⏱ Total Time Used: {total} hrs")

    if total > 24:
        st.error("⚠️ Total exceeds 24 hours!")
    else:
        st.info(f"Remaining Free Time: {24 - total} hrs")

    # ------------------ PREDICT BUTTON ------------------
    if st.button("🚀 Predict Stress"):

        input_data = np.array([[study, sleep, activity]])
        pred = model.predict(input_data)
        result = le.inverse_transform(pred)[0]

        st.subheader("🎯 Prediction Result")

        # ------------------ PROFESSIONAL SUGGESTIONS ------------------
        if result.lower() == "high":
            st.error("🔴 HIGH STRESS")

            st.markdown("### 💡 Recommendations:")
            st.write("• Increase sleep to at least 7–8 hours")
            st.write("• Reduce continuous study sessions (use Pomodoro)")
            st.write("• Include daily physical activity (≥30 mins)")
            st.write("• Practice mindfulness or breathing exercises")
            st.write("• Take regular breaks to avoid burnout")

        elif result.lower() == "medium":
            st.warning("🟡 MEDIUM STRESS")

            st.markdown("### 💡 Recommendations:")
            st.write("• Maintain balanced study schedule")
            st.write("• Ensure consistent sleep routine")
            st.write("• Add light exercise or walking")
            st.write("• Avoid last-minute workload spikes")

        else:
            st.success("🟢 LOW STRESS")

            st.markdown("### 💡 Recommendations:")
            st.write("• Maintain your current routine")
            st.write("• Keep consistent sleep and activity levels")
            st.write("• Continue stress management habits")
            st.write("• Stay socially engaged and active")

        # ------------------ FEATURE IMPORTANCE ------------------
        st.subheader("📊 Feature Importance")

        importance = model.feature_importances_
        features = selected_features

        fig_imp, ax = plt.subplots()
        sns.barplot(x=importance, y=features, ax=ax)
        st.pyplot(fig_imp)

        st.download_button(
            label="📥 Download Feature Importance",
            data=save_fig(fig_imp),
            file_name="feature_importance.png",
            mime="image/png"
        )

    # ------------------ DISTRIBUTION ------------------
    st.subheader("📂 Dataset Insights")

    if st.checkbox("Show Stress Distribution"):

        dist = df["Stress_Level"].value_counts()

        fig_dist, ax = plt.subplots()
        dist.plot(kind="bar", ax=ax)

        st.pyplot(fig_dist)

        st.download_button(
            label="📥 Download Distribution",
            data=save_fig(fig_dist),
            file_name="distribution.png",
            mime="image/png"
        )

        st.write(dist)

    # ------------------ FINAL SUMMARY ------------------
    st.subheader("📄 Summary Report")

    st.write("✔ Spearman correlation used for robust relationship analysis")
    st.write("✔ Sleep strongly impacts stress reduction")
    st.write("✔ High study hours increase stress risk")
    st.write("✔ Physical activity improves mental well-being")
    st.write("✔ Balanced routine leads to optimal performance")
