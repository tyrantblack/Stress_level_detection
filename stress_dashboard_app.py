
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Stress Prediction Dashboard", layout="wide")

st.title("🧠 Stress Prediction Dashboard")
st.markdown("### AI-based Student Stress Analysis System")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📌 Key Insights")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Avg Study Hours", round(df["Study_Hours_Per_Day"].mean(), 2))
    col2.metric("Avg Sleep Hours", round(df["Sleep_Hours_Per_Day"].mean(), 2))
    col3.metric("Avg Activity", round(df["Physical_Activity_Hours_Per_Day"].mean(), 2))
    col4.metric("Most Common Stress", df["Stress_Level"].mode()[0])

    st.subheader("📊 Correlation Analysis")

    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.markdown("### 📖 Insights:")
    st.write("- Sleep is usually negatively correlated with stress")
    st.write("- Study hours may increase stress levels")
    st.write("- Physical activity helps reduce stress")

    st.subheader("📈 Feature vs Stress Analysis")

    fig, ax = plt.subplots(1, 3, figsize=(15,5))

    sns.boxplot(x="Stress_Level", y="Study_Hours_Per_Day", data=df, ax=ax[0])
    ax[0].set_title("Study vs Stress")

    sns.boxplot(x="Stress_Level", y="Sleep_Hours_Per_Day", data=df, ax=ax[1])
    ax[1].set_title("Sleep vs Stress")

    sns.boxplot(x="Stress_Level", y="Physical_Activity_Hours_Per_Day", data=df, ax=ax[2])
    ax[2].set_title("Activity vs Stress")

    st.pyplot(fig)

    le = LabelEncoder()
    df["Stress_Level"] = le.fit_transform(df["Stress_Level"])

    X = df[[
        "Study_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day"
    ]]
    y = df["Stress_Level"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader("🤖 Model Performance")
    st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 2))

    st.subheader("🔮 Predict Stress Level")

    col1, col2, col3 = st.columns(3)

    study = col1.number_input("Study Hours", 0.0, 24.0, 5.0)
    sleep = col2.number_input("Sleep Hours", 0.0, 24.0, 7.0)
    activity = col3.number_input("Activity Hours", 0.0, 10.0, 1.0)

    total = study + sleep + activity
    st.write(f"⏱ Total Time Used: {total} hrs")

    if total > 24:
        st.error("⚠️ Total exceeds 24 hours!")
    else:
        remaining = 24 - total
        st.info(f"Remaining Free Time: {remaining} hrs")

    if st.button("Predict"):

        input_data = np.array([[study, sleep, activity]])
        pred = model.predict(input_data)
        result = le.inverse_transform(pred)[0]

        st.subheader("🎯 Prediction Result")

        if result.lower() == "high":
            st.error("🔴 High Stress")
        elif result.lower() == "medium":
            st.warning("🟡 Medium Stress")
        else:
            st.success("🟢 Low Stress")

        st.subheader("💡 Insights")

        if sleep < 6:
            st.write("👉 Increase sleep to 7–8 hours")

        if study > 10:
            st.write("👉 Reduce study overload")

        if activity < 1:
            st.write("👉 Add physical activity")

        st.subheader("📊 Feature Importance")

        importance = model.feature_importances_
        features = X.columns

        fig, ax = plt.subplots()
        sns.barplot(x=importance, y=features, ax=ax)
        st.pyplot(fig)

    st.subheader("📂 Batch Summary")

    if st.checkbox("Show Stress Distribution"):
        dist = df["Stress_Level"].value_counts()

        fig, ax = plt.subplots()
        dist.plot(kind="bar", ax=ax)
        st.pyplot(fig)

        st.write(dist)

    st.subheader("📄 Summary Report")

    st.write("✔ Sleep plays a key role in reducing stress")
    st.write("✔ High study hours increase stress risk")
    st.write("✔ Balanced routine leads to low stress")
