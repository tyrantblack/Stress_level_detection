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
section = st.sidebar.radio("Go to:", ["Upload & EDA", "Model Performance", "Prediction", "Batch Testing","User History" ])

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("📂 Upload Training Dataset (CSV)", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    selected_features = [
        "Study_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day"
    ]

    # Drop missing values
    df = df.dropna(subset=selected_features + ["Stress_Level"])

    # ------------------ EDA ------------------
    if section == "Upload & EDA":

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("📌 Summary Statistics")
        st.dataframe(df.describe())

        st.subheader("📈 Feature Distributions")

        col1, col2, col3 = st.columns(3)

        for i, col in enumerate(selected_features):
            fig, ax = plt.subplots(figsize=(3,3))
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(col.replace("_", " "))
            [col1, col2, col3][i].pyplot(fig)

        st.subheader("📦 Feature vs Stress")

        c1, c2, c3 = st.columns(3)

        for i, col in enumerate(selected_features):
            fig, ax = plt.subplots(figsize=(3,3))
            sns.boxplot(x=df["Stress_Level"], y=df[col], ax=ax)
            ax.set_title(col.replace("_", " "))
            [c1, c2, c3][i].pyplot(fig)

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

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )

    model.fit(X_train, y_train)

    # 🔥 ADDITION 1: FEATURE IMPORTANCE (SAFE ADD)
    st.sidebar.subheader("📊 Feature Importance")
    importance = pd.DataFrame({
        "Feature": selected_features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    st.sidebar.dataframe(importance)

    # ------------------ MODEL PERFORMANCE ------------------
    if section == "Model Performance":

        st.subheader("🤖 Model Performance")

        y_pred = model.predict(X_test)

        # 🔥 ADDITION 2: KPI CARDS
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 2))
        c2.metric("Train Size", len(X_train))
        c3.metric("Test Size", len(X_test))

        st.text(classification_report(y_test, y_pred))

        fig_cm, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig_cm)

        # ROC
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        y_score = model.predict_proba(X_test)

        fig_roc, ax = plt.subplots(figsize=(4,3))
        for i in range(len(np.unique(y))):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")
        ax.legend()
        st.pyplot(fig_roc)

    # ------------------ PREDICTION ------------------
    if section == "Prediction":

        st.subheader("🔮 Predict Stress Level")

        c1, c2, c3 = st.columns(3)

        study = c1.number_input("Study Hours", 0.0, 24.0, 5.0)
        sleep = c2.number_input("Sleep Hours", 0.0, 24.0, 7.0)
        activity = c3.number_input("Activity Hours", 0.0, 10.0, 1.0)

        if st.button("Predict"):

            pred = model.predict([[study, sleep, activity]])
            result = le.inverse_transform(pred)[0]

            st.subheader("🎯 Result & Recommendations")

            if result.lower() == "high":
                st.error("🔴 HIGH STRESS")
            
                st.write("• Increase sleep duration to 7–8 hours daily for recovery")
                st.write("• Reduce excessive study hours and avoid continuous long sessions")
                st.write("• Follow structured study techniques (e.g., Pomodoro method)")
                st.write("• Engage in at least 30 minutes of physical activity daily")
                st.write("• Practice mindfulness, meditation, or deep breathing exercises")
                st.write("• Take regular breaks and limit screen exposure")
                st.write("• Prioritize important tasks and avoid multitasking overload")

            elif result.lower() == "medium":
                st.warning("🟡 MEDIUM STRESS")
            
                st.write("• Maintain a balanced routine between study, sleep, and activity")
                st.write("• Avoid last-minute workload and plan tasks in advance")
                st.write("• Include light physical activities like walking or stretching")
                st.write("• Ensure consistent sleep schedule (6–8 hours)")
                st.write("• Take short breaks during study sessions")
                st.write("• Practice basic relaxation techniques when feeling overwhelmed")
            else:
                st.success("🟢 LOW STRESS")
            
                st.write("• Maintain your current healthy lifestyle and routine")
                st.write("• Continue consistent sleep and activity patterns")
                st.write("• Stay organized with your daily tasks")
                st.write("• Engage in hobbies and recreational activities")
                st.write("• Monitor stress levels periodically to avoid sudden spikes")
                st.write("• Maintain social interaction and positive habits")
                        
            user_data = pd.DataFrame({
                "Study_Hours": [study],
                "Sleep_Hours": [sleep],
                "Activity_Hours": [activity],
                "Predicted_Stress": [result]
            })
        
            try:
                old_data = pd.read_csv("user_history.csv")
                updated_data = pd.concat([old_data, user_data], ignore_index=True)
            except:
                updated_data = user_data
        
            updated_data.to_csv("user_history.csv", index=False)
        
            st.success("✅ Prediction saved successfully")         
                
    # ------------------ BATCH TEST ------------------
    if section == "Batch Testing":

        st.subheader("📂 Upload Test Dataset")

        test_file = st.file_uploader("Upload Excel", type=["xlsx"])

        if test_file:

            new_df = pd.read_excel(test_file)

            st.write("🔍 Raw Columns Detected:")
            st.write(list(new_df.columns))

            new_df.columns = new_df.columns.str.strip()

            missing_cols = [col for col in selected_features if col not in new_df.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.stop()

            st.subheader("📊 Cleaned Dataset Preview")
            st.dataframe(new_df.head())

            X_new = new_df[selected_features]

            preds = model.predict(X_new)
            new_df["Predicted_Stress"] = le.inverse_transform(preds)

            st.subheader("📊 Predictions")
            st.dataframe(new_df.head())

            # 🔥 ADDITION 3: DISTRIBUTION
            st.subheader("📊 Prediction Distribution")
            st.bar_chart(new_df["Predicted_Stress"].value_counts())

            if "Stress_Level" in new_df.columns:

                try:
                    y_true = le.transform(new_df["Stress_Level"])

                    st.subheader("🤖 Test Dataset Performance")

                    acc = accuracy_score(y_true, preds)
                    st.write(f"Accuracy: {round(acc, 2)}")

                    st.text(classification_report(y_true, preds))

                    fig_cm, ax = plt.subplots()
                    sns.heatmap(confusion_matrix(y_true, preds), annot=True, fmt="d", cmap="Blues", ax=ax)
                    st.pyplot(fig_cm)

                    # 🔥 ADDITION 4: ROC FOR BATCH
                    y_bin = label_binarize(y_true, classes=np.unique(y))
                    y_score = model.predict_proba(X_new)

                    fig_roc, ax = plt.subplots()
                    for i in range(len(np.unique(y))):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")
                    ax.legend()
                    st.pyplot(fig_roc)

                except:
                    st.warning("⚠️ Stress_Level format mismatch. Cannot evaluate.")

            output = BytesIO()
            new_df.to_excel(output, index=False)
            output.seek(0)

            st.download_button("📥 Download Results", output, "results.xlsx")

    # 🔥 ADDITION 5: SUMMARY
    st.sidebar.subheader("📌 Model Summary")
    st.sidebar.write(f"""
    Samples: {len(df)}
    Features: {selected_features}
    Model: Random Forest
    SMOTE Applied
    """)

else:
    st.info("📌 Please upload a training dataset to proceed.")



