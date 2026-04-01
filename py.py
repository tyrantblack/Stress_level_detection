import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Stress Prediction System", layout="wide")

st.title("🧠 AI-Powered Stress Prediction System")
st.markdown("Built using Machine Learning + EDA + Streamlit")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload Training Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Overview")
    st.write(df.head())

    # -----------------------------
    # SIDEBAR NAVIGATION
    # -----------------------------
    menu = st.sidebar.radio(
        "Navigation",
        ["EDA", "Model Training", "Prediction", "Batch Prediction"]
    )

    # =====================================================
    # 1. EDA SECTION
    # =====================================================
    if menu == "EDA":

        st.header("📈 Exploratory Data Analysis")

        # UNIVARIATE
        st.subheader("🔹 Univariate Analysis")
        col = st.selectbox("Select Feature", df.columns)

        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

        # BIVARIATE
        st.subheader("🔹 Bivariate Analysis")
        col1 = st.selectbox("X-axis", df.columns, key="x")
        col2 = st.selectbox("Y-axis", df.columns, key="y")

        fig, ax = plt.subplots()
        sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
        st.pyplot(fig)

        # PEARSON
        st.subheader("🔹 Correlation (Pearson)")
        corr = df.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # SPEARMAN
        st.subheader("🔹 Spearman Correlation")
        spearman = df.corr(method='spearman', numeric_only=True)

        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(spearman, annot=True, cmap='viridis', ax=ax)
        st.pyplot(fig)

    # =====================================================
    # 2. MODEL TRAINING
    # =====================================================
    elif menu == "Model Training":

        st.header("🤖 Model Training")

        # Encode
        le = LabelEncoder()
        df['Stress_Level'] = le.fit_transform(df['Stress_Level'])

        X = df[[
            "Study_Hours_Per_Day",
            "Sleep_Hours_Per_Day",
            "Physical_Activity_Hours_Per_Day"
        ]]
        y = df['Stress_Level']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # SMOTE
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

        # MODEL
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        # PREDICTION
        y_pred = model.predict(X_test)

        st.subheader("📊 Performance Metrics")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.text(classification_report(y_test, y_pred))

        # FEATURE IMPORTANCE
        st.subheader("📌 Feature Importance")

        importance = model.feature_importances_
        features = X.columns

        fig, ax = plt.subplots()
        sns.barplot(x=importance, y=features, ax=ax)
        st.pyplot(fig)

        # Save model in session
        st.session_state['model'] = model
        st.session_state['encoder'] = le

    # =====================================================
    # 3. PREDICTION
    # =====================================================
    elif menu == "Prediction":

        st.header("🔮 Real-Time Stress Prediction")

        if 'model' not in st.session_state:
            st.warning("⚠️ Please train the model first")
        else:
            model = st.session_state['model']
            le = st.session_state['encoder']

            # INPUTS
            study = st.number_input("Study Hours", 0.0, 24.0, 5.0, step=0.5)
            sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0, step=0.5)
            activity = st.number_input("Physical Activity Hours", 0.0, 10.0, 1.0, step=0.5)

            # VALIDATION
            if study + sleep + activity > 24:
                st.error("⚠️ Total hours exceed 24")
            else:
                input_data = np.array([[study, sleep, activity]], dtype=float)
                pred = model.predict(input_data)
                result = le.inverse_transform(pred)[0]

                st.success(f"Predicted Stress Level: {result}")

                # SUGGESTIONS
                st.subheader("💡 Recommendations")

                if result.lower() == "high":
                    st.error("""
                    🔴 HIGH STRESS:
                    - Sleep at least 7–8 hours
                    - Reduce overload
                    - Practice meditation
                    - Exercise regularly
                    """)

                elif result.lower() == "medium":
                    st.warning("""
                    🟡 MEDIUM STRESS:
                    - Maintain balance
                    - Improve time management
                    - Take breaks
                    """)

                else:
                    st.success("""
                    🟢 LOW STRESS:
                    - Maintain current routine
                    - Stay consistent
                    """)

    # =====================================================
    # 4. BATCH PREDICTION
    # =====================================================
    elif menu == "Batch Prediction":

        st.header("📂 Batch Prediction (Excel Upload)")

        if 'model' not in st.session_state:
            st.warning("⚠️ Train model first")
        else:
            model = st.session_state['model']
            le = st.session_state['encoder']

            file = st.file_uploader("Upload Excel File", type=["xlsx"])

            if file is not None:
                new_df = pd.read_excel(file, header=None)

                new_df.columns = [
                    "Study_Hours_Per_Day",
                    "Extracurricular_Hours_Per_Day",
                    "Sleep_Hours_Per_Day",
                    "Social_Hours_Per_Day",
                    "Physical_Activity_Hours_Per_Day",
                    "Stress_Level"
                ]

                X_new = new_df[[
                    "Study_Hours_Per_Day",
                    "Sleep_Hours_Per_Day",
                    "Physical_Activity_Hours_Per_Day"
                ]]

                preds = model.predict(X_new)
                new_df['Predicted_Stress'] = le.inverse_transform(preds)

                st.write(new_df.head())

                new_df.to_excel("predictions_output.xlsx", index=False)

                st.success("✅ File saved as predictions_output.xlsx")