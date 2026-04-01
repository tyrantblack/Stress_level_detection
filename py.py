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
st.markdown("⚡ Model uses ONLY 3 features: Study, Sleep, Physical Activity")

# -----------------------------
# REQUIRED FEATURES
# -----------------------------
FEATURES = [
    "Study_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day"
]

TARGET = "Stress_Level"

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload Training Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # -----------------------------
    # VALIDATION
    # -----------------------------
    required_cols = FEATURES + [TARGET]

    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Dataset must contain: {required_cols}")
        st.stop()

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # Sidebar
    menu = st.sidebar.radio(
        "Navigation",
        ["EDA", "Model Training", "Prediction", "Batch Prediction"]
    )

    # =====================================================
    # EDA
    # =====================================================
    if menu == "EDA":

        st.header("📈 Exploratory Data Analysis (3 Features Only)")

        eda_df = df[FEATURES + [TARGET]]

        # UNIVARIATE
        st.subheader("🔹 Univariate Analysis")
        col = st.selectbox("Select Feature", eda_df.columns)

        fig, ax = plt.subplots()
        sns.histplot(eda_df[col], kde=True, ax=ax)
        st.pyplot(fig)

        # BIVARIATE
        st.subheader("🔹 Bivariate Analysis")
        col1 = st.selectbox("X-axis", FEATURES, key="x")
        col2 = st.selectbox("Y-axis", FEATURES, key="y")

        fig, ax = plt.subplots()
        sns.scatterplot(x=eda_df[col1], y=eda_df[col2], ax=ax)
        st.pyplot(fig)

        # CORRELATION
        st.subheader("🔹 Pearson Correlation")
        corr = eda_df.corr(numeric_only=True)

        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # SPEARMAN
        st.subheader("🔹 Spearman Correlation")
        spearman = eda_df.corr(method='spearman', numeric_only=True)

        fig, ax = plt.subplots()
        sns.heatmap(spearman, annot=True, cmap='viridis', ax=ax)
        st.pyplot(fig)

    # =====================================================
    # MODEL TRAINING
    # =====================================================
    elif menu == "Model Training":

        st.header("🤖 Model Training (3 Features)")

        df_model = df[FEATURES + [TARGET]].copy()

        # Encode target
        le = LabelEncoder()
        df_model[TARGET] = le.fit_transform(df_model[TARGET])

        X = df_model[FEATURES]
        y = df_model[TARGET]

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

        # TEST
        y_pred = model.predict(X_test)

        st.subheader("📊 Performance")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.text(classification_report(y_test, y_pred))

        # FEATURE IMPORTANCE
        st.subheader("📌 Feature Importance")
        fig, ax = plt.subplots()
        sns.barplot(x=model.feature_importances_, y=FEATURES, ax=ax)
        st.pyplot(fig)

        # Save
        st.session_state['model'] = model
        st.session_state['encoder'] = le

    # =====================================================
    # PREDICTION
    # =====================================================
    elif menu == "Prediction":

        st.header("🔮 Real-Time Prediction")

        if 'model' not in st.session_state:
            st.warning("⚠️ Train model first")
        else:
            model = st.session_state['model']
            le = st.session_state['encoder']

            study = st.number_input("Study Hours", 0.0, 24.0, 5.0)
            sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
            activity = st.number_input("Physical Activity Hours", 0.0, 10.0, 1.0)

            if study + sleep + activity > 24:
                st.error("⚠️ Total hours exceed 24")
            else:
                input_data = np.array([[study, sleep, activity]], dtype=float)

                pred = model.predict(input_data)
                result = le.inverse_transform(pred)[0]

                st.success(f"Predicted Stress Level: {result}")

    # =====================================================
    # BATCH PREDICTION
    # =====================================================
    elif menu == "Batch Prediction":

        st.header("📂 Batch Prediction")

        if 'model' not in st.session_state:
            st.warning("⚠️ Train model first")
        else:
            model = st.session_state['model']
            le = st.session_state['encoder']

            file = st.file_uploader("Upload Excel", type=["xlsx"])

            if file is not None:
                new_df = pd.read_excel(file)

                # VALIDATE INPUT FILE
                if not all(col in new_df.columns for col in FEATURES):
                    st.error(f"❌ Excel must contain: {FEATURES}")
                    st.stop()

                X_new = new_df[FEATURES]

                preds = model.predict(X_new)
                new_df['Predicted_Stress'] = le.inverse_transform(preds)

                st.write(new_df.head())

                new_df.to_excel("predictions_output.xlsx", index=False)

                st.success("✅ File saved")
