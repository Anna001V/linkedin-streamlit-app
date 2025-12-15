import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(page_title="LinkedIn Usage Predictor", layout="wide")


lr = joblib.load("linkedin_logreg.joblib")
features = joblib.load("model_features.joblib")


@st.cache_data
def load_clean_data(csv_path="social_media_usage.csv"):
    s = pd.read_csv(csv_path)

    def clean_sm(x):
        return np.where(x == 1, 1, 0)

    ss = pd.DataFrame({
        "linkedin_user": clean_sm(s["web1h"]),
        "income": s["income"],
        "education": s["educ2"],
        "parent": clean_sm(s["par"]),
        "married": np.where(s["marital"] == 1, 1, 0),
        "female": np.where(s["gender"] == 2, 1, 0),
        "age": s["age"]
    })

    ss["income"] = ss["income"].where(ss["income"] <= 9, np.nan)
    ss["education"] = ss["education"].where(ss["education"] <= 8, np.nan)
    ss["age"] = ss["age"].where(ss["age"] <= 98, np.nan)

    return ss.dropna()

try:
    ss = load_clean_data()
except Exception:
    ss = None


st.title("LinkedIn Usage Predictor")
st.caption("An interactive dashboard that predicts LinkedIn usage to support data-driven marketing decisions.")


st.sidebar.header("User Characteristics")

income = st.sidebar.slider("Income (1–9)", 1, 9, 5)
education = st.sidebar.slider("Education Level (1–8)", 1, 8, 4)
age = st.sidebar.slider("Age", 0, 98, 35)

parent = int(st.sidebar.toggle("Parent", False))
married = int(st.sidebar.toggle("Married", False))


female = 1



eda_gender = st.sidebar.selectbox("Gender", ["Female", "Male"])



X_input = pd.DataFrame([{
    "income": income,
    "education": education,
    "parent": parent,
    "married": married,
    "female": female,
    "age": age
}])[features]

prob = lr.predict_proba(X_input)[0, 1]
pred_class = int(prob >= 0.5)


tab_pred, tab_perf, tab_eda = st.tabs(
    ["Prediction", "Model Performance", "Exploratory Data Analysis"]
)


with tab_pred:
    st.subheader("Prediction Result")

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Predicted LinkedIn User", "Yes" if pred_class else "No")
    c2.metric("Probability of LinkedIn Usage", f"{prob:.3f}")
    c3.dataframe(X_input, use_container_width=True)

    st.progress(int(prob * 100))

    st.subheader("Probability Breakdown")
    st.bar_chart(pd.DataFrame(
        {"Probability": [prob, 1 - prob]},
        index=["Uses LinkedIn", "Does Not Use LinkedIn"]
    ))

    st.subheader("Effect of Age on Predicted LinkedIn Usage")
    ages = list(range(18, 99))


    age_df = pd.DataFrame({
        "income": income,
        "education": education,
        "parent": parent,
        "married": married,
        "female": female,
        "age": ages
    })[features]

    age_probs = lr.predict_proba(age_df)[:, 1]

    st.line_chart(pd.DataFrame({
        "Age": ages,
        "Predicted Probability": age_probs
    }).set_index("Age"))

    st.subheader("Model Coefficients (Interpretability)")
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": lr.coef_[0]
    }).sort_values("Coefficient", ascending=False)
    st.dataframe(coef_df, use_container_width=True)


with tab_perf:
    st.subheader("Model Performance Metrics")

    if ss is None:
        st.warning("Dataset not found. Place social_media_usage.csv in the app folder.")
    else:
        X_all = ss[features]
        y_all = ss["linkedin_user"]

        y_pred = lr.predict(X_all)
        y_prob = lr.predict_proba(X_all)[:, 1]

        acc = accuracy_score(y_all, y_pred)
        prec = precision_score(y_all, y_pred, zero_division=0)
        rec = recall_score(y_all, y_pred, zero_division=0)
        f1 = f1_score(y_all, y_pred, zero_division=0)
        auc = roc_auc_score(y_all, y_prob)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("ROC AUC", f"{auc:.3f}")
        m3.metric("Precision", f"{prec:.3f}")
        m4.metric("Recall", f"{rec:.3f}")
        m5.metric("F1 Score", f"{f1:.3f}")

        fpr, tpr, _ = roc_curve(y_all, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)

        cm = confusion_matrix(y_all, y_pred)
        st.subheader("Confusion Matrix")
        st.dataframe(pd.DataFrame(
            cm,
            index=["Actual: No", "Actual: Yes"],
            columns=["Predicted: No", "Predicted: Yes"]
        ))


with tab_eda:
    st.subheader("Exploratory Data Analysis")

    if ss is None:
        st.warning("Dataset not available.")
    else:
        if eda_gender == "Female":
            eda_df = ss[ss["female"] == 1]
        elif eda_gender == "Male":
            eda_df = ss[ss["female"] == 0]
        else:
            eda_df = ss

        st.write(f"Observations after filters: **{eda_df.shape[0]}**")

        st.dataframe(eda_df.describe().T, use_container_width=True)

        st.subheader("LinkedIn Usage by Income")
        st.plotly_chart(
            px.bar(
                eda_df.groupby("income")["linkedin_user"].mean().reset_index(),
                x="income", y="linkedin_user"
            ),
            use_container_width=True
        )

        st.subheader("LinkedIn Usage by Education")
        st.plotly_chart(
            px.bar(
                eda_df.groupby("education")["linkedin_user"].mean().reset_index(),
                x="education", y="linkedin_user"
            ),
            use_container_width=True
        )

        st.subheader("Income × Education Heatmap")
        heat = eda_df.pivot_table(
            values="linkedin_user", index="education", columns="income", aggfunc="mean"
        )
        st.plotly_chart(
            px.imshow(heat, aspect="auto", labels=dict(color="LinkedIn Usage")),
            use_container_width=True
        )
