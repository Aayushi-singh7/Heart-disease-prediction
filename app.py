import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load and prepare data
@st.cache_data
def load_model():
    data = pd.read_csv("data.csv")
    X = data.drop(columns='target', axis=1)
    Y = data['target']
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=100))
    ])
    
    model.fit(X_train, Y_train)
    return model, X.columns.tolist()

# Load model and feature names
model, features = load_model()

# Streamlit UI
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("💓 Heart Disease Predictor")
st.write("Enter the following medical information to predict the risk of heart disease:")

# Input fields
user_input = []
for feature in features:
    value = st.number_input(f"{feature.replace('_', ' ').capitalize()}", format="%.3f")
    user_input.append(value)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input], columns=features)
    prediction = model.predict(input_df)

    if prediction[0] == 0:
        st.success("✅ The person is likely **healthy** (no heart disease).")
    else:
        st.error("⚠️ The person is likely **at risk** of heart disease.")

st.markdown("---")
st.caption("Built with 🧠 Logistic Regression | Powered by Streamlit")
