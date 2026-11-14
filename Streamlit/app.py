
import streamlit as st
import pickle
import sys
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="AI Text Predictor",
    page_icon="🤖",
    layout="centered"
)

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "Jupyter" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "Jupyter" / "vectorizer.pkl"

# Load Model and Vectorizer
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer not found: {VECTORIZER_PATH}")

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


# App UI
st.title("🤖 AI Text Predictor")
st.write("Enter a message to check whether it is Spam or Ham.")

try:
    model, vectorizer = load_model()
    st.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()


# User Input
user_text = st.text_area(
    "Enter your message:",
    placeholder="Type your message here...",
    height=150
)

# Prediction
col1, col2 = st.columns(2)

with col1:
    predict_button = st.button("🔍 Predict", use_container_width=True)

with col2:
    clear_button = st.button("🧹 Clear", use_container_width=True)

if predict_button:
    if user_text.strip():

        # Convert text into numerical features
        transformed_text = vectorizer.transform([user_text])

        # Predict
        prediction = model.predict(transformed_text)[0]
        prediction = str(prediction).lower()

        st.subheader("Prediction Result")

        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE DETECTED!")

        elif prediction == "ham":
            st.success("✅ NOT SPAM - SAFE MESSAGE")

        else:
            st.info(f"Prediction: {prediction}")

        # Confidence Score
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transformed_text)[0]
            confidence = float(max(probabilities))

            st.write(f"Confidence: {confidence * 100:.2f}%")
            st.progress(confidence)

    else:
        st.warning("Please enter a message first.")

if clear_button:
    st.rerun()
