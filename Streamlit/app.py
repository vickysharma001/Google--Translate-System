
import pickle
import streamlit as st
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Text Predictor",
    page_icon="🤖",
    layout="centered"
)

# ---------------- MODEL LOADING ----------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "Jupyter" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "Jupyter" / "vectorizer.pkl"

model = None
cv = None
model_loaded = False

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        cv = pickle.load(file)

    model_loaded = True

except Exception as e:
    st.error(f"Model Loading Error: {e}")


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background:
    radial-gradient(circle at 50% 0%, #35216f 0%, transparent 45%),
    linear-gradient(135deg, #080817, #11112d, #050510);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 900;
    background: linear-gradient(90deg, #c084fc, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0px 0px 25px rgba(139,92,246,0.3);
    animation: glow 3s infinite alternate;
}

@keyframes glow {
    from { filter: drop-shadow(0 0 5px #8b5cf6); }
    to { filter: drop-shadow(0 0 18px #3b82f6); }
}

.subtitle {
    text-align: center;
    color: #b8b8dc;
    margin-bottom: 30px;
}

.card {
    background: linear-gradient(145deg,
    rgba(80,65,145,0.35),
    rgba(15,15,40,0.85));
    padding: 30px;
    border-radius: 22px;
    border: 1px solid rgba(160,130,255,0.4);
    box-shadow: 0 15px 40px rgba(0,0,0,0.5),
                inset 0 1px 15px rgba(255,255,255,0.05);
    transition: 0.4s;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 50px rgba(110,70,255,0.25);
}

.stTextArea textarea {
    background: #17172d !important;
    color: white !important;
    border: 1px solid #7655ff !important;
    border-radius: 14px !important;
}

.stButton button {
    width: 100%;
    background: linear-gradient(90deg,#7c3aed,#2563eb);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 12px;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 0 7px 22px rgba(100,70,255,0.4);
    transition: 0.3s;
}

.stButton button:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 12px 30px rgba(100,70,255,0.7);
    color: white;
}

.result-card {
    background: rgba(50,40,100,0.35);
    border: 1px solid rgba(160,130,255,0.5);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

.footer {
    text-align: center;
    color: #8585aa;
    margin-top: 40px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">🤖 AI TEXT PREDICTOR</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent NLP & Machine Learning System</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ---------------- MAIN APP ----------------
st.markdown("### ✨ Enter Your Message")

user = st.text_area(
    "Write something here",
    placeholder="Example: Congratulations! You won a free prize...",
    height=140
)

col1, col2 = st.columns(2)

with col1:
    predict = st.button("🚀 Predict Now", use_container_width=True)

with col2:
    clear = st.button("🗑️ Clear Text", use_container_width=True)

if clear:
    st.rerun()


# ---------------- PREDICTION ----------------
if predict:

    if not model_loaded:
        st.error("Please fix the model loading error first.")

    elif not user.strip():
        st.warning("Please enter some text!")

    else:
        try:
            data = cv.transform([user])
            output = model.predict(data)

            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")

            result = str(output[0]).lower()

            if result == "spam":
                st.error("🚨 SPAM MESSAGE DETECTED!")

            elif result == "ham":
                st.success("✅ NOT SPAM - SAFE MESSAGE")

            else:
                st.info(f"Prediction: {output[0]}")

            # Prediction Probability
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(data)[0]
                confidence = max(probabilities) * 100

                st.markdown("#### Prediction Probability")
                st.progress(float(max(probabilities)))
                st.write(
                    f"Probability of predicted class: {confidence:.2f}%"
                )

        except Exception as e:
            st.error(f"Prediction Error: {e}")


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    POWERED BY MACHINE LEARNING & NLP
    <br>
    Designed with 💜 | Streamlit Application
</div>
""", unsafe_allow_html=True)
