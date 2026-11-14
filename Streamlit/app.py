
import streamlit as st
import pickle

# Load Model and Vectorizer
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

# Page Configuration
st.set_page_config(
    page_title="AI Text Predictor",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

.stApp {
    background: radial-gradient(circle at top,
    #25205c 0%, #10101f 45%, #050510 100%);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    text-shadow: 0px 0px 20px #8b5cf6;
}

.subtitle {
    text-align: center;
    color: #b9b9d5;
    font-size: 16px;
    margin-bottom: 35px;
}

.card {
    background: linear-gradient(145deg,
    rgba(70,60,120,0.45),
    rgba(20,20,45,0.85));
    padding: 30px;
    border-radius: 20px;
    border: 1px solid rgba(150,120,255,0.4);
    box-shadow: 0px 15px 35px rgba(0,0,0,0.5),
                0px 0px 20px rgba(120,80,255,0.15);
    transition: 0.4s;
}

.card:hover {
    transform: translateY(-6px) scale(1.01);
    box-shadow: 0px 20px 45px rgba(100,70,255,0.3);
}

.stTextArea textarea {
    background-color: #17172d;
    color: white;
    border: 1px solid #7655ff;
    border-radius: 12px;
}

.stButton button {
    width: 100%;
    background: linear-gradient(90deg,#6d28d9,#2563eb);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 0px 6px 20px rgba(100,70,255,0.4);
    transition: 0.3s;
}

.stButton button:hover {
    transform: translateY(-4px);
    box-shadow: 0px 10px 30px #6d28d9;
    color: white;
}

.footer {
    text-align: center;
    color: #8888aa;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">🤖 AI TEXT PREDICTOR</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent NLP Prediction System</div>',
    unsafe_allow_html=True
)

# Main Card
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("✨ Enter Your Text")

user = st.text_area(
    "Write something here...",
    placeholder="Type your message here...",
    height=130
)

predict = st.button("🚀 Predict Now")

st.markdown('</div>', unsafe_allow_html=True)

# Prediction
if predict:

    if user.strip() == "":
        st.warning("Please enter some text!")

    else:
        data = cv.transform([user]).toarray()
        output = model.predict(data)

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        st.success(f"Prediction: {output[0]}")

# Footer
st.markdown(
    '<div class="footer">Powered by Machine Learning & NLP | Made with ❤️</div>',
    unsafe_allow_html=True
)
