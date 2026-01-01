import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from datetime import datetime


# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "dermalscan_streamlit.keras"
IMG_SIZE = 224
CLASS_NAMES = ['Wrinkles', 'clear skin', 'dark spots', 'puffy eyes']
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

st.set_page_config(page_title="DermalScan", layout="centered")

# -----------------------------
# SESSION STATE INIT
# -----------------------------
for key in ["result", "uploaded_image", "uploaded_name"]:
    if key not in st.session_state:
        st.session_state[key] = None

import base64
def set_bg_image(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# SET BACKGROUND IMAGE
set_bg_image("hero3.png")


# -----------------------------
# UI THEME
# -----------------------------
st.markdown("""
<style>
header, footer { visibility: hidden; }
.block-container { max-width: 850px; padding-top: 2rem; }

.card {
    background: linear-gradient(135deg, #dcfce7, #ccfbf1);
    color: #064e3b;
    border-radius: 22px;
    padding: 28px;
    margin-bottom: 28px;
    box-shadow: 0 18px 40px rgba(0,0,0,0.15);
}

.title {
    text-align: center;
    font-size: 48px;
    font-weight: 900;
    color: #2e1065;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    color: #7f1d1d;
    margin-bottom: 35px;
}

.stButton button {
    width: 100%;
    background: linear-gradient(90deg, #7c3aed, #db2777);
    color: white;
    font-size: 18px;
    font-weight: 700;
    border-radius: 14px;
    padding: 10px;
}

.result-box {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    padding: 16px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
}
/* ========= HIDE STREAMLIT DEFAULT TEXT ========= */

/* Hide uploader label */
div[data-testid="stFileUploader"] label {
    display: none !important;
}

/* Hide uploaded file text */
div[data-testid="stFileUploaderFile"] span,
div[data-testid="stFileUploaderFile"] small {
    display: none !important;
}

/* Hide image caption */
figcaption {
    display: none !important;
}

/* Hide spinner text */
div[data-testid="stSpinner"] span {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_clf_model():
    return load_model(MODEL_PATH, compile=False)

model = load_clf_model()
face_cascade = cv2.CascadeClassifier(HAAR_PATH)

# -----------------------------
# FUNCTIONS
# -----------------------------
def predict_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # MORE RELAXED FACE DETECTION
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,     # more sensitive
        minNeighbors=3,       # less strict
        minSize=(80, 80)      # avoid tiny false faces
    )

    results = []

    # IF NO FACE DETECTED → USE FULL IMAGE
    if len(faces) == 0:
        faces = [(0, 0, image.shape[1], image.shape[0])]

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = preprocess_input(face.astype("float32"))
        face = np.expand_dims(face, axis=0)

        probs = model.predict(face, verbose=0)[0]
        idx = np.argmax(probs)

        # DRAW STRONG BOX (ALWAYS VISIBLE)
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),   # pure green
            4              # thicker
        )

        cv2.putText(
            image,
            f"{CLASS_NAMES[idx]} {probs[idx]*100:.1f}%",
            (x, max(y - 12, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        results.append(probs)

    # RETURN FIRST FACE PROBABILITIES (SAFE)
    return image, results[0], len(faces)


def log_prediction(filename, probs, face_count):
    pd.DataFrame({
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "image_name": [filename],
        "wrinkles_%": [probs[0]*100],
        "clear_skin_%": [probs[1]*100],
        "dark_spots_%": [probs[2]*100],
        "puffy_eyes_%": [probs[3]*100],
        "faces_detected": [face_count]
    }).to_csv("prediction_logs.csv",
              mode="a",
              header=not os.path.exists("prediction_logs.csv"),
              index=False)

# -----------------------------
# HEADER
# -----------------------------
# -----------------------------
# HERO / INTRO IMAGE
# -----------------------------

st.markdown("<div class='title'>DermalScan</div>", unsafe_allow_html=True)
st.markdown("<b><div class='subtitle'>AI-based Facial Skin Aging Detection System</b></div>", unsafe_allow_html=True)


# -----------------------------
# UPLOAD
# -----------------------------
st.markdown("<b style='color:black;'>📤 Upload a facial image</b>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"])


if uploaded_file:
    img_bytes = uploaded_file.read()
    st.session_state.uploaded_image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)
    st.session_state.uploaded_name = uploaded_file.name
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# SHOW UPLOADED IMAGE
# -----------------------------
if st.session_state.uploaded_image is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    

    # Show image WITHOUT caption
    st.image(
        st.session_state.uploaded_image,
        channels="BGR",
        width=320
    )

    # Custom black caption
    st.markdown(
        "<div style='color:black; text-align:left; font-weight:500; margin-top:0px; margin-bottom:11px;'>Uploaded Image</div>",
        unsafe_allow_html=True
    )

    if st.button("✨ Analyze Image"):
        # Custom black analyzing text (instead of spinner text)
        with st.spinner(""):
            st.markdown(
                "<span style='color:black;'>Analyzing skin features...</span>",
                unsafe_allow_html=True
            )

            out, probs, face_count = predict_image(
                st.session_state.uploaded_image.copy()
            )

            st.session_state.result = {
                "output_img": out,
                "probs": probs,
                "face_count": face_count
            }

            log_prediction(st.session_state.uploaded_name, probs, face_count)

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# RESULT (PERSISTENT)
# -----------------------------
if st.session_state.result:
    output_img = st.session_state.result["output_img"]
    probs = st.session_state.result["probs"]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(output_img, channels="BGR", width=360)
    top = np.argmax(probs)
    st.markdown(
        f"<div class='result-box'>Final Prediction: {CLASS_NAMES[top]} ({probs[top]*100:.2f}%)</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    _, buffer = cv2.imencode(".jpg", output_img)

    st.download_button("⬇️ Download Annotated Image",
                       buffer.tobytes(),
                       "dermalscan_result.jpg",
                       "image/jpeg")

    if os.path.exists("prediction_logs.csv"):
        st.download_button("⬇️ Download Prediction Log (CSV)",
                           open("prediction_logs.csv", "rb"),
                           "prediction_logs.csv",
                           "text/csv")

    # Chart
    colors = ["#ff6f91", "#6fffb0", "#6fa8ff", "#c77dff"]
    fig, ax = plt.subplots(figsize=(6,3))
    ax.barh(CLASS_NAMES, probs*100, color=colors)
    ax.set_xlim(0,100)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Model Confidence")

    for i, v in enumerate(probs*100):
        ax.text(v+1, i, f"{v:.1f}%")

    ax.grid(axis="x", linestyle="--", alpha=0.3)
    st.pyplot(fig)
