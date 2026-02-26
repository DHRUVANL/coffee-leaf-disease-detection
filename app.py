import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import time

st.set_page_config(
    page_title="Coffee Leaf Disease Detection",
    layout="centered"
)

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("coffee_leaf_model.h5")

model = load_model()

class_names = ['Healthy', 'Leaf_miner', 'Rust']

# ------------------------------
# Header Section
# ------------------------------
st.markdown("""
    <h1 style='text-align:center; color:#2E8B57;'>
        ☕ Coffee Leaf Disease Detection
    </h1>
    <p style='text-align:center; font-size:18px;'>
        AI-powered deep learning model for plant disease classification
    </p>
""", unsafe_allow_html=True)

st.divider()

uploaded_file = st.file_uploader("📂 Upload Coffee Leaf Image", type=["jpg","jpeg","png"])

# ------------------------------
# If Image Uploaded
# ------------------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width="stretch")

    img = image.resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analyzing leaf condition..."):
        time.sleep(1)
        predictions = model.predict(img_array)

    probs = predictions[0]
    predicted_class = class_names[np.argmax(probs)]
    confidence = float(np.max(probs) * 100)

    st.divider()

    # ------------------------------
    # Color Based on Disease
    # ------------------------------
    if predicted_class == "Healthy":
        bg_color = "#D4EDDA"
        text_color = "#155724"
    elif predicted_class == "Rust":
        bg_color = "#F8D7DA"
        text_color = "#721C24"
    else:
        bg_color = "#FFF3CD"
        text_color = "#856404"

    st.markdown(f"""
        <div style="
            padding:25px;
            border-radius:15px;
            background-color:{bg_color};
            text-align:center;
            font-size:22px;
            color:{text_color};
            box-shadow: 0 6px 15px rgba(0,0,0,0.15);">
            <b>Prediction:</b> {predicted_class} <br><br>
            <b>Confidence:</b> {confidence:.2f}%
        </div>
    """, unsafe_allow_html=True)

    # ------------------------------
    # Animated Confidence Bar
    # ------------------------------
    st.subheader("Confidence Level")
    progress = st.progress(0)

    for i in range(int(confidence)):
        time.sleep(0.01)
        progress.progress(i + 1)

    # ------------------------------
    # Probability Chart
    # ------------------------------
    st.subheader("Class Probability Distribution")

    prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability": probs
    })

    st.bar_chart(prob_df.set_index("Class"))

    # ------------------------------
    # Disease Information
    # ------------------------------
    st.divider()
    st.subheader("About This Condition")

    if predicted_class == "Healthy":
        st.success("The leaf appears healthy with no visible disease symptoms.")
    elif predicted_class == "Rust":
        st.error("Rust is a fungal disease that causes orange powdery spots on leaves.")
    else:
        st.warning("Leaf Miner causes white winding trails inside leaves.")

    # ------------------------------
    # Download Report
    # ------------------------------
    report = f"""
    Coffee Leaf Disease Prediction Report
    --------------------------------------
    Prediction: {predicted_class}
    Confidence: {confidence:.2f}%
    """

    st.download_button(
        label="📄 Download Prediction Report",
        data=report,
        file_name="prediction_report.txt"
    )