import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import time

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Coffee Leaf Disease Detection",
    layout="centered"
)

# ---------------------------------
# Load Model
# ---------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("coffee_leaf_model.keras")

model = load_model()
class_names = ['Healthy', 'Leaf_miner', 'Rust']

# ---------------------------------
# Grad-CAM Function (Automatic Conv Detection)
# ---------------------------------
def make_gradcam_heatmap(img_array, model):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer("Conv_1").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()
# ---------------------------------
# UI Header
# ---------------------------------
st.markdown("""
    <h1 style='text-align:center; color:#2E8B57;'>
        ☕ Coffee Leaf Disease Detection
    </h1>
    <p style='text-align:center; font-size:18px;'>
        AI-powered Deep Learning with Explainable AI (Grad-CAM)
    </p>
""", unsafe_allow_html=True)

st.divider()

uploaded_file = st.file_uploader("📂 Upload Coffee Leaf Image", type=["jpg", "jpeg", "png"])

# ---------------------------------
# Prediction Section
# ---------------------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width="stretch")

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analyzing leaf condition..."):
        time.sleep(1)
        predictions = model.predict(img_array)

    probs = predictions[0]
    predicted_class = class_names[np.argmax(probs)]
    confidence = float(np.max(probs) * 100)

    st.divider()

    # ---------------------------------
    # Dynamic Prediction Card
    # ---------------------------------
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

    # ---------------------------------
    # Animated Confidence Bar
    # ---------------------------------
    st.subheader("Confidence Level")
    progress = st.progress(0)
    for i in range(int(confidence)):
        time.sleep(0.01)
        progress.progress(i + 1)

    # ---------------------------------
    # Probability Chart
    # ---------------------------------
    st.subheader("Class Probability Distribution")

    prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability": probs
    })

    st.bar_chart(prob_df.set_index("Class"))

    # ---------------------------------
    # Grad-CAM Heatmap
    # ---------------------------------
    st.subheader("Model Attention (Grad-CAM)")

    heatmap = make_gradcam_heatmap(img_array, model)

    if heatmap is not None:

        heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = cv2.addWeighted(
            np.array(image), 0.6, heatmap, 0.4, 0
        )

        st.image(superimposed_img, caption="Grad-CAM Heatmap", width="stretch")
    else:
        st.warning("Grad-CAM visualization not available.")

    # ---------------------------------
    # Disease Information
    # ---------------------------------
    st.divider()
    st.subheader("About This Condition")

    if predicted_class == "Healthy":
        st.success("The leaf appears healthy with no visible disease symptoms.")
    elif predicted_class == "Rust":
        st.error("Rust is a fungal disease causing orange powdery spots on leaves.")
    else:
        st.warning("Leaf Miner causes white winding trails inside leaves.")

    # ---------------------------------
    # Download Report
    # ---------------------------------
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