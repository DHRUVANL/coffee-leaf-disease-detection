import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import time

st.set_page_config(page_title="Coffee Leaf AI", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("coffee_leaf_model.h5")

model = load_model()
class_names = ['Healthy', 'Leaf_miner', 'Rust']

translations = {
    "English": {
        "title": "☕ Coffee Leaf Disease Detection",
        "subtitle": "AI-powered plant disease classification system",
        "upload": "Upload Coffee Leaf Image",
        "prediction": "Prediction",
        "confidence": "Confidence",
        "probability": "Class Probability Distribution",
        "recommendation": "Farmer Recommendation",
        "about": "About This Project"
    },
    "Kannada": {
        "title": "☕ ಕಾಫಿ ಎಲೆ ರೋಗ ಪತ್ತೆ",
        "subtitle": "AI ಆಧಾರಿತ ಗಿಡರೋಗ ಗುರುತುಪಡಿಸುವ ವ್ಯವಸ್ಥೆ",
        "upload": "ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ",
        "prediction": "ಭವಿಷ್ಯವಾಣಿ",
        "confidence": "ನಂಬಿಕೆ ಮಟ್ಟ",
        "probability": "ವರ್ಗ ಸಂಭವನೀಯತೆ",
        "recommendation": "ರೈತರ ಸಲಹೆ",
        "about": "ಈ ಯೋಜನೆ ಬಗ್ಗೆ"
    }
}

language = st.sidebar.selectbox("Language", ["English", "Kannada"])

st.markdown(f"""
<h1 style='text-align:center; color:#2E8B57;'>
{translations[language]['title']}
</h1>
<p style='text-align:center; font-size:18px;'>
{translations[language]['subtitle']}
</p>
""", unsafe_allow_html=True)

st.divider()

uploaded_file = st.file_uploader(
    translations[language]["upload"],
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width="stretch")

    img = image.resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing leaf..."):
        time.sleep(1)
        predictions = model.predict(img_array)

    probs = predictions[0]
    predicted_class = class_names[np.argmax(probs)]
    confidence = float(np.max(probs) * 100)

    st.divider()

    if predicted_class == "Healthy":
        bg = "#28a745"
        icon = "✅"
    elif predicted_class == "Rust":
        bg = "#dc3545"
        icon = "⚠️"
    else:
        bg = "#fd7e14"
        icon = "🐛"

    st.markdown(f"""
    <div style="
    padding:30px;
    border-radius:20px;
    background: linear-gradient(135deg, {bg}, #1e1e1e);
    color:white;
    text-align:center;">
    <h2>{icon} {translations[language]['prediction']}: {predicted_class}</h2>
    <h3>{translations[language]['confidence']}: {confidence:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    # Confidence Progress
    st.subheader("Confidence Level")
    progress = st.progress(0)
    for i in range(int(confidence)):
        time.sleep(0.01)
        progress.progress(i + 1)

    # Probability Chart
    st.subheader(translations[language]["probability"])
    prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability": probs
    })
    st.bar_chart(prob_df.set_index("Class"))

    st.divider()
    st.subheader(translations[language]["recommendation"])

    if predicted_class == "Healthy":
        st.success("The plant is healthy. Continue monitoring and proper irrigation.")
    elif predicted_class == "Rust":
        st.error("Apply recommended fungicides and remove infected leaves.")
    else:
        st.warning("Use pest control measures for leaf miner management.")

    # Download Report
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

st.divider()
st.subheader(translations[language]["about"])

st.write("""
This AI system is built using Deep Learning (CNN).
It classifies coffee leaf diseases into:

• Healthy  
• Leaf Miner  
• Rust  

The model was trained on a labeled dataset and optimized for accuracy.
""")