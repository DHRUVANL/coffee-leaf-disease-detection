import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

# --- CLOUD PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'coffee_leaf_model.keras')
LABELS_PATH = os.path.join(BASE_DIR, 'class_names.json')

# Page Config
st.set_page_config(page_title="CoffeeDoc AI", page_icon="☕", layout="centered")

# --- TRANSLATIONS ---
translations = {
    "English": {
        "title": "☕ Coffee Leaf Disease AI",
        "subtitle": "Upload a photo of a coffee leaf for diagnosis.",
        "uploader": "Choose a leaf image...",
        "button": "🔍 Run Diagnosis",
        "loading": "AI is analyzing...",
        "confidence": "Confidence Level",
        "Healthy": "Healthy",
        "Rust": "Coffee Rust",
        "Leaf_miner": "Leaf Miner",
        "advice_rust": "⚠️ **Action:** Apply copper-based fungicides and prune affected branches.",
        "advice_miner": "⚠️ **Action:** Remove mined leaves and consider organic neem oil.",
        "advice_healthy": "✅ **Great News:** This leaf appears to be healthy!",
        "footer": "Disclaimer: Consult an agronomist for critical decisions."
    },
    "ಕನ್ನಡ": {
        "title": "☕ ಕಾಫಿ ಎಲೆ ರೋಗ ಪತ್ತೆ ಹಚ್ಚುವ AI",
        "subtitle": "ರೋಗ ಪತ್ತೆಹಚ್ಚಲು ಕಾಫಿ ಎಲೆಯ ಫೋಟೋವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        "uploader": "ಎಲೆಯ ಚಿತ್ರವನ್ನು ಆರಿಸಿ...",
        "button": "🔍 ರೋಗ ಪತ್ತೆಹಚ್ಚಿ",
        "loading": "AI ವಿಶ್ಲೇಷಿಸುತ್ತಿದೆ...",
        "confidence": "ನಂಬಿಕೆಯ ಮಟ್ಟ",
        "Healthy": "ಆರೋಗ್ಯಕರವಾಗಿದೆ",
        "Rust": "ಕಾಫಿ ತುಕ್ಕು ರೋಗ (Rust)",
        "Leaf_miner": "ಎಲೆ ಕೊರೆಯುವ ಹುಳು (Leaf Miner)",
        "advice_rust": "⚠️ **ಪರಿಹಾರ:** ತಾಮ್ರ ಆಧಾರಿತ ಶಿಲೀಂಧ್ರನಾಶಕಗಳನ್ನು ಬಳಸಿ ಮತ್ತು ಪೀಡಿತ ಕೊಂಬೆಗಳನ್ನು ಕತ್ತರಿಸಿ.",
        "advice_miner": "⚠️ **ಪರಿಹಾರ:** ಪೀಡಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಬೇವಿನ ಎಣ್ಣೆ ಚಿಕಿತ್ಸೆಯನ್ನು ಪರಿಗಣಿಸಿ.",
        "advice_healthy": "✅ **ಶುಭ ಸುದ್ದಿ:** ಈ ಎಲೆಯು ಆರೋಗ್ಯಕರವಾಗಿ ಕಂಡುಬರುತ್ತದೆ!",
        "footer": "ಸೂಚನೆ: ಕೃಷಿ ನಿರ್ಧಾರಗಳಿಗಾಗಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ."
    }
}

# Language Selection in Sidebar
st.sidebar.title("Language / ಭಾಷೆ")
lang = st.sidebar.radio("Select Language:", ("English", "ಕನ್ನಡ"))
t = translations[lang]

@st.cache_resource
def load_coffee_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

def load_labels():
    if not os.path.exists(LABELS_PATH):
        return ["Healthy", "Leaf_miner", "Rust"] # Fallback
    with open(LABELS_PATH, 'r') as f:
        return json.load(f)

# Initialize
model = load_coffee_model()
class_names = load_labels()

# UI Header
st.title(t["title"])
st.write(t["subtitle"])

# Image Upload
uploaded_file = st.file_uploader(t["uploader"], type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button(t["button"]):
        if model is not None:
            with st.spinner(t["loading"]):
                img = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                predictions = model.predict(img_array)
                idx = np.argmax(predictions[0])
                label_key = class_names[idx]
                confidence = 100 * np.max(tf.nn.softmax(predictions[0]))
                
                # Show Results
                st.subheader(f"{t[label_key]}")
                st.write(f"**{t['confidence']}:** {confidence:.2f}%")
                
                # Advice logic
                if label_key == "Rust":
                    st.warning(t["advice_rust"])
                elif label_key == "Leaf_miner":
                    st.warning(t["advice_miner"])
                else:
                    st.success(t["advice_healthy"])
        else:
            st.error("Model Error")

st.divider()
st.caption(t["footer"])
