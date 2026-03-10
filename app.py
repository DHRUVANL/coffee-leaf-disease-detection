import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import plotly.express as px
import json

# --- 1. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="CoffeeDoc AI", page_icon="☕", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .prediction-card {
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1e5128 0%, #4e944f 100%);
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CORE LOGIC ---

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('coffee_leaf_model.keras')

@st.cache_resource
def load_class_names():
    """Loads the exact label order saved during training."""
    try:
        with open('class_names.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("class_names.json not found! Using default alphabetical order.")
        return ['Healthy', 'Leaf_miner', 'Rust']

def is_valid_leaf(image):
    """Gatekeeper: Checks if the image has enough green/leaf-like colors."""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(img_cv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / (image.size[0] * image.size[1])
    return green_ratio > 0.05

def get_refined_gradcam(img_array, model, last_conv_layer_name, threshold):
    """Refined Grad-CAM to isolate disease spots from background."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        class_idx = np.argmax(preds[0])
        class_channel = preds[:, class_idx]

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap[heatmap < threshold] = 0
    return heatmap

# --- 3. TRANSLATIONS ---
LANG_DICT = {
    'English': {
        'title': "CoffeeDoc AI: Smart Diagnosis",
        'upload': "Upload a leaf photo",
        'analysis': "AI Diagnostic Analysis",
        'conf': "Confidence Level",
        'advice': "Farmer's Action Plan",
        'invalid': "❌ Invalid Image: This doesn't look like a coffee leaf. Please take a clear photo of a leaf.",
        'classes': {'Healthy': 'Healthy', 'Rust': 'Coffee Leaf Rust', 'Leaf_miner': 'Leaf Miner'},
        'tips': {
            'Healthy': ["Continue regular monitoring", "Check soil pH levels"],
            'Rust': ["Apply Copper-based fungicide", "Prune for better airflow"],
            'Leaf_miner': ["Introduce predatory wasps", "Remove infested leaves"]
        }
    },
    'Kannada': {
        'title': "ಕಾಫಿಡಾಕ್ AI: ಸ್ಮಾರ್ಟ್ ರೋಗ ಪತ್ತೆ",
        'upload': "ಎಲೆಯ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        'analysis': "AI ರೋಗನಿರ್ಣಯದ ವಿಶ್ಲೇಷಣೆ",
        'conf': "ನಿಖರತೆಯ ಮಟ್ಟ",
        'advice': "ರೈತರಿಗೆ ಸಲಹೆಗಳು",
        'invalid': "❌ ಅಮಾನ್ಯ ಚಿತ್ರ: ಇದು ಕಾಫಿ ಎಲೆಯಂತೆ ಕಾಣುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟವಾದ ಎಲೆಯ ಫೋಟೋ ತೆಗೆಯಿರಿ.",
        'classes': {'Healthy': 'ಆರೋಗ್ಯಕರ', 'Rust': 'ಎಲೆ ತುಕ್ಕು ರೋಗ', 'Leaf_miner': 'ಎಲೆ ಕೊರೆಯುವ ಹುಳು'},
        'tips': {
            'Healthy': ["ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ ಮುಂದುವರಿಸಿ", "ಮಣ್ಣಿನ pH ಮಟ್ಟವನ್ನು ಪರೀಕ್ಷಿಸಿ"],
            'Rust': ["ತಾಮ್ರ ಆಧಾರಿತ ಶಿಲೀಂಧ್ರನಾಶಕ ಬಳಸಿ", "ಗಾಳಿಯಾಡಲು ಕೊಂಬೆ ಕತ್ತರಿಸಿ"],
            'Leaf_miner': ["ಕೀಟನಾಶಕಗಳನ್ನು ಬಳಸಿ", "ಪೀಡಿತ ಎಲೆಗಳನ್ನು ಸುಟ್ಟು ಹಾಕಿ"]
        }
    }
}

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🌿 Settings")
    lang = st.selectbox("Language / ಭಾಷೆ", ["English", "Kannada"])
    st.markdown("---")
    sensitivity = st.slider("Heatmap Sensitivity", 0.1, 0.9, 0.5)

T = LANG_DICT[lang]
class_names = load_class_names() # Loads the exact order from train.py

# --- 5. MAIN INTERFACE ---
st.title(T['title'])
uploaded_file = st.file_uploader(T['upload'], type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    if not is_valid_leaf(image):
        st.error(T['invalid'])
        st.image(image, width=300, caption="Rejected Image")
    else:
        model = load_model()
        if model:
            img_resized = image.resize((224, 224))
            img_array = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img_resized), axis=0)

            # Prediction
            preds = model.predict(img_array)
            idx = np.argmax(preds[0])
            raw_label = class_names[idx] # Gets exact folder name
            display_label = T['classes'].get(raw_label, raw_label) # Translates to UI language
            conf = preds[0][idx] * 100

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="Uploaded Specimen", use_container_width=True)
                
            with col2:
                st.markdown(f"""
                    <div class="prediction-card">
                        <h3>{T['analysis']}</h3>
                        <h1>{display_label}</h1>
                        <p>{T['conf']}: {conf:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Grad-CAM Overlay
                try:
                    heatmap = get_refined_gradcam(img_array, model, "top_activation", sensitivity)
                    heatmap_resized = cv2.resize(heatmap, (224, 224))
                    heatmap_cv = np.uint8(255 * heatmap_resized)
                    heatmap_color = cv2.applyColorMap(heatmap_cv, cv2.COLORMAP_JET)
                    original_cv = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
                    superimposed = cv2.addWeighted(original_cv, 0.6, heatmap_color, 0.4, 0)
                    st.image(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB), caption="AI Heatmap", use_container_width=True)
                except ValueError:
                    st.warning("Grad-CAM is not available for this model architecture.")

            st.markdown("---")
            tab1, tab2 = st.tabs(["📋 " + T['advice'], "📊 Statistics"])
            
            with tab1:
                tips_list = T['tips'].get(raw_label, ["Consult an agronomist."])
                cols = st.columns(len(tips_list))
                for i, tip in enumerate(tips_list):
                    cols[i].success(tip)

            with tab2:
                # Plotly Chart
                df = pd.DataFrame({
                    'Disease': [T['classes'].get(c, c) for c in class_names], 
                    'Prob': preds[0]
                })
                fig = px.bar(df, x='Prob', y='Disease', orientation='h', color='Prob', color_continuous_scale='Greens')
                st.plotly_chart(fig, use_container_width=True)
