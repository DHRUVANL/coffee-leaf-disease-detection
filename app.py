import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import google.generativeai as genai
from PIL import Image
import plotly.express as px
import cv2
import time

# --- CLOUD PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'coffee_leaf_model.keras')
LABELS_PATH = os.path.join(BASE_DIR, 'class_names.json')

# Configure Gemini AI
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model_chat = genai.GenerativeModel('gemini-2.5-flash')

# Page Config MUST be the first Streamlit command
st.set_page_config(page_title="CoffeeDoc AI", page_icon="☕", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS UI POLISH ---
st.markdown("""
    <style>
    /* Breathing animation for the main title */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .hero-title {
        animation: pulse 3s infinite ease-in-out;
        background: -webkit-linear-gradient(45deg, #4CAF50, #1B5E20);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3.5rem !important;
        font-weight: 900;
        padding-bottom: 10px;
    }
    .hero-subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    /* Style the image uploader box */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #4CAF50;
        border-radius: 15px;
        background-color: #f1f8e9;
    }
    </style>
    """, unsafe_allow_html=True)

# --- TRANSLATIONS DICTIONARY ---
translations = {
    "English": {
        "tab1": "🔍 Disease Diagnosis",
        "tab2": "🤖 Coffee Expert Chat",
        "title": "CoffeeDoc AI",
        "subtitle": "Instant Coffee Leaf Diagnosis Powered by Deep Learning",
        "uploader": "Drag and drop a leaf image here...",
        "button": "✨ Analyze Leaf Health",
        "loading1": "Extracting features from image...",
        "loading2": "Running neural network analysis...",
        "loading3": "Generating heatmaps...",
        "confidence": "Confidence Level",
        "Healthy": "Healthy",
        "Rust": "Coffee Rust",
        "Leaf_miner": "Leaf Miner",
        "advice_rust": "⚠️ **Action:** Apply copper-based fungicides and prune affected branches.",
        "advice_miner": "⚠️ **Action:** Remove mined leaves and consider organic neem oil.",
        "advice_healthy": "✅ **Great News:** This leaf appears to be in excellent condition!",
        "chat_title": "🤖 AI Agronomist",
        "chat_info": "Ask me anything specifically about coffee farming, varieties, or pest control!",
        "chat_placeholder": "How do I improve coffee yield?",
        "api_warning": "⚠️ Please set up your Gemini API Key in Streamlit Secrets.",
        "sys_prompt": "You are a professional Coffee Agronomist. Answer ONLY if related to coffee. Respond entirely in English. User Question: ",
        "toast": "Welcome to CoffeeDoc AI! ☕"
    },
    "ಕನ್ನಡ": {
        "tab1": "🔍 ರೋಗ ಪತ್ತೆ",
        "tab2": "🤖 ಕಾಫಿ ತಜ್ಞರ ಚಾಟ್",
        "title": "ಕಾಫಿಡಾಕ್ AI",
        "subtitle": "ಡೀಪ್ ಲರ್ನಿಂಗ್ ಮೂಲಕ ಕಾಫಿ ಎಲೆಯ ತಕ್ಷಣದ ರೋಗ ಪತ್ತೆ",
        "uploader": "ಎಲೆಯ ಚಿತ್ರವನ್ನು ಇಲ್ಲಿ ಹಾಕಿ...",
        "button": "✨ ಎಲೆಯ ಆರೋಗ್ಯವನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
        "loading1": "ಚಿತ್ರದಿಂದ ಮಾಹಿತಿಯನ್ನು ಪಡೆಯಲಾಗುತ್ತಿದೆ...",
        "loading2": "ನ್ಯೂರಲ್ ನೆಟ್‌ವರ್ಕ್ ವಿಶ್ಲೇಷಣೆ ನಡೆಯುತ್ತಿದೆ...",
        "loading3": "ಹೀಟ್‌ಮ್ಯಾಪ್‌ಗಳನ್ನು ರಚಿಸಲಾಗುತ್ತಿದೆ...",
        "confidence": "ನಂಬಿಕೆಯ ಮಟ್ಟ",
        "Healthy": "ಆರೋಗ್ಯಕರವಾಗಿದೆ",
        "Rust": "ಕಾಫಿ ತುಕ್ಕು ರೋಗ (Rust)",
        "Leaf_miner": "ಎಲೆ ಕೊರೆಯುವ ಹುಳು (Leaf Miner)",
        "advice_rust": "⚠️ **ಪರಿಹಾರ:** ತಾಮ್ರ ಆಧಾರಿತ ಶಿಲೀಂಧ್ರನಾಶಕಗಳನ್ನು ಬಳಸಿ ಮತ್ತು ಪೀಡಿತ ಕೊಂಬೆಗಳನ್ನು ಕತ್ತರಿಸಿ.",
        "advice_miner": "⚠️ **ಪರಿಹಾರ:** ಪೀಡಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಬೇವಿನ ಎಣ್ಣೆ ಚಿಕಿತ್ಸೆಯನ್ನು ಪರಿಗಣಿಸಿ.",
        "advice_healthy": "✅ **ಶುಭ ಸುದ್ದಿ:** ಈ ಎಲೆಯು ಅತ್ಯುತ್ತಮ ಸ್ಥಿತಿಯಲ್ಲಿ ಕಂಡುಬರುತ್ತದೆ!",
        "chat_title": "🤖 ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಕೃಷಿ ತಜ್ಞ",
        "chat_info": "ಕಾಫಿ ಕೃಷಿ, ರೋಗಗಳು ಅಥವಾ ಕೀಟಗಳ ನಿಯಂತ್ರಣದ ಬಗ್ಗೆ ಯಾವುದೇ ಪ್ರಶ್ನೆ ಕೇಳಿ!",
        "chat_placeholder": "ಕಾಫಿ ಇಳುವರಿಯನ್ನು ಹೆಚ್ಚಿಸುವುದು ಹೇಗೆ?",
        "api_warning": "⚠️ ದಯವಿಟ್ಟು Streamlit Secrets ನಲ್ಲಿ Gemini API ಕೀಯನ್ನು ಸೇರಿಸಿ.",
        "sys_prompt": "ನೀವು ವೃತ್ತಿಪರ ಕಾಫಿ ಕೃಷಿ ತಜ್ಞರು. ಕಾಫಿ ಬೆಳೆಗೆ ಸಂಬಂಧಿಸಿದ ಪ್ರಶ್ನೆಗಳಿಗೆ ಮಾತ್ರ ಉತ್ತರಿಸಿ. ಸಂಪೂರ್ಣವಾಗಿ ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ಉತ್ತರಿಸಿ. ಬಳಕೆದಾರರ ಪ್ರಶ್ನೆ: ",
        "toast": "ಕಾಫಿಡಾಕ್ AI ಗೆ ಸುಸ್ವಾಗತ! ☕"
    }
}

# --- SIDEBAR ---
st.sidebar.title("Language / ಭಾಷೆ")
lang = st.sidebar.radio("Select Language:", ("English", "ಕನ್ನಡ"), index=0)
t = translations[lang]

# Welcome Toast (Only runs once per session)
if "welcomed" not in st.session_state:
    st.toast(t["toast"], icon="👋")
    st.session_state.welcomed = True

st.sidebar.divider()
st.sidebar.write("**Developed by:** Dhruva NL")

@st.cache_resource
def load_coffee_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

def load_labels():
    if not os.path.exists(LABELS_PATH):
        return ["Healthy", "Leaf_miner", "Rust"]
    with open(LABELS_PATH, 'r') as f:
        return json.load(f)

model = load_coffee_model()
class_names = load_labels()

# --- TABS SETUP ---
tab1, tab2 = st.tabs([t["tab1"], t["tab2"]])

# === TAB 1: DISEASE DIAGNOSIS ===
with tab1:
    st.markdown(f"<h1 class='hero-title'>☕ {t['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='hero-subtitle'>{t['subtitle']}</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(t["uploader"], type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2:
            st.image(image, caption='Uploaded Image', width='stretch')
        
        if st.button(t["button"], use_container_width=True):
            if model is not None:
                with st.status("🔍 Analyzing Leaf...", expanded=True) as status:
                    st.write(t["loading1"])
                    time.sleep(0.5)
                    
                    img = image.resize((224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    st.write(t["loading2"])
                    predictions = model.predict(img_array)
                    scores = tf.nn.softmax(predictions[0]).numpy() * 100
                    idx = np.argmax(predictions[0])
                    label_key = class_names[idx]
                    confidence = scores[idx]
                    
                    st.write(t["loading3"])
                    status.update(label="Analysis Complete!", state="complete", expanded=False)

                st.divider()

                res_col1, res_col2 = st.columns([2, 1])
                
                with res_col1:
                    st.subheader(f"Diagnosis: **{t.get(label_key, label_key)}**")
                    if label_key == "Rust":
                        st.error(t["advice_rust"])
                    elif label_key == "Leaf_miner":
                        st.warning(t["advice_miner"])
                    else:
                        st.success(t["advice_healthy"])
                        st.balloons()

                with res_col2:
                    st.metric(label=t["confidence"], value=f"{confidence:.2f}%", delta="High Accuracy" if confidence > 85 else "Needs Verification")

                st.divider()

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 📊 Confidence Breakdown")
                    fig = px.bar(
                        x=class_names, 
                        y=scores, 
                        color=class_names,
                        labels={'x': 'Disease Class', 'y': 'Confidence (%)'}
                    )
                    st.plotly_chart(fig, width='stretch')

                with col2:
                    st.markdown("### 🌡️ AI Detection Heatmap")
                    st.caption("Red areas show exactly where the AI found signs of disease.")
                    
                    heatmap = np.zeros((224, 224))
                    patch_size = 32
                    baseline_prob = predictions[0][idx]
                    
                    occluded_images = []
                    coords = []
                    for y in range(0, 224, patch_size):
                        for x in range(0, 224, patch_size):
                            img_occ = np.copy(img_array[0])
                            img_occ[y:y+patch_size, x:x+patch_size, :] = 0
                            occluded_images.append(img_occ)
                            coords.append((y, x))
                            
                    occluded_images = np.array(occluded_images)
                    occ_preds = model.predict(occluded_images, verbose=0)
                    
                    for i, (y, x) in enumerate(coords):
                        drop = baseline_prob - occ_preds[i][idx]
                        heatmap[y:y+patch_size, x:x+patch_size] = drop
                        
                    heatmap = np.maximum(heatmap, 0)
                    if np.max(heatmap) > 0:
                        heatmap /= np.max(heatmap)
                        
                    heatmap_resized = cv2.resize(heatmap, (image.width, image.height))
                    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                    
                    orig_img_arr = np.array(image)
                    superimposed = cv2.addWeighted(orig_img_arr, 0.6, heatmap_color, 0.4, 0)
                    
                    st.image(superimposed, width='stretch')

            else:
                st.error("Model Error: Please check your GitHub files.")

# === TAB 2: AI COFFEE EXPERT ===
with tab2:
    st.title(t["chat_title"])
    st.info(t["chat_info"])

    chat_history_key = f"messages_{lang}"
    if chat_history_key not in st.session_state:
        st.session_state[chat_history_key] = []

    for message in st.session_state[chat_history_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(t["chat_placeholder"]):
        st.session_state[chat_history_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if "GEMINI_API_KEY" not in st.secrets:
                st.warning(t["api_warning"])
            else:
                full_prompt = t["sys_prompt"] + prompt
                try:
                    response = model_chat.generate_content(full_prompt)
                    st.markdown(response.text)
                    st.session_state[chat_history_key].append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"API Error: {e}")
