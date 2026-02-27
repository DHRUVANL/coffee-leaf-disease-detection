import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import tempfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

st.set_page_config(page_title="Coffee Leaf AI", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("coffee_leaf_model.keras")

model = load_model()
class_names = ['Healthy', 'Leaf_miner', 'Rust']

translations = {
    "English": {
        "title": "☕ Coffee Leaf Disease Detection",
        "subtitle": "AI-powered Explainable System using Grad-CAM",
        "upload": "Upload Coffee Leaf Image",
        "prediction": "Prediction",
        "confidence": "Confidence",
        "probability": "Class Probability Distribution",
        "cause": "Cause & Prevention",
        "download": "Download PDF Report"
    },
    "Kannada": {
        "title": "☕ ಕಾಫಿ ಎಲೆ ರೋಗ ಪತ್ತೆ",
        "subtitle": "Grad-CAM ಬಳಸಿ AI ವಿವರಣೆ",
        "upload": "ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ",
        "prediction": "ಭವಿಷ್ಯವಾಣಿ",
        "confidence": "ನಂಬಿಕೆ ಮಟ್ಟ",
        "probability": "ವರ್ಗ ಸಂಭವನೀಯತೆ",
        "cause": "ಕಾರಣ ಮತ್ತು ತಡೆಗಟ್ಟುವಿಕೆ",
        "download": "PDF ವರದಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ"
    }
}

language = st.sidebar.selectbox("Language", ["English", "Kannada"])

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

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
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img = image.resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    probs = predictions[0]

    predicted_class = class_names[np.argmax(probs)]
    confidence = float(np.max(probs) * 100)

    st.subheader(f"{translations[language]['prediction']}: {predicted_class}")
    st.write(f"{translations[language]['confidence']}: {confidence:.2f}%")

    heatmap = make_gradcam_heatmap(img_array, model)
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(
        np.array(image),
        0.6,
        heatmap,
        0.4,
        0
    )

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", width="stretch")

    with col2:
        st.image(superimposed_img, caption="Grad-CAM Heatmap", width="stretch")

    st.divider()
    st.subheader(translations[language]["probability"])

    prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability": probs
    })
    st.bar_chart(prob_df.set_index("Class"))

    if language == "English":
        if predicted_class == "Healthy":
            cause_text = "Plant is healthy. Maintain irrigation and soil nutrients."
        elif predicted_class == "Rust":
            cause_text = "Rust caused by fungal infection. Use fungicide and remove infected leaves."
        else:
            cause_text = "Leaf Miner caused by insect larvae. Apply pest control measures."
    else:
        if predicted_class == "Healthy":
            cause_text = "ಗಿಡ ಆರೋಗ್ಯಕರವಾಗಿದೆ. ಸರಿಯಾದ ನೀರಾವರಿ ಮತ್ತು ಪೋಷಕಾಂಶಗಳನ್ನು ಕಾಪಾಡಿಕೊಳ್ಳಿ."
        elif predicted_class == "Rust":
            cause_text = "ರಸ್ಟ್ ಫಂಗಲ್ ಸೋಂಕಿನಿಂದ ಉಂಟಾಗುತ್ತದೆ. ಫಂಗಿಸೈಡ್ ಬಳಸಿ ಮತ್ತು ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ."
        else:
            cause_text = "ಲೀಫ್ ಮೈನರ್ ಕೀಟಗಳಿಂದ ಉಂಟಾಗುತ್ತದೆ. ಕೀಟ ನಿಯಂತ್ರಣ ಕ್ರಮಗಳನ್ನು ಅನುಸರಿಸಿ."

    st.divider()
    st.subheader(f"🌿 {translations[language]['cause']}")
    st.info(cause_text)

    def create_pdf():
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(temp_file.name)

        elements = []

        pdfmetrics.registerFont(UnicodeCIDFont('HYSMyeongJo-Medium'))

        style = ParagraphStyle(
            name='NormalStyle',
            fontName='HYSMyeongJo-Medium',
            fontSize=12,
            textColor=colors.black
        )

        elements.append(Paragraph("Coffee Leaf Disease Report", style))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"Prediction: {predicted_class}", style))
        elements.append(Paragraph(f"Confidence: {confidence:.2f}%", style))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(cause_text, style))

        doc.build(elements)

        return temp_file.name

    pdf_path = create_pdf()

    with open(pdf_path, "rb") as f:
        st.download_button(
            label=translations[language]["download"],
            data=f,
            file_name="coffee_leaf_report.pdf",
            mime="application/pdf"
        )