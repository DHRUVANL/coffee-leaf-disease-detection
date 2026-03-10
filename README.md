# ☕ CoffeeDoc AI: Smart Leaf Disease Diagnosis

An **Explainable AI-powered web application** that helps coffee farmers instantly detect and diagnose leaf diseases using a smartphone image.

The system uses **Deep Learning + Computer Vision** to identify diseases such as **Coffee Rust** and **Leaf Miner**, and provides **treatment recommendations** in **English and Kannada**.

**Author:** Dhruva NL

---

# 🚀 Live Demo

Once deployed on Streamlit Cloud:

https://your-app-name.streamlit.app

*(Update this link after deployment)*

---

# ✨ Key Features

### 🧠 High-Accuracy AI Model
Powered by a **fine-tuned EfficientNetB0 Convolutional Neural Network (CNN)** trained on coffee leaf disease datasets.

---

### 🔬 Explainable AI (Grad-CAM)
Uses **Grad-CAM visualization** to highlight **where the model detects disease patterns** on the leaf.

This increases **trust and interpretability** of the AI system.

---

### 🛡️ Image Validation (Gatekeeper System)

Before prediction, the system verifies whether the uploaded image actually contains a **green leaf** using **OpenCV color-space filtering**.

This prevents:

- Random image predictions
- Non-leaf uploads
- False diagnoses

---

### 🌍 Bilingual Farmer Interface

Supports **two languages**:

- English
- Kannada

This improves accessibility for **local farming communities**.

---

### 🌿 Farmer Recommendation System

After prediction, the system provides **disease-specific advice**.

#### Rust
- Remove infected leaves  
- Apply fungicide  
- Improve air circulation  

#### Leaf Miner
- Use pest control methods  
- Monitor leaves regularly  
- Remove damaged leaves  

#### Healthy
- Maintain balanced irrigation  
- Ensure proper soil nutrients  

---

# 🧠 AI Model Architecture

The deep learning pipeline uses **Transfer Learning**.

Input Image (224x224)  
↓  
Data Augmentation  
↓  
Rescaling  
↓  
EfficientNetB0 (ImageNet pretrained)  
↓  
Global Average Pooling  
↓  
Batch Normalization  
↓  
Dropout  
↓  
Softmax Classification  

Classes predicted:

- Healthy
- Leaf_miner
- Rust

---

# 📊 Explainability with Grad-CAM

Grad-CAM produces **visual heatmaps** showing which areas of the leaf influenced the model's decision.

Example output:

Original Leaf Image | AI Attention Heatmap

This makes the system **transparent and explainable**.

---

# 🛠️ Tech Stack

### Machine Learning
- TensorFlow
- Keras
- EfficientNetB0
- OpenCV

### Web Framework
- Streamlit

### Visualization
- Plotly
- Seaborn
- Matplotlib

### Language
- Python 3

---

# 📂 Project Structure

```
coffee-leaf-disease-detection
│
├── app.py
│     Streamlit web application
│
├── train.py
│     Model training pipeline
│
├── evaluate.py
│     Generates confusion matrix and classification report
│
├── coffee_leaf_model.keras
│     Trained neural network model
│
├── class_names.json
│     Encoded label mapping
│
├── requirements.txt
│     Dependencies for deployment
│
└── README.md
      Project documentation
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/DHRUVANL/coffee-leaf-disease-detection.git
cd coffee-leaf-disease-detection
```

Create virtual environment:

```bash
python -m venv tfenv
```

Activate environment:

Windows

```bash
tfenv\Scripts\activate
```

Mac / Linux

```bash
source tfenv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🧪 Train the Model

```bash
python train.py
```

This will generate:

```
coffee_leaf_model.keras
```

---

# 🚀 Run the Application

```bash
streamlit run app.py
```

The app will open in your browser:

```
http://localhost:8501
```

---

# ☁️ Deployment

The project can be deployed on:

- Streamlit Cloud
- HuggingFace Spaces
- AWS EC2
- Google Cloud

For quick deployment:

https://streamlit.io/cloud

---

# 📈 Future Improvements

Possible extensions:

- Mobile app for farmers
- Real-time camera disease detection
- Disease severity estimation
- Yield prediction AI
- Integration with weather data
- Multi-crop disease detection

---

# 👨‍💻 Author

**Dhruva NL**

AI & Machine Learning Engineering Student

GitHub:  
https://github.com/DHRUVANL

---

# ⭐ Support

If you found this project useful:

⭐ Star the repository  
🍴 Fork the project  
📢 Share with others

---

# 🎯 Final Result

CoffeeDoc AI helps farmers:

- Detect diseases early
- Understand AI decisions
- Take immediate treatment action

Making agriculture **smarter and more accessible with AI**.
