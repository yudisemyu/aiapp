import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import time
import numpy as np
import pandas as pd

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Fashion Model Comparison", layout="wide")
st.markdown("<h1 style='text-align: center;'>👗 Fashion Image Classifier: ResNet vs CLIP</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- Kelas Label ---
class_labels = ['Tshirts', 'Jeans', 'Casual Shoes', 'Handbags', 'Sunglasses']

# --- Load CLIP ---
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# --- Load ResNet ---
@st.cache_resource
def load_resnet():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_labels))
    model.load_state_dict(torch.load("best_resnet_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# --- Prediksi CLIP ---
def predict_clip(image):
    model, processor = load_clip()
    inputs = processor(text=class_labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        start = time.time()
        outputs = model(**inputs)
        end = time.time()
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1).numpy().flatten()
    return class_labels[np.argmax(probs)], np.max(probs), probs, end - start

# --- Prediksi ResNet ---
def predict_resnet(image):
    model = load_resnet()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        start_time = time.time()
        output = model(img_tensor)
        end_time = time.time()
        probs = torch.nn.functional.softmax(output, dim=1).numpy().flatten()
    return class_labels[np.argmax(probs)], np.max(probs), probs, end_time - start_time

# --- Upload dan Tampilan Gambar ---
uploaded_file = st.file_uploader("📤 Upload gambar fashion untuk diklasifikasikan", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="🖼️ Gambar yang diupload", use_column_width=True)
    with col2:
        st.markdown("### 🔍 Prediksi Model")
        with st.spinner("Memproses dengan ResNet dan CLIP..."):
            # Prediksi dari kedua model
            res_label, res_conf, res_probs, res_time = predict_resnet(image)
            clip_label, clip_conf, clip_probs, clip_time = predict_clip(image)

        # --- Hasil Prediksi ---
        st.markdown("#### ✅ Hasil Prediksi:")
        st.markdown(f"- **ResNet:** `{res_label}` dengan confidence `{res_conf*100:.2f}%` (⏱ {res_time:.3f}s)")
        st.markdown(f"- **CLIP:** `{clip_label}` dengan confidence `{clip_conf*100:.2f}%` (⏱ {clip_time:.3f}s)")

        # --- Tabel Perbandingan Confidence ---
        df = pd.DataFrame({
            "Label": class_labels,
            "ResNet": res_probs,
            "CLIP": clip_probs
        }).set_index("Label")

        st.markdown("#### 📊 Confidence Kedua Model:")
        st.bar_chart(df)

        # --- Highlight Winner ---
        winner = "🤖 CLIP" if clip_conf > res_conf else "🧠 ResNet"
        st.markdown(f"### 🥇 Model yang Lebih Yakin: **{winner}**")
else:
    st.info("Silakan upload gambar fashion terlebih dahulu untuk memulai.")
