import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Kategori fashion
selected_categories = ['Tshirts', 'Jeans', 'Casual Shoes', 'Handbags', 'Sunglasses']
label_to_idx = {label: idx for idx, label in enumerate(selected_categories)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

# Load model ResNet yang sudah dilatih ulang
@st.cache_resource(show_spinner=False)
def load_resnet_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(selected_categories))
    model.load_state_dict(torch.load('best_resnet_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

resnet_model = load_resnet_model()

# Transformasi gambar untuk ResNet
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model dan processor CLIP pretrained
@st.cache_resource(show_spinner=False)
def load_clip_model_processor():
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

clip_model, clip_processor = load_clip_model_processor()

clip_text_prompts = [
    "a photo of a T-shirt",
    "a photo of jeans",
    "a photo of casual shoes",
    "a photo of a handbag",
    "a photo of sunglasses"
]

def predict_resnet(image: Image.Image):
    input_tensor = resnet_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = resnet_model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, dim=1)
        return idx_to_label[predicted.item()], conf.item()

def predict_clip(image: Image.Image):
    inputs = clip_processor(text=clip_text_prompts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        conf, predicted = torch.max(probs, dim=1)
        label = clip_text_prompts[predicted.item()].replace("a photo of a ", "")
        return label, conf.item()


st.title("Prototype Perbandingan Model Fashion: ResNet-50 vs CLIP")
st.write("Unggah gambar fashion dan lihat prediksi dari kedua model secara berdampingan.")

uploaded_file = st.file_uploader("Unggah gambar (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar input", use_column_width=True)

    if st.button("Prediksi"):
        with st.spinner("Menjalankan prediksi..."):
            resnet_label, resnet_conf = predict_resnet(image)
            clip_label, clip_conf = predict_clip(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ResNet-50 Fine-tuned")
            st.write(f"Prediksi: **{resnet_label}**")
            st.write(f"Confidence: {resnet_conf*100:.2f}%")
        with col2:
            st.subheader("CLIP Zero-shot")
            st.write(f"Prediksi: **{clip_label}**")
            st.write(f"Confidence: {clip_conf*100:.2f}%")

        st.markdown("---")
        st.write("Model ResNet-50 dilatih ulang pada dataset fashion Anda, sedangkan CLIP melakukan klasifikasi zero-shot berdasarkan prompt teks.")

