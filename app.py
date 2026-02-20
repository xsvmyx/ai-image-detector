import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from model_arch import get_model 
import os
from dotenv import load_dotenv

load_dotenv()

FILENAME = os.getenv("FILENAME")
REPO_ID = os.getenv("REPO_ID")


# --- IMAGE PREPROCESSING ---
# Must match the normalization and size used during training
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@st.cache_resource
def load_model():
    # Download weights from Hugging Face Hub
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = get_model()
    
    
    checkpoint = torch.load(path, map_location="cpu")
    
    # Check if checkpoint is a state_dict or a full dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

# --- STREAMLIT UI ---
st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️")
st.title("🛡️ Deepfake Face Detector")
st.markdown("""
    This application uses a custom Convolutional Neural Network (CNN) to detect whether a face image is **Real** or a **Deepfake**.
""")


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model from Hugging Face: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocessing
    img_t = transform(image).unsqueeze(0) # Add batch dimension (1, 3, 256, 256)
    
    with torch.no_grad():
        output = model(img_t)
        # Apply Softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prob_fake = probabilities[0][0].item()
        prob_real = probabilities[0][1].item()
        prediction = torch.argmax(probabilities, dim=1).item()

    # Display Results
    st.divider()
    if prediction == 0:
        st.error(f"🚨 Prediction: **FAKE** (Confidence: {prob_fake:.2%})")
    else:
        st.success(f"✅ Prediction: **REAL** (Confidence: {prob_real:.2%})")
    
   