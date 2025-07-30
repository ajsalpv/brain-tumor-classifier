import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import numpy as np
from model import load_model


st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)


@st.cache_resource
def load_trained_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('brain_tumor_model_weights.pth', device)
    
    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    
    return model, class_names, device


def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def predict_brain_tumor(model, image_tensor, class_names, device):
    """Make prediction on the preprocessed image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = logits.argmax(1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence, probabilities[0].cpu().numpy()


def main():
    st.title("🧠 Brain Tumor Classification")
    st.markdown("Upload a brain MRI image to classify the type of tumor")
    
  
    try:
        model, class_names, device = load_trained_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
   
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a brain MRI scan image"
    )
    
    if uploaded_file is not None:
       
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
        
        with col2:
            st.subheader("Prediction Results")
            
         
            with st.spinner("Analyzing image..."):
                try:
                    
                    image_tensor = preprocess_image(image)
                    
                    
                    predicted_class, confidence, all_probabilities = predict_brain_tumor(
                        model, image_tensor, class_names, device
                    )
                    
                    
                    st.success(f"**Predicted Class:** {predicted_class}")
                    st.info(f"**Confidence:** {confidence:.2%}")
                    
                    
                    st.subheader("All Class Probabilities")
                    for i, (class_name, prob) in enumerate(zip(class_names, all_probabilities)):
                        st.write(f"**{class_name}:** {prob:.2%}")
                        st.progress(prob)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    

    # with st.expander("ℹ️ Model Information"):
    #     st.write(f"**Classes:** {', '.join(class_names) if 'class_names' in locals() else 'Loading...'}")
    #     st.write("**Model Architecture:** CNN with 3 convolutional layers and 2 fully connected layers")
    #     st.write("**Input Size:** 128x128 pixels")
    #     st.write("**Device:** GPU" if torch.cuda.is_available() else "CPU")

if __name__ == "__main__":
    main()
