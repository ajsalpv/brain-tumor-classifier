from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import base64
import io
from model import load_model

app = Flask(__name__)

model = None
class_names = None
device = None

def load_trained_model():
    """Load the trained model and class names"""
    global model, class_names, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('brain_tumor_model_weights.pth', device)
    
    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_brain_tumor(image_tensor):
    """Make prediction on the preprocessed image"""
    global model, class_names, device
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = logits.argmax(1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {
        'predicted_class': class_names[predicted_class],
        'confidence': confidence,
        'all_probabilities': {
            class_names[i]: float(prob) for i, prob in enumerate(probabilities[0].cpu().numpy())
        }
    }

@app.route('/')
def index():
    return render_template('index.html', class_names=class_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        
        image = Image.open(file.stream)
        image_tensor = preprocess_image(image)
        
        
        result = predict_brain_tumor(image_tensor)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    
    load_trained_model()
    print("Model loaded successfully!")
    print(f"Available classes: {class_names}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
