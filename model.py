import torch
import torch.nn as nn

def create_brain_tumor_model():
    """Create and return the brain tumor classification model"""
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128*16*16, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 4)
    )
    return model

def load_model(model_path, device='cpu'):
    """Load the trained model"""
    model = create_brain_tumor_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
