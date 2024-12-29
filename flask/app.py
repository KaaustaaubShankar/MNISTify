from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from lime import lime_image
from skimage.segmentation import slic
from io import BytesIO
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
from skimage.segmentation import mark_boundaries
import matplotlib
from waitress import serve

matplotlib.use('Agg')  # Use non-GUI backend for matplotlib


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'mnist_convnet.pth'
model = ConvNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# LIME prediction function
def predict_proba(images):
    """
    Converts RGB images from LIME to PyTorch-compatible grayscale tensors,
    performs inference, and converts predictions back to NumPy arrays.
    """
    model.eval()
    images = torch.tensor(images).float()
    images = images.mean(dim=-1, keepdim=True)
    images = images.permute(0, 3, 1, 2)
    with torch.no_grad():
        outputs = model(images)
    return F.softmax(outputs, dim=1).cpu().numpy()

def segmentation_fn(image):
    """
    Applies SLIC segmentation to create superpixels.
    """
    return slic(image, n_segments=50, compactness=1, start_label=1)

def generate_lime_explanation(image_array):
    """
    Generates LIME explanation for the given image.
    """
    explainer = lime_image.LimeImageExplainer()
    
    # Convert grayscale to RGB
    image_rgb = gray2rgb(image_array)
    
    # Get LIME explanation
    explanation = explainer.explain_instance(
        image_rgb,
        predict_proba,
        top_labels=1,
        hide_color=0,
        num_samples=1000,
        segmentation_fn=segmentation_fn
    )
    
    # Get the top label
    top_label = explanation.top_labels[0]
    
    # Generate the explanation image
    temp, mask = explanation.get_image_and_mask(
        label=top_label,
        positive_only=False,
        num_features=10,
        hide_rest=False
    )
    
    # Create the visualization
    explanation_image = mark_boundaries(image_rgb, mask, color=(1, 0, 0))
    
    return explanation_image, top_label

@app.route('/')
def home():
    return "MNISTfy Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pixels = data.get('pixels', [])

    if len(pixels) != 784:
        return jsonify({'error': 'Invalid input size, expected 784 pixels'}), 400

    # Convert pixels to image array
    input_array = np.array(pixels).reshape(28, 28).astype(np.float32)
    
    # Create PIL Image
    image = Image.fromarray((input_array * 255).astype(np.uint8))
    
    # Transform for model prediction
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Make prediction
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = output.argmax(dim=1).item()

    # Generate LIME explanation
    explanation_image, explained_label = generate_lime_explanation(input_array)
    
    # Convert explanation image to bytes
    plt.figure(figsize=(6, 6))
    plt.imshow(explanation_image)
    plt.axis('off')
    
    # Save plot to bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return jsonify({
        'prediction': int(predicted_class),
        'explanation_image': buf.getvalue().hex()  # Convert bytes to hex string for JSON
    })

@app.route('/explain', methods=['POST'])
def explain():
    data = request.get_json()
    pixels = data.get('pixels', [])

    if len(pixels) != 784:
        return jsonify({'error': 'Invalid input size, expected 784 pixels'}), 400

    input_array = np.array(pixels).reshape(28, 28).astype(np.float32)
    explanation_image, explained_label = generate_lime_explanation(input_array)

    # Convert explanation image to bytes
    plt.figure(figsize=(6, 6))
    plt.imshow(explanation_image)
    plt.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return send_file(
        buf,
        mimetype='image/png'
    )

if __name__ == '__main__':
    import os
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
    else:
        print("Starting Flask app...")
    serve(app, host='0.0.0.0', port=8000)