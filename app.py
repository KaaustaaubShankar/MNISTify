from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

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
        out = out.reshape(out.size(0), -1)  # Use reshape instead of view
        out = self.fc(out)
        return out

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained CNN model when the app starts
MODEL_PATH = 'mnist_convnet.pth'

# Initialize the model
model = ConvNet()

# Load the state_dict into the model
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

@app.route('/')
def home():
    return "MNISTfy Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the pixel data from the frontend
    data = request.get_json()

    # Get the pixel values (normalized between 0 and 1)
    pixels = data.get('pixels', [])

    if len(pixels) != 784:  # Ensure the pixel data has the correct length (28x28 = 784)
        return jsonify({'error': 'Invalid input size, expected 784 pixels'}), 400

    # Convert the list of pixels into a numpy array
    input_data = np.array(pixels).reshape(28, 28).astype(np.float32)

    # Convert to PIL Image (for easier handling with PyTorch transforms)
    image = Image.fromarray(input_data)

    # Ensure the image is in a mode that can be saved (uint8)
    image = image.convert('L')  # Convert to grayscale (mode 'L')

    # Normalize the image (ensure pixel values are in range 0-255)
    image = np.array(image) * 255.0
    image = image.astype(np.uint8)

    # Convert back to PIL Image
    image = Image.fromarray(image)

    # Save the input image to the current directory
    image.save('input_image.png')  # Save the image with a default name

    # Preprocessing the image as MNIST images (grayscale, normalize to [0, 1])
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),  # Make sure it's grayscale
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the pixel values
    ])

    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension (1, 1, 28, 28)

    # Make the prediction
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = output.argmax(dim=1).item()  # Get the class with the highest probability

    # Return the result as a JSON response
    return jsonify({'prediction': int(predicted_class)})

if __name__ == '__main__':
    # Check if the model exists before running the server
    import os
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
    else:
        print("Starting Flask app...")
    app.run(debug=True, port=8000)
