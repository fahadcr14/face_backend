import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F  # For softmax

# Initialize Flask app
app = Flask(__name__)

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model and transformation setup
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()  # Set model to evaluation mode
    model.to(device)
    return model

def preprocess_image(image_path):
    transform = T.Compose([
        T.Resize((224, 224)),  # Resize image to 224x224
        T.ToTensor(),          # Convert image to Tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    image = Image.open(image_path).convert("RGB")  # Open and convert image to RGB
    return transform(image).unsqueeze(0)  # Add batch dimension

def get_probabilities(logits):
    # Apply softmax to get probabilities
    probabilities = F.softmax(logits, dim=1)
    percentages = probabilities * 100
    return percentages

def predict(image_path, model, class_names):
    image_tensor = preprocess_image(image_path).to(device)
    model.eval()
    with torch.inference_mode():  # Disable gradient calculations
        outputs = model(image_tensor)
        percentages = get_probabilities(outputs)
        _, predicted_class = torch.max(outputs, 1)  # Get the index of the highest logit
    predicted_label = class_names[predicted_class.item()]
    return predicted_label, percentages

# Define class names
class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

# Load model
model_path = r"model_85_nn_.pth"  # Update this with the correct model path
model = load_model(model_path)

# API to render the index page
@app.route('/')
def index():
    return render_template('index.html')

# API to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict_face_shape():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        os.makedirs('uploads',exist_ok=True)
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        predicted_label, percentages = predict(file_path, model, class_names)

        result = {class_names[i]: percentages[0, i].item() for i in range(len(class_names))}
        sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
        print(sorted_result)
        return jsonify(sorted_result)

if __name__ == '__main__':
    app.run(debug=False)
