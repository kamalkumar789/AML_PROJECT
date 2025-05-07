from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os
import io
import torchvision.models as models
from densenet import DenseNet

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained DenseNet model first
model_path = os.path.join('/user/HS401/kk01579/APML_PROJECT_KAMALS/saved_models', 'densenet121_samplying_0.0001.pth')
model = DenseNet(pretrained=False)  # Don't load pretrained weights here
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

transform = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Same mean/std as training
])


@app.route('/predict', methods=['POST'])
def predict():
    # Check if the 'image' key is in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    
    image_file = request.files['image']

    # Safely get form data
    age = request.form.get('age', default=None)
    lesion_location = request.form.get('lesion_location', default=None)
    sex = request.form.get('sex', default=None)

    try:
        # Read the image bytes and open the image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'error': 'Invalid image format or corrupted image', 'message': str(e)}), 400
    
    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0).to(device)  # Move image to device
    # Perform prediction
    with torch.no_grad():
        output = model(input_tensor.to(device))  # shape: [1, 1]
        probs = torch.sigmoid(output).item()
    
    print(probs)
    # Determine diagnosis and confidence
    diagnosis = "malignant" if probs > 0.5 else "benign"
    confidence = round(probs * 100, 2) if diagnosis == "malignant" else round((1 - probs) * 100, 2)

    # Return the result
    return jsonify({
        'diagnosis': diagnosis,
        'confidence_score': confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
