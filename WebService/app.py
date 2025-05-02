from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os

app = Flask(__name__)

model_path = os.path.join('model', 'model.pth')
model = torch.load(model_path, map_location=torch.device('cpu'))  # or 'cuda' if you have GPU
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).item()

    diagnosis = "malignant" if probs > 0.5 else "benign"
    confidence = round(probs * 100, 2) if diagnosis == "malignant" else round((1 - probs) * 100, 2)

    return jsonify({
        'diagnosis': diagnosis,
        'confidence_score': confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
