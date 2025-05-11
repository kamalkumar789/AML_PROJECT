from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os
import io
import torchvision.models as models
from densenet import DenseNet
from flask_cors import CORS
from grad_cam import GradCAM
import base64
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained DenseNet model
model_path = os.path.join('/user/HS401/kk01579/APML_PROJECT_KAMALS/saved_models', 'densenet121_samplying_50_oversampling.pth')
model = DenseNet(pretrained=False)  # Custom DenseNet class (not torchvision)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# Helper function to overlay heatmap on original image
def apply_colormap_on_image(org_img, activation_map, colormap_name='jet'):
    heatmap = cv2.resize(activation_map, (org_img.size[0], org_img.size[1]))  # Resize to match original image
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    org_img = np.array(org_img.convert("RGB"))
    org_img = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)

    overlayed_img = cv2.addWeighted(heatmap_color, 0.5, org_img, 0.5, 0)
    overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlayed_img)

# Helper function to convert PIL Image to base64
def pil_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

# Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    age = request.form.get('age', default=None)
    lesion_location = request.form.get('lesion_location', default=None)
    sex = request.form.get('sex', default=None)

    try:
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'error': 'Invalid image format or corrupted image', 'message': str(e)}), 400

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Define DenseBlock layers to visualize
    denseblock_layers = {
        "denseblock2": model.densenet.features[4],
        "denseblock3": model.densenet.features[6],
        "denseblock4": model.densenet.features[8],
        "denseblock5": model.densenet.features[10]
    }

    # Generate Grad-CAMs for each layer
    gradcam_images = {}
    for layer_name, layer in denseblock_layers.items():
        gradcam = GradCAM(model.densenet, [layer])
        gradcam.clear_hooks()
        cam_output = gradcam.generate_gradcam(input_tensor, class_idx=None)
        overlay_img = apply_colormap_on_image(image, cam_output)
        gradcam_images[layer_name] = pil_to_base64(overlay_img)

    # Predict using the model
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).item()

    diagnosis = "malignant" if probs > 0.5 else "benign"
    confidence = round(probs * 100, 2) if diagnosis == "malignant" else round((1 - probs) * 100, 2)

    return jsonify({
        'diagnosis': diagnosis,
        'confidence_score': confidence,
        'gradcam_images': gradcam_images
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
