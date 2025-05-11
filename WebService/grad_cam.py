import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import models
from matplotlib import pyplot as plt
import io
import base64
from PIL import Image

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.activations = []

        # Register hooks to capture activations and gradients
        for layer in self.target_layers:
            layer.register_forward_hook(self.save_activation)
            layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def generate_gradcam(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)

        # Backpropagate to get gradients for the class
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Get the activation and gradients of the last convolutional layer
        activation = self.activations[-1]
        gradient = self.gradients[-1]

        # Calculate the weighted sum of activations
        weights = torch.mean(gradient, dim=[2, 3], keepdim=True)
        weighted_activation = weights * activation
        gradcam = torch.sum(weighted_activation, dim=1, keepdim=True)

        # Normalize Grad-CAM
        gradcam = F.relu(gradcam)
        gradcam = gradcam.squeeze().cpu().detach().numpy()
        gradcam = cv2.resize(gradcam, (256, 256))  # Resize to match input image
        gradcam -= gradcam.min()
        gradcam /= gradcam.max()

        return gradcam

    def clear_hooks(self):
        self.gradients = []
        self.activations = []

