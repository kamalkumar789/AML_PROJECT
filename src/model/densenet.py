import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet, self).__init__()
        
        # Load pretrained DenseNet
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # Get input features of last FC layer
        in_features = self.densenet.classifier.in_features
        
        # Replace classifier with a binary classifier (output = 1)
        self.densenet.classifier = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()  # Outputs probability between 0 and 1
        )

    def forward(self, x):
        return self.densenet(x)
