import torch
import torch.nn as nn
from torchvision import models


def build_model(architecture, num_classes, pretrained=True):
    """
    Build ResNet-18 with a custom final layer
    for CIFAR-10 10-class classification.
    """
    if architecture == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def get_device():
    """Return GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device