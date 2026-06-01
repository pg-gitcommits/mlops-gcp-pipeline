import io
import os
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(image_size: int) -> transforms.Compose:
    """
    Preprocessing pipeline for inference.
    Matches exactly what was used during training.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load a PIL image from raw bytes."""
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert('RGB')


def preprocess_image(image: Image.Image, image_size: int) -> torch.Tensor:
    """
    Preprocess a PIL image for model inference.
    Returns a tensor of shape (1, 3, image_size, image_size).
    """
    transform = get_transform(image_size)
    tensor = transform(image)
    return tensor.unsqueeze(0)