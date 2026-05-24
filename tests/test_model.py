import torch
import yaml
import pytest
from src.model import build_model, get_device


def load_config(config_path='configs/train_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_resnet18_output_shape():
    """Test ResNet-18 outputs correct shape for 10 classes."""
    model = build_model('resnet18', num_classes=10, pretrained=False)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 10), f"Expected (1, 10) got {output.shape}"
    print("✅ ResNet-18 output shape correct")


def test_model_forward_pass():
    """Test model runs without errors on a batch."""
    model = build_model('resnet18', num_classes=10, pretrained=False)
    model.eval()
    dummy_batch = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_batch)
    assert output.shape == (4, 10), f"Expected (4, 10) got {output.shape}"
    print("✅ Forward pass successful")


def test_config_loads():
    """Test config file loads correctly."""
    config = load_config()
    assert 'model' in config
    assert 'training' in config
    assert 'data' in config
    assert config['model']['num_classes'] == 10
    assert len(config['data']['classes']) == 10
    print("✅ Config loads correctly")


def test_invalid_architecture():
    """Test that invalid architecture raises error."""
    with pytest.raises(ValueError):
        build_model('resnet999', num_classes=10, pretrained=False)
    print("✅ Invalid architecture raises ValueError")


if __name__ == '__main__':
    test_resnet18_output_shape()
    test_model_forward_pass()
    test_config_loads()
    test_invalid_architecture()
    print("\n✅ All tests passed")