import torch
import yaml
import pytest
from src.model import build_model, get_device


def load_config(config_path='configs/train_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_model_forward_pass():
    """Test model runs without errors on a batch."""
    model = build_model('resnet18', num_classes=10, pretrained=False)
    model.eval()
    dummy_batch = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_batch)
    assert output.shape == (4, 10), f"Expected (4, 10) got {output.shape}"
    print("✅ Forward pass successful")


def test_resnet18_output_shape_at_native_resolution():
    """
    Test ResNet-18 works at 32x32 — CIFAR-10's actual native resolution,
    matching what the real training/inference pipeline uses (IMAGE_SIZE
    env var, train_config.yaml's image_size). The other shape tests use
    224x224 (ImageNet-standard) which the architecture also handles fine,
    but never actually exercises the real production input size.
    """
    model = build_model('resnet18', num_classes=10, pretrained=False)
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 10), f"Expected (1, 10) got {output.shape}"
    print("✅ ResNet-18 output shape correct at native 32x32 resolution")


def test_pretrained_weights_load():
    """
    Test pretrained=True actually downloads and loads ImageNet weights
    without error. train_config.yaml sets pretrained: true, so this is
    the real production path.
    """
    model = build_model('resnet18', num_classes=10, pretrained=True)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 10), f"Expected (1, 10) got {output.shape}"
    print("✅ Pretrained weights load and forward pass succeeds")


def test_gradients_flow():
    """
    Test the model is actually trainable — a backward pass produces
    gradients on the parameters. All other forward-pass tests use
    torch.no_grad(), which is correct for inference checks but would
    never catch a bug that silently freezes weights (e.g. a stray
    requires_grad=False) during real training.
    """
    model = build_model('resnet18', num_classes=10, pretrained=False)
    model.train()
    dummy_input = torch.randn(2, 3, 32, 32)
    dummy_labels = torch.tensor([0, 1])

    output = model(dummy_input)
    loss = torch.nn.functional.cross_entropy(output, dummy_labels)
    loss.backward()

    # Check at least the final layer actually received gradients.
    assert model.fc.weight.grad is not None, "No gradient reached model.fc"
    assert not torch.all(model.fc.weight.grad == 0), "Gradient on model.fc is all zero"
    print("✅ Gradients flow correctly on backward pass")


def test_get_device_returns_valid_device():
    """
    Test get_device() returns a usable torch.device. Doesn't assert which
    one.
    """
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ('cuda', 'cpu')
    print(f"✅ get_device() returned a valid device: {device.type}")


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
    test_model_forward_pass()
    test_resnet18_output_shape_at_native_resolution()
    test_pretrained_weights_load()
    test_gradients_flow()
    test_get_device_returns_valid_device()
    test_config_loads()
    test_invalid_architecture()
    print("\n✅ All tests passed")