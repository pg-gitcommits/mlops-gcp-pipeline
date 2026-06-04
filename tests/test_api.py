import io
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def create_test_image(size=(32, 32), format='PNG'):
    """Create a simple test image in memory."""
    img = Image.new('RGB', size, color=(128, 128, 128))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=format)
    img_bytes.seek(0)
    return img_bytes


def test_health_endpoint():
    """Test /health returns correct response."""
    with patch('api.main.model', MagicMock()), \
         patch('api.main.device', torch.device('cpu')):
        from api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "checkpoint" in data
        print("✅ Health endpoint working")


def test_classes_endpoint():
    """Test /classes returns list of 10 classes."""
    with patch('api.main.model', MagicMock()), \
         patch('api.main.device', torch.device('cpu')):
        from api.main import app
        client = TestClient(app)
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert len(data["classes"]) == 10
        print("✅ Classes endpoint working")


def test_predict_invalid_file_type():
    """Test /predict rejects non-image files."""
    with patch('api.main.model', MagicMock()), \
         patch('api.main.device', torch.device('cpu')):
        from api.main import app
        client = TestClient(app)
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400
        print("✅ Invalid file type rejected correctly")


def test_predict_valid_image():
    """Test /predict accepts valid image and returns prediction."""
    mock_model = MagicMock()
    mock_output = torch.zeros(1, 10)
    mock_output[0][0] = 5.0
    mock_model.return_value = mock_output

    with patch('api.main.model', mock_model), \
         patch('api.main.device', torch.device('cpu')):
        from api.main import app
        client = TestClient(app)
        img_bytes = create_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert len(data["probabilities"]) == 10
        print("✅ Predict endpoint working")


if __name__ == '__main__':
    test_health_endpoint()
    test_classes_endpoint()
    test_predict_invalid_file_type()
    test_predict_valid_image()
    print("\n✅ All API tests passed")