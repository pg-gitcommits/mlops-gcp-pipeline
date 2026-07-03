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


def test_predict_oversized_image():
    """
    Test /predict rejects images larger than MAX_UPLOAD_BYTES with 413.
    Patches MAX_UPLOAD_BYTES down to a tiny value rather than generating a
    real multi-megabyte image — the size check happens on raw bytes before
    the image is decoded, so this is a valid, fast way to trigger it.
    """
    with patch('api.main.model', MagicMock()), \
         patch('api.main.device', torch.device('cpu')), \
         patch('api.main.MAX_UPLOAD_BYTES', 100):
        from api.main import app
        client = TestClient(app)
        oversized_bytes = io.BytesIO(b"x" * 1000)  # 1000 bytes > 100 byte limit
        response = client.post(
            "/predict",
            files={"file": ("test.png", oversized_bytes, "image/png")}
        )
        assert response.status_code == 413
        print("✅ Oversized image rejected correctly")


def test_predict_rate_limit_exceeded():
    """
    Test /predict returns 429 once PREDICT_RATE_LIMIT is exceeded. Fires
    more requests than any reasonable configured limit would allow within
    this test run, and asserts a 429 shows up — rather than asserting an
    exact request count, since the FastAPI app (and its rate limiter
    state) is a module-level singleton shared across all tests in this
    file, so exact counts would be order-dependent and fragile.
    """
    mock_model = MagicMock()
    mock_output = torch.zeros(1, 10)
    mock_output[0][0] = 5.0
    mock_model.return_value = mock_output

    with patch('api.main.model', mock_model), \
         patch('api.main.device', torch.device('cpu')):
        from api.main import app
        client = TestClient(app)
        statuses = []
        for _ in range(20):
            img_bytes = create_test_image()
            response = client.post(
                "/predict",
                files={"file": ("test.png", img_bytes, "image/png")}
            )
            statuses.append(response.status_code)

        assert 429 in statuses, f"Expected a 429 among 20 rapid requests, got: {statuses}"
        print("✅ Rate limit correctly enforced")


def test_predict_missing_file():
    """Test /predict returns a validation error when no file is provided."""
    with patch('api.main.model', MagicMock()), \
         patch('api.main.device', torch.device('cpu')):
        from api.main import app
        client = TestClient(app)
        response = client.post("/predict")
        assert response.status_code == 422
        print("✅ Missing file correctly rejected")


def test_health_endpoint_reflects_correct_checkpoint():
    """
    Test /health's checkpoint field actually reflects MODEL_CHECKPOINT,
    not just that the field exists. Patches a known value and confirms
    the response matches it exactly.
    """
    with patch('api.main.model', MagicMock()), \
         patch('api.main.device', torch.device('cpu')), \
         patch('api.main.MODEL_CHECKPOINT', 'gs://test-bucket/checkpoints/best_model.pt'):
        from api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["checkpoint"] == 'gs://test-bucket/checkpoints/best_model.pt'
        print("✅ Health endpoint reflects correct checkpoint")


if __name__ == '__main__':
    test_health_endpoint()
    test_classes_endpoint()
    test_predict_invalid_file_type()
    test_predict_valid_image()
    test_predict_oversized_image()
    test_predict_rate_limit_exceeded()
    test_predict_missing_file()
    test_health_endpoint_reflects_correct_checkpoint()
    print("\n✅ All API tests passed")