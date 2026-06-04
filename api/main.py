import os
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from api.models import PredictionResponse, HealthResponse
from api.utils import load_image_from_bytes, preprocess_image
from src.model import build_model


# Environment variables
MODEL_CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT",
    "checkpoints/resnet18_ep020_vacc83.42_20260601.pt"
)
MODEL_ARCHITECTURE = os.environ.get("MODEL_ARCHITECTURE", "resnet18")
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "10"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "32"))
CLASSES = os.environ.get(
    "CLASSES",
    "airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck"
).split(",")

# Global model variable
model = None
device = None


@asynccontextmanager
@asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Download checkpoint from GCS if path starts with gs://
    checkpoint_path = MODEL_CHECKPOINT
    if MODEL_CHECKPOINT.startswith("gs://"):
        print(f"Downloading checkpoint from GCS: {MODEL_CHECKPOINT}")
        from google.cloud import storage as gcs
        bucket_name = MODEL_CHECKPOINT.replace("gs://", "").split("/")[0]
        blob_path = "/".join(MODEL_CHECKPOINT.replace("gs://", "").split("/")[1:])
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        local_path = f"/tmp/{os.path.basename(MODEL_CHECKPOINT)}"
        blob.download_to_filename(local_path)
        checkpoint_path = local_path
        print(f"Checkpoint downloaded to: {local_path}")

    print(f"Loading model from: {checkpoint_path}")
    model = build_model(
        architecture=MODEL_ARCHITECTURE,
        num_classes=NUM_CLASSES,
        pretrained=False
    ).to(device)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"Model loaded successfully.")

    yield 

    print("Shutting down inference server.")

app = FastAPI(
    title="CIFAR-10 Image Classifier",
    description="ResNet-18 inference API for CIFAR-10 classification",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and model is loaded."""
    return HealthResponse(
        status="healthy",
        model=MODEL_ARCHITECTURE,
        checkpoint=MODEL_CHECKPOINT
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded image.
    Accepts: JPEG, PNG image files
    Returns: predicted class, confidence, and all class probabilities
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Accept JPEG or PNG only."
        )

    # Read and preprocess image
    image_bytes = await file.read()
    image = load_image_from_bytes(image_bytes)
    tensor = preprocess_image(image, image_size=IMAGE_SIZE)
    tensor = tensor.to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = probabilities.max(0)

    # Build response
    predicted_class = CLASSES[predicted_idx.item()]
    confidence_pct = round(confidence.item() * 100, 2)
    all_probs = {
        cls: round(probabilities[i].item() * 100, 2)
        for i, cls in enumerate(CLASSES)
    }

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence_pct,
        probabilities=all_probs
    )


@app.get("/classes")
async def get_classes():
    """Return the list of classes the model can predict."""
    return {"classes": CLASSES}