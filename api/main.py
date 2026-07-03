import os
import io
import uuid
import torch
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from contextlib import asynccontextmanager
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from dotenv import load_dotenv
from api.models import PredictionResponse, HealthResponse
from api.utils import load_image_from_bytes, preprocess_image
from src.model import build_model

# Load variables from a local .env file, if one exists. Inside the Docker
# container this is usually overridden via --env-file/-e instead, but this
# keeps local (non-Docker) runs of the API consistent with train.py.
load_dotenv()


# Environment variables
# MODEL_CHECKPOINT has no hardcoded fallback — a stale/wrong default here
# means silently loading the wrong model. Fail loudly instead.
MODEL_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT")
MODEL_ARCHITECTURE = os.environ.get("MODEL_ARCHITECTURE", "resnet18")
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "10"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "32"))
CLASSES = os.environ.get(
    "CLASSES",
    "airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck"
).split(",")
GCP_PROJECT = os.environ.get("GCP_PROJECT", "mlops-500119")
GCS_BUCKET = os.environ.get("GCS_BUCKET")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_TABLE = os.environ.get("BQ_TABLE", "predictions")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")

# Max upload size for /predict, in bytes. Default 5MB.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))

# Rate limit for /predict, e.g. "10/minute". Configurable via env var.
PREDICT_RATE_LIMIT = os.environ.get("PREDICT_RATE_LIMIT", "10/minute")

# Global variables
model = None
device = None

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global model, device

    if not MODEL_CHECKPOINT:
        raise RuntimeError(
            "MODEL_CHECKPOINT env var is required — set it to a local "
            "checkpoint path or a gs://bucket/path.pt GCS path."
        )

    # GCS_BUCKET / BQ_DATASET are not fatal if missing — predictions still
    # work without them, only image/prediction logging is skipped. Warn
    # once at startup so this is obvious, rather than discovering it later
    # from a buried "Warning: Failed to save image to GCS" on every request.
    if not GCS_BUCKET:
        print("Warning: GCS_BUCKET not set — prediction images will not be saved.")
    if not BQ_DATASET:
        print("Warning: BQ_DATASET not set — predictions will not be logged to BigQuery.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def save_image_to_gcs(image_bytes: bytes, filename: str) -> str:
    """Save image to GCS and return the GCS path. No-op if GCS_BUCKET unset."""
    if not GCS_BUCKET:
        return ""
    try:
        from google.cloud import storage as gcs
        client = gcs.Client()
        bucket = client.bucket(GCS_BUCKET)
        gcs_path = f"predictions/{filename}"
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(image_bytes, content_type="image/png")
        return f"gs://{GCS_BUCKET}/{gcs_path}"
    except Exception as e:
        print(f"Warning: Failed to save image to GCS: {e}")
        return ""


def log_to_bigquery(predicted_class: str, confidence: float,
                    image_filename: str, image_gcs_path: str):
    """Log prediction to BigQuery. No-op if BQ_DATASET unset."""
    if not BQ_DATASET:
        return
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=GCP_PROJECT)
        table_id = f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
        rows = [{
            "timestamp": datetime.utcnow().isoformat(),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "image_filename": image_filename,
            "image_gcs_path": image_gcs_path,
            "model_version": MODEL_VERSION
        }]
        errors = client.insert_rows_json(table_id, rows)
        if errors:
            print(f"Warning: BigQuery insert errors: {errors}")
        else:
            print(f"Logged prediction to BigQuery: {predicted_class} ({confidence}%)")
    except Exception as e:
        print(f"Warning: Failed to log to BigQuery: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and model is loaded."""
    return HealthResponse(
        status="healthy",
        model=MODEL_ARCHITECTURE,
        checkpoint=MODEL_CHECKPOINT
    )


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit(PREDICT_RATE_LIMIT)
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Predict the class of an uploaded image.
    Accepts: JPEG, PNG image files
    Returns: predicted class, confidence, and all class probabilities
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Accept JPEG or PNG only."
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image too large ({len(image_bytes)} bytes). "
                f"Max allowed: {MAX_UPLOAD_BYTES} bytes."
            )
        )

    image = load_image_from_bytes(image_bytes)
    tensor = preprocess_image(image, image_size=IMAGE_SIZE)
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = probabilities.max(0)

    predicted_class = CLASSES[predicted_idx.item()]
    confidence_pct = round(confidence.item() * 100, 2)
    all_probs = {
        cls: round(probabilities[i].item() * 100, 2)
        for i, cls in enumerate(CLASSES)
    }

    # Generate unique filename
    image_filename = f"{uuid.uuid4()}.png"

    # Save image to GCS
    image_gcs_path = save_image_to_gcs(image_bytes, image_filename)

    # Log to BigQuery
    log_to_bigquery(predicted_class, confidence_pct,
                    image_filename, image_gcs_path)

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence_pct,
        probabilities=all_probs
    )


@app.get("/classes")
async def get_classes():
    """Return the list of classes the model can predict."""
    return {"classes": CLASSES}