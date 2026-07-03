import os
import json
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from tqdm import tqdm
from datetime import datetime, timezone
import wandb
from dotenv import load_dotenv
from src.model import build_model, get_device
from src.dataset import get_dataloaders

# Load variables from a local .env file into the process environment, if
# one exists. Without this, GCS_BUCKET/GCP_PROJECT etc. defined in .env
# are invisible to this script unless manually exported in the shell
# first — that gap is what caused GCS promotion to be silently skipped
# during an earlier run, despite .env having the correct values.
load_dotenv()

# MLflow is optional — fails gracefully if server unavailable
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not installed — skipping MLflow tracking")

# google-cloud-storage is optional at import time — promotion to GCS is
# skipped gracefully (with a clear warning) if the library or bucket isn't
# available, same pattern as MLflow/W&B elsewhere in this file.
try:
    from google.cloud import storage as gcs
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("google-cloud-storage not installed — skipping GCS promotion")

GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCS_BUCKET = os.environ.get("GCS_BUCKET")

BEST_MODEL_FILENAME = "best_model.pt"
BEST_MODEL_META_FILENAME = "best_model.json"
PROMOTION_HISTORY_FILENAME = "promotion_history.jsonl"


def load_config(config_path='configs/train_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 classifier')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to config file'
    )
    return parser.parse_args()


def setup_mlflow(experiment_name):
    """Set up MLflow tracking. Returns True if successful."""
    if not MLFLOW_AVAILABLE:
        return False
    try:
        mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)
        print(f"MLflow tracking enabled at {mlflow_uri}")
        return True
    except Exception as e:
        print(f"MLflow not available: {e}. Continuing without MLflow.")
        return False


def experiment_already_done(experiment_name, run_name):
    """grep dotenv requirements.txt
    Check MLflow for a completed run with these hyperparameters.

    If FORCE_RETRAIN=true is set in the environment, this always returns
    False — used by Airflow to guarantee training actually runs on every
    trigger (e.g. for scheduled retraining on new data), bypassing the
    skip-if-already-tried behavior that's useful for manual hyperparameter
    sweeps but wrong for scheduled retraining.
    """
    if os.environ.get('FORCE_RETRAIN', 'false').lower() == 'true':
        return False
    if not MLFLOW_AVAILABLE:
        return False
    try:
        client = mlflow.tracking.MlflowClient(
            os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        )
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return False
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.`mlflow.runName` = '{run_name}' and status = 'FINISHED'"
        )
        if runs:
            print(f"Skipping {run_name} — completed run found in MLflow.")
            return True
        return False
    except Exception:
        return False


def get_gcs_bucket():
    """
    Return a GCS bucket handle.

    Raises RuntimeError if GCP_PROJECT or GCS_BUCKET env vars are not set —
    this is a configuration error and should fail loudly, not silently
    skip promotion. (Network/auth failures while actually talking to GCS
    are still caught and reported separately, below.)
    """
    if not GCS_AVAILABLE:
        return None
    if not GCP_PROJECT:
        raise RuntimeError("GCP_PROJECT environment variable is not set.")
    if not GCS_BUCKET:
        raise RuntimeError("GCS_BUCKET environment variable is not set.")
    try:
        client = gcs.Client(project=GCP_PROJECT)
        return client.bucket(GCS_BUCKET)
    except Exception as e:
        print(f"GCS not available: {e}. Continuing without GCS promotion.")
        return None



def read_local_best_meta(checkpoint_dir):
    """
    Read best_model.json from local disk. Returns a dict, or None if it
    doesn't exist yet (first ever promotion) or can't be read.
    Mirrors read_current_best_meta(), but for the local filesystem rather
    than GCS — kept independent so local promotion works even when GCS is
    not configured.
    """
    local_meta_path = os.path.join(checkpoint_dir, BEST_MODEL_META_FILENAME)
    if not os.path.exists(local_meta_path):
        return None
    try:
        with open(local_meta_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not read local best_model.json: {e}")
        return None


def append_local_promotion_history(checkpoint_dir, entry):
    """Append one JSON line to the local promotion_history.jsonl."""
    local_history_path = os.path.join(checkpoint_dir, PROMOTION_HISTORY_FILENAME)
    try:
        with open(local_history_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Warning: failed to append local promotion history: {e}")


def promote_if_better_local(local_checkpoint_path, checkpoint_name, val_acc,
                             run_name, checkpoint_dir):
    """
    Local-disk equivalent of promote_if_better(). Compares this run's
    val_acc against checkpoints/best_model.json on local disk (not GCS,
    and not inferred from a filename), and if better, copies this
    checkpoint to checkpoints/best_model.pt and updates the local
    best_model.json + promotion_history.jsonl.

    Runs independently of GCS availability — this is what makes
    best_model.pt usable for local testing even when GCS_BUCKET isn't
    configured (e.g. running train.py directly without sourcing .env).
    """
    current_best = read_local_best_meta(checkpoint_dir)
    current_best_acc = current_best['val_acc'] if current_best else -1.0

    if val_acc <= current_best_acc:
        print(
            f"Run val_acc {val_acc:.2f}% does not beat current local best "
            f"{current_best_acc:.2f}% — not promoting locally."
        )
        return

    timestamp = datetime.now(timezone.utc).isoformat()

    if current_best is not None:
        append_local_promotion_history(checkpoint_dir, {
            **current_best,
            "demoted_at": timestamp
        })

    try:
        import shutil
        local_best_path = os.path.join(checkpoint_dir, BEST_MODEL_FILENAME)
        shutil.copyfile(local_checkpoint_path, local_best_path)

        meta = {
            "checkpoint": checkpoint_name,
            "run_name": run_name,
            "val_acc": val_acc,
            "promoted_at": timestamp
        }
        local_meta_path = os.path.join(checkpoint_dir, BEST_MODEL_META_FILENAME)
        with open(local_meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(
            f"Promoted new local best model: {checkpoint_name} "
            f"(val_acc {val_acc:.2f}%, previous best {current_best_acc:.2f}%)"
        )
    except Exception as e:
        print(f"Warning: failed to promote new best model locally: {e}")


def read_current_best_meta(bucket):
    """
    Read best_model.json from GCS. Returns a dict, or None if it doesn't
    exist yet (e.g. first ever promotion) or GCS is unavailable.
    """
    if bucket is None:
        return None
    try:
        blob = bucket.blob(f"checkpoints/{BEST_MODEL_META_FILENAME}")
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text())
    except Exception as e:
        print(f"Could not read current best_model.json from GCS: {e}")
        return None


def append_promotion_history(bucket, entry):
    """Append one JSON line to promotion_history.jsonl in GCS."""
    if bucket is None:
        return
    try:
        blob = bucket.blob(f"checkpoints/{PROMOTION_HISTORY_FILENAME}")
        existing = blob.download_as_text() if blob.exists() else ""
        updated = existing + json.dumps(entry) + "\n"
        blob.upload_from_string(updated)
    except Exception as e:
        print(f"Warning: failed to append promotion history to GCS: {e}")


def promote_if_better(local_checkpoint_path, checkpoint_name, val_acc,
                       run_name, checkpoint_dir):
    """
    Compare this run's val_acc against the currently promoted best model
    (recorded in GCS's best_model.json, not inferred from a filename).
    If this run is better, upload it as the new best_model.pt, update
    best_model.json, and append the outgoing model's info to the
    promotion history log (for rollback).

    Always uploads the raw timestamped checkpoint to GCS regardless of
    whether it's promoted, so every run is durably stored.

    Note: this is the GCS promotion path. Local promotion is handled
    separately by promote_if_better_local(), called alongside this in
    run_experiment(), so a local best_model.pt always exists regardless
    of whether GCS is configured.
    """
    bucket = get_gcs_bucket()
    if bucket is None:
        print("Skipping GCS checkpoint upload/promotion (no bucket available).")
        return

    # Always upload the raw checkpoint for durability, promoted or not.
    try:
        raw_blob = bucket.blob(f"checkpoints/{checkpoint_name}")
        raw_blob.upload_from_filename(local_checkpoint_path)
        print(f"Uploaded checkpoint to GCS: gs://{GCS_BUCKET}/checkpoints/{checkpoint_name}")
    except Exception as e:
        print(f"Warning: failed to upload checkpoint to GCS: {e}")
        return

    current_best = read_current_best_meta(bucket)
    current_best_acc = current_best['val_acc'] if current_best else -1.0

    if val_acc <= current_best_acc:
        print(
            f"Run val_acc {val_acc:.2f}% does not beat current best "
            f"{current_best_acc:.2f}% — not promoting."
        )
        return

    # Promote: this run becomes the new best_model.pt
    timestamp = datetime.now(timezone.utc).isoformat()

    if current_best is not None:
        append_promotion_history(bucket, {
            **current_best,
            "demoted_at": timestamp
        })

    try:
        best_blob = bucket.blob(f"checkpoints/{BEST_MODEL_FILENAME}")
        best_blob.upload_from_filename(local_checkpoint_path)

        meta = {
            "checkpoint": checkpoint_name,
            "run_name": run_name,
            "val_acc": val_acc,
            "promoted_at": timestamp
        }

        meta_blob = bucket.blob(f"checkpoints/{BEST_MODEL_META_FILENAME}")
        meta_blob.upload_from_string(json.dumps(meta, indent=2))

        print(
            f"Promoted new best model: {checkpoint_name} "
            f"(val_acc {val_acc:.2f}%, previous best {current_best_acc:.2f}%)"
        )
    except Exception as e:
        print(f"Warning: failed to promote new best model in GCS: {e}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total


def run_experiment(config, architecture, lr, batch_size, momentum,
                   weight_decay, experiment_name, mlflow_enabled):
    device = get_device()
    config['training']['batch_size'] = batch_size

    model = build_model(
        architecture=architecture,
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)

    train_loader, val_loader = get_dataloaders(config)

    optimizer_name = config['training'].get('optimizer', 'sgd').lower()
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    step_size = config['training'].get('scheduler_step_size', 7)
    gamma = config['training'].get('scheduler_gamma', 0.1)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    run_name = (f"{architecture}_lr{lr}_bs{batch_size}_"
                f"mom{momentum}_wd{weight_decay}")

    timestamp = datetime.now().strftime('%Y%m%d')
    epochs = config['training']['epochs']

    print(f"\nExperiment: {run_name}")

    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    temp_checkpoint_path = os.path.join(
        config['paths']['checkpoint_dir'],
        f"temp_{run_name}.pt"
    )

    # Start MLflow run if available
    mlflow_run = None
    if mlflow_enabled:
        try:
            mlflow_run = mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                'architecture': architecture,
                'learning_rate': lr,
                'batch_size': batch_size,
                'momentum': momentum,
                'weight_decay': weight_decay,
                'epochs': epochs,
                'pretrained': config['model']['pretrained'],
                'optimizer': optimizer_name,
                'scheduler_step_size': step_size,
                'scheduler_gamma': gamma
            })
        except Exception as e:
            print(f"MLflow run start failed: {e}")
            mlflow_run = None

    # W&B init
    try:
        wandb.init(
            project='cifar10-classification',
            name=run_name,
            config={
                'architecture': architecture,
                'learning_rate': lr,
                'batch_size': batch_size,
                'momentum': momentum,
                'weight_decay': weight_decay,
                'epochs': epochs,
                'pretrained': config['model']['pretrained'],
                'optimizer': optimizer_name,
            }
        )
        wandb_enabled = True
    except Exception as e:
        print(f"W&B init failed: {e}")
        wandb_enabled = False

    best_val_acc = 0.0
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f} "
              f"Val Acc: {val_acc:.2f}%")

        if mlflow_run:
            try:
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, step=epoch)
            except Exception:
                pass

        if wandb_enabled:
            try:
                wandb.log({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'epoch': epoch + 1
                })
            except Exception:
                pass

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), temp_checkpoint_path)
            print(f"Saved best model: {best_val_acc:.2f}%")

            if mlflow_run:
                try:
                    mlflow.log_metric('best_val_acc', best_val_acc)
                except Exception:
                    pass

    # Rename checkpoint with actual best val_acc
    final_checkpoint_name = (
        f"{architecture}_ep{epochs:03d}_"
        f"vacc{best_val_acc:.2f}_{timestamp}.pt"
    )
    final_checkpoint_path = os.path.join(
        config['paths']['checkpoint_dir'],
        final_checkpoint_name
    )
    os.rename(temp_checkpoint_path, final_checkpoint_path)
    print(f"Checkpoint saved as: {final_checkpoint_name}")

    if mlflow_run:
        try:
            mlflow.log_metric('final_best_val_acc', best_val_acc)
            mlflow.log_artifact(final_checkpoint_path)
            mlflow.set_tag('best_val_acc', f"{best_val_acc:.2f}")
            mlflow.set_tag('dataset', config['data']['dataset'])
            mlflow.set_tag('date', timestamp)
            mlflow.set_tag('checkpoint', final_checkpoint_name)
            mlflow.end_run()
        except Exception as e:
            print(f"MLflow end run failed: {e}")

    if wandb_enabled:
        try:
            wandb.log({'best_val_acc': best_val_acc})
            wandb.finish()
        except Exception:
            pass

    # Promote locally first — always runs, independent of GCS, so
    # checkpoints/best_model.pt exists for local testing even when
    # GCS_BUCKET isn't configured in the current shell.
    promote_if_better_local(
        local_checkpoint_path=final_checkpoint_path,
        checkpoint_name=final_checkpoint_name,
        val_acc=best_val_acc,
        run_name=run_name,
        checkpoint_dir=config['paths']['checkpoint_dir']
    )

    # Then upload this run's checkpoint to GCS, and promote it to
    # best_model.pt there too if it beats the currently recorded best.
    # This is the step that makes "what's currently live" a fact recorded
    # in GCS, not a filename a human has to remember to update — works
    # identically whether this script is run by hand or triggered by
    # Airflow, as long as GCS_BUCKET is set in the environment.
    promote_if_better(
        local_checkpoint_path=final_checkpoint_path,
        checkpoint_name=final_checkpoint_name,
        val_acc=best_val_acc,
        run_name=run_name,
        checkpoint_dir=config['paths']['checkpoint_dir']
    )

    return best_val_acc, final_checkpoint_name


def main():
    args = parse_args()
    config = load_config(args.config)

    experiment_name = f"{config['data']['dataset']}-classification"

    mlflow_enabled = setup_mlflow(experiment_name)

    gs = config['grid_search']
    architectures = [config['model']['architecture']]
    results = []

    for architecture, lr, batch_size, momentum, weight_decay in product(
        architectures,
        gs['learning_rate'],
        gs['batch_size'],
        gs['momentum'],
        gs['weight_decay']
    ):
        run_name = (f"{architecture}_lr{lr}_bs{batch_size}_"
                    f"mom{momentum}_wd{weight_decay}")

        if experiment_already_done(experiment_name, run_name):
            continue

        best_val_acc, checkpoint_name = run_experiment(
            config=config,
            architecture=architecture,
            lr=lr,
            batch_size=batch_size,
            momentum=momentum,
            weight_decay=weight_decay,
            experiment_name=experiment_name,
            mlflow_enabled=mlflow_enabled
        )
        results.append({
            'architecture': architecture,
            'lr': lr,
            'batch_size': batch_size,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'best_val_acc': best_val_acc,
            'checkpoint': checkpoint_name
        })

    print("\n=== Grid Search Results ===")
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    for r in results:
        print(f"{r['architecture']} | lr={r['lr']} | "
              f"batch={r['batch_size']} | "
              f"Best Val Acc: {r['best_val_acc']:.2f}% | "
              f"Checkpoint: {r['checkpoint']}")


if __name__ == '__main__':
    main()