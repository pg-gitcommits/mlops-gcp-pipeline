import os
import glob
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from tqdm import tqdm
from datetime import datetime
import wandb
from src.model import build_model, get_device
from src.dataset import get_dataloaders

# MLflow is optional — fails gracefully if server unavailable
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not installed — skipping MLflow tracking")


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
    """Check MLflow for a completed run with these hyperparameters."""
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