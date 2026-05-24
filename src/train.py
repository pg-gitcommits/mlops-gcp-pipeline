import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from tqdm import tqdm
from datetime import datetime
import mlflow
import mlflow.pytorch
import wandb
from src.model import build_model, get_device
from src.dataset import get_dataloaders


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


def experiment_already_done(checkpoint_dir, checkpoint_name):
    """Skip experiment if checkpoint already exists."""
    import glob
    pattern = os.path.join(checkpoint_dir, f"{checkpoint_name}_ep*_*.pt")
    existing = glob.glob(pattern)
    if existing:
        print(f"Skipping {checkpoint_name} — already completed: {existing[0]}")
        return True
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


def run_experiment(config, architecture, lr, batch_size, momentum, weight_decay):
    device = get_device()
    config['training']['batch_size'] = batch_size

    # Build model
    model = build_model(
        architecture=architecture,
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)

    # Data
    train_loader, val_loader = get_dataloaders(config)

    # Loss and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=2, gamma=0.1
    )

    run_name = (f"{architecture}_lr{lr}_bs{batch_size}_"
                f"mom{momentum}_wd{weight_decay}")

    epochs = config['training']['epochs']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    checkpoint_filename = f"{run_name}_ep{epochs}_{timestamp}"

    print(f"\nExperiment: {run_name}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            'architecture': architecture,
            'learning_rate': lr,
            'batch_size': batch_size,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'epochs': epochs,
            'pretrained': config['model']['pretrained'],
            'timestamp': timestamp
        })

        wandb.init(
            project='cifar10-classification',
            name=f"{run_name}_ep{epochs}_{timestamp}",
            config={
                'architecture': architecture,
                'learning_rate': lr,
                'batch_size': batch_size,
                'momentum': momentum,
                'weight_decay': weight_decay,
                'epochs': epochs,
                'pretrained': config['model']['pretrained'],
                'timestamp': timestamp
            }
        )

        best_val_acc = 0.0

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

            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, step=epoch)

            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch': epoch + 1
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
                checkpoint_path = os.path.join(
                    config['paths']['checkpoint_dir'],
                    f"{checkpoint_filename}.pt"
                )
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_metric('best_val_acc', best_val_acc)
                print(f"Saved best model: {best_val_acc:.2f}%")

        mlflow.pytorch.log_model(model, artifact_path='model')
        mlflow.log_metric('final_best_val_acc', best_val_acc)
        wandb.finish()

    return best_val_acc, checkpoint_filename


def main():
    args = parse_args()
    config = load_config(args.config)

    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('cifar10-classification')

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

        if experiment_already_done(config['paths']['checkpoint_dir'], run_name):
            continue

        best_val_acc, checkpoint_filename = run_experiment(
            config=config,
            architecture=architecture,
            lr=lr,
            batch_size=batch_size,
            momentum=momentum,
            weight_decay=weight_decay
        )
        results.append({
            'architecture': architecture,
            'lr': lr,
            'batch_size': batch_size,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'best_val_acc': best_val_acc,
            'checkpoint': checkpoint_filename
        })

    print("\n=== Grid Search Results ===")
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    for r in results:
        print(f"{r['architecture']} | lr={r['lr']} | "
              f"batch={r['batch_size']} | "
              f"mom={r['momentum']} | wd={r['weight_decay']} | "
              f"Best Val Acc: {r['best_val_acc']:.2f}% | "
              f"Checkpoint: {r['checkpoint']}")


if __name__ == '__main__':
    main()