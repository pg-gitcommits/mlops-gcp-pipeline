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


def experiment_already_done(experiment_name, run_name):
    """Check MLflow for a completed run with these hyperparameters."""
    try:
        client = mlflow.tracking.MlflowClient('http://localhost:5000')
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
                   weight_decay, experiment_name):
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

    # Scheduler
    step_size = config['training'].get('scheduler_step_size', 7)
    gamma = config['training'].get('scheduler_gamma', 0.1)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    # Run name — hyperparameters only, val_acc added after training
    run_name = (f"{architecture}_lr{lr}_bs{batch_size}_"
                f"mom{momentum}_wd{weight_decay}")

    timestamp = datetime.now().strftime('%Y%m%d')
    epochs = config['training']['epochs']

    print(f"\nExperiment: {run_name}")

    # Temporary checkpoint path — renamed after training
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    temp_checkpoint_path = os.path.join(
        config['paths']['checkpoint_dir'],
        f"temp_{run_name}.pt"
    )

    with mlflow.start_run(run_name=run_name) as run:
        # Log hyperparameters
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

        # W&B run
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
                'scheduler_step_size': step_size,
                'scheduler_gamma': gamma
            }
        )

        best_val_acc = 0.0

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion=nn.CrossEntropyLoss(),
                optimizer=optimizer, device=device
            )
            val_loss, val_acc = validate(
                model, val_loader,
                criterion=nn.CrossEntropyLoss(), device=device
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
                torch.save(model.state_dict(), temp_checkpoint_path)
                print(f"Saved best model: {best_val_acc:.2f}%")

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

        # Log final metrics and tags to MLflow
        mlflow.log_metric('final_best_val_acc', best_val_acc)
        mlflow.log_artifact(final_checkpoint_path)

        # Update MLflow run with outcome tags
        mlflow.set_tag('best_val_acc', f"{best_val_acc:.2f}")
        mlflow.set_tag('dataset', config['data']['dataset'])
        mlflow.set_tag('date', timestamp)
        mlflow.set_tag('checkpoint', final_checkpoint_name)

        # Update W&B
        wandb.log({'best_val_acc': best_val_acc})
        wandb.finish()

    return best_val_acc, final_checkpoint_name


def main():
    args = parse_args()
    config = load_config(args.config)

    experiment_name = f"{config['data']['dataset']}-classification"

    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment(experiment_name)

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
            experiment_name=experiment_name
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