import os
import argparse
import yaml
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from src.model import build_model, get_device
from src.dataset import get_dataloaders

# Files in checkpoints/ that are aliases/pointers rather than individual
# timestamped training runs — these don't follow the
# "{architecture}_ep{...}_vacc{...}_{date}.pt" naming convention that
# architecture inference below relies on, and are always a duplicate of
# one of the other timestamped checkpoints anyway. Skip them here.
NON_RUN_CHECKPOINT_FILES = {"best_model.pt"}


def load_config(config_path='configs/train_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CIFAR-10 classifier')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to config file'
    )
    return parser.parse_args()


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = probabilities.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    avg_loss = running_loss / len(loader)
    accuracy = 100. * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return avg_loss, accuracy, all_preds, all_labels, all_confidences


def plot_confusion_matrix(all_labels, all_preds, classes, checkpoint_name, save_dir):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        title=f'Confusion Matrix\n{checkpoint_name}',
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'confusion_matrix_{checkpoint_name}.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def save_classification_report(all_labels, all_preds, classes,
                                checkpoint_name, save_dir, accuracy, loss):
    report = classification_report(all_labels, all_preds, target_names=classes)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'classification_report_{checkpoint_name}.txt')
    with open(save_path, 'w') as f:
        f.write(f"Checkpoint: {checkpoint_name}\n")
        f.write(f"Val Loss: {loss:.4f} | Val Accuracy: {accuracy:.2f}%\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"Classification report saved to {save_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    classes = config['data']['classes']
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = config['paths']['checkpoint_dir']
    log_dir = config['paths']['log_dir']

    all_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    checkpoints = [f for f in all_checkpoints if f not in NON_RUN_CHECKPOINT_FILES]
    skipped = [f for f in all_checkpoints if f in NON_RUN_CHECKPOINT_FILES]

    if not checkpoints:
        print("No checkpoints found. Run training first.")
        return

    print(f"Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  {cp}")
    if skipped:
        print(f"Skipped (alias, not an individual run): {', '.join(skipped)}")

    _, val_loader = get_dataloaders(config)
    results = []

    for checkpoint_name in checkpoints:
        architecture = checkpoint_name.split('_')[0]
        checkpoint_stem = checkpoint_name.replace('.pt', '')

        print(f"\nEvaluating: {checkpoint_name}")

        model = build_model(
            architecture=architecture,
            num_classes=config['model']['num_classes'],
            pretrained=False
        ).to(device)

        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )

        loss, accuracy, all_preds, all_labels, all_confidences = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Val Loss: {loss:.4f} | Val Acc: {accuracy:.2f}%")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=classes))

        plot_confusion_matrix(
            all_labels, all_preds, classes,
            checkpoint_stem, log_dir
        )

        save_classification_report(
            all_labels, all_preds, classes,
            checkpoint_stem, log_dir, accuracy, loss
        )

        # Save raw predictions CSV
        raw_df = pd.DataFrame({
            'true_class': [classes[l] for l in all_labels],
            'predicted_class': [classes[p] for p in all_preds],
            'confidence': [round(c * 100, 2) for c in all_confidences],
            'correct': [p == l for p, l in zip(all_preds, all_labels)]
        })
        raw_csv_path = os.path.join(
            log_dir, f'raw_predictions_{checkpoint_stem}.csv'
        )
        os.makedirs(log_dir, exist_ok=True)
        raw_df.to_csv(raw_csv_path, index=False)
        print(f"Raw predictions saved to {raw_csv_path}")

        results.append({
            'checkpoint': checkpoint_name,
            'val_loss': loss,
            'val_acc': accuracy
        })

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    results.sort(key=lambda x: x['val_acc'], reverse=True)
    for r in results:
        print(f"{r['checkpoint']:<55} Val Acc: {r['val_acc']:.2f}%")

    best = results[0]
    print(f"\nBest model: {best['checkpoint']}")
    print(f"Best Val Acc: {best['val_acc']:.2f}%")


if __name__ == '__main__':
    main()