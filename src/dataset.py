import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import yaml


def load_config(config_path='configs/train_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_transforms(config, split='train'):
    aug = config['augmentation']
    image_size = config['data']['image_size']

    if split == 'train':
        transform_list = [
            transforms.Resize((image_size, image_size)),
        ]
        if aug.get('random_horizontal_flip'):
            transform_list.append(transforms.RandomHorizontalFlip())
        if aug.get('random_rotation'):
            transform_list.append(transforms.RandomRotation(aug['random_rotation']))
        if aug.get('color_jitter'):
            transform_list.append(transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ))
    else:
        transform_list = [
            transforms.Resize((image_size, image_size)),
        ]

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transforms.Compose(transform_list)


def get_dataloaders(config):
    train_transform = get_transforms(config, split='train')
    val_transform = get_transforms(config, split='val')

    # Download CIFAR-10 automatically
    full_train = datasets.CIFAR10(
        root='data/',
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root='data/',
        train=False,
        download=True,
        transform=val_transform
    )

    # Split training into train and validation
    val_split = config['data']['val_split']
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size

    torch.manual_seed(config['training']['random_seed'])
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training'].get('batch_size', 64),
        shuffle=True,
        num_workers=config['data']['num_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training'].get('batch_size', 64),
        shuffle=False,
        num_workers=config['data']['num_workers']
    )

    return train_loader, val_loader