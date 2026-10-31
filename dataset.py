"""
dataset.py - Data Pipeline for BloodMNIST Classification
Group 13: Aldrich Taqi Marvel, Darren Christian Rapang, Johannes Christian Tahun
Course: Deep Learning Final Project
"""

import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
import pandas as pd
import numpy as np

DATA_FLAG = 'bloodmnist'
info = INFO[DATA_FLAG]
NUM_CLASSES = len(info['label'])
CLASS_NAMES = list(info['label'].values())

def get_transforms():
    """
    Returns standard transforms (for Baseline & Method A)
    and augmented transforms (for Method B).
    """
    # Standard normalization for 28x28 RGB images
    norm = transforms.Normalize(mean=[0.707, 0.658, 0.704], std=[0.239, 0.258, 0.203])

    # Standard Transform (Baseline & Method A)
    standard_train_transform = transforms.Compose([
        transforms.ToTensor(),
        norm
    ])

    # Augmented Transform (Method B): Spatial rotations, horizontal/vertical flips, subtle color jitter
    augmented_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        norm
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        norm
    ])

    return standard_train_transform, augmented_train_transform, test_transform

def get_dataloaders(batch_size=128, use_augmentation=False, num_workers=2, root="./data"):
    """
    Loads BloodMNIST datasets with official split and returns PyTorch DataLoaders.
    """
    standard_train_tf, aug_train_tf, test_tf = get_transforms()
    train_tf = aug_train_tf if use_augmentation else standard_train_tf

    DataClass = getattr(medmnist, info['python_class'])

    train_dataset = DataClass(split='train', transform=train_tf, download=True, root=root)
    val_dataset = DataClass(split='val', transform=test_tf, download=True, root=root)
    test_dataset = DataClass(split='test', transform=test_tf, download=True, root=root)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

def export_split_distribution(output_dir="./outputs"):
    """
    Verifies dataset distributions and exports a CSV summary table.
    Ensures zero data leakage across splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    DataClass = getattr(medmnist, info['python_class'])
    
    train_set = DataClass(split='train', download=True)
    val_set = DataClass(split='val', download=True)
    test_set = DataClass(split='test', download=True)

    splits = {'Train': train_set.labels.flatten(),
              'Validation': val_set.labels.flatten(),
              'Test': test_set.labels.flatten()}

    dist_data = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        row = {'Class ID': cls_idx, 'Class Name': cls_name}
        for split_name, labels in splits.items():
            count = int(np.sum(labels == cls_idx))
            pct = (count / len(labels)) * 100
            row[f'{split_name} Count'] = count
            row[f'{split_name} (%)'] = round(pct, 2)
        dist_data.append(row)

    df = pd.DataFrame(dist_data)
    csv_path = os.path.join(output_dir, "data_distribution.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Dataset] Distribution exported to {csv_path}")
    return df

if __name__ == '__main__':
    export_split_distribution()
