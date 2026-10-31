"""
train.py - Multi-Seed Training Engine & Fair Benchmarking
Group 13: Aldrich Taqi Marvel, Darren Christian Rapang, Johannes Christian Tahun
Course: Deep Learning Final Project
"""

import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_dataloaders, CLASS_NAMES
from models import BaselineCNN, get_method_a_resnet18, get_method_b_se_resnet18, count_parameters

SEEDS = [42, 123, 999]
EPOCHS = 12
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4

def set_seed(seed):
    """Guarantees full reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.squeeze().long().to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
        
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.squeeze().long().to(device)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)
        
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    epoch_loss = total_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    
    return epoch_loss, acc, macro_f1, all_preds, all_targets

def run_single_experiment(model_name, seed, device, output_dir):
    set_seed(seed)
    print(f"\n[Training] {model_name} | Seed: {seed}")
    
    use_aug = (model_name == "Method_B_SE_ResNet18")
    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(
        batch_size=BATCH_SIZE, use_augmentation=use_aug
    )
    
    if model_name == "Baseline_CNN":
        model = BaselineCNN().to(device)
    elif model_name == "Method_A_ResNet18":
        model = get_method_a_resnet18().to(device)
    elif model_name == "Method_B_SE_ResNet18":
        model = get_method_b_se_resnet18().to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_val_f1 = -1.0
    best_model_path = os.path.join(output_dir, f"{model_name}_seed{seed}_best.pth")
    
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            
    train_duration = time.time() - start_time
    
    # Evaluate best checkpoint on Test Set
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_f1, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    
    print(f"[Done] {model_name} (Seed {seed}) -> Test Acc: {test_acc*100:.2f}%, Test Macro-F1: {test_f1*100:.2f}% (Time: {train_duration:.1f}s)")
    
    return {
        'model': model_name,
        'seed': seed,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'duration_s': train_duration,
        'params': count_parameters(model),
        'history': history,
        'test_preds': test_preds,
        'test_targets': test_targets
    }
