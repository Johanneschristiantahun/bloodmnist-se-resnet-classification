"""
evaluate.py - Visualizations, Statistical Tables & Error Analysis
Group 13: Aldrich Taqi Marvel, Darren Christian Rapang, Johannes Christian Tahun
Course: Deep Learning Final Project
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import medmnist
from medmnist import INFO

DATA_FLAG = 'bloodmnist'
info = INFO[DATA_FLAG]
CLASS_NAMES = list(info['label'].values())

def generate_results_table(all_results, output_dir="./outputs"):
    """
    Summarizes 3-seed results into the required rubric table format (mean ± std).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(all_results)
    
    summary = []
    for model_name, group in df.groupby('model'):
        params = group['params'].iloc[0]
        acc_mean = group['test_acc'].mean() * 100
        acc_std = group['test_acc'].std() * 100
        f1_mean = group['test_f1'].mean() * 100
        f1_std = group['test_f1'].std() * 100
        time_mean = group['duration_s'].mean()
        time_std = group['duration_s'].std()
        
        summary.append({
            'Model': model_name,
            'Params': f"{params:,}",
            'Test Accuracy (%)': f"{acc_mean:.2f} ± {acc_std:.2f}",
            'Test Macro-F1 (%)': f"{f1_mean:.2f} ± {f1_std:.2f}",
            'Avg Train Time (s)': f"{time_mean:.1f} ± {time_std:.1f}"
        })
        
    summary_df = pd.DataFrame(summary)
    csv_path = os.path.join(output_dir, "results_table.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n[Results] Summary Table saved to {csv_path}:\n")
    print(summary_df.to_string(index=False))
    return summary_df

def plot_learning_curves(all_results, output_dir="./outputs"):
    """
    Plots training and validation loss per epoch for all 3 models.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    
    colors = {'Baseline_CNN': '#1f77b4', 'Method_A_ResNet18': '#ff7f0e', 'Method_B_SE_ResNet18': '#2ca02c'}
    
    for res in all_results:
        # Plot seed 42 as representative curve
        if res['seed'] == 42:
            model = res['model']
            epochs = range(1, len(res['history']['train_loss']) + 1)
            plt.plot(epochs, res['history']['train_loss'], label=f"{model} (Train)", color=colors[model], linestyle='--')
            plt.plot(epochs, res['history']['val_loss'], label=f"{model} (Val)", color=colors[model], linewidth=2)
            
    plt.title("Learning Curves (Train vs Validation Loss per Epoch)", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "learning_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Plot] Learning curves saved to {plot_path}")

def plot_confusion_matrix(best_res, output_dir="./outputs"):
    """
    Generates high-resolution confusion matrix heatmap for the best performing model.
    """
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(best_res['test_targets'], best_res['test_preds'], normalize='true')
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Normalized Confusion Matrix - {best_res['model']}", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Plot] Confusion matrix saved to {cm_path}")

def plot_failure_cases(best_res, test_dataset, output_dir="./outputs", num_cases=3):
    """
    Extracts and visualizes test samples where the best model failed,
    along with the ground truth and predicted labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    preds = best_res['test_preds']
    targets = best_res['test_targets']
    
    fail_indices = np.where(preds != targets)[0]
    selected_indices = fail_indices[:num_cases]
    
    fig, axes = plt.subplots(1, num_cases, figsize=(12, 4))
    
    for i, idx in enumerate(selected_indices):
        raw_img, _ = test_dataset[idx] # Raw PIL or Tensor
        if hasattr(raw_img, 'permute'):
            img_np = raw_img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        else:
            img_np = np.array(raw_img)
            
        true_lbl = CLASS_NAMES[targets[idx]]
        pred_lbl = CLASS_NAMES[preds[idx]]
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"Sample #{idx}\nTrue: {true_lbl}\nPred: {pred_lbl}", fontsize=11, color='red')
        axes[i].axis('off')
        
    plt.suptitle("Failure Cases Analysis (Best Performing Model)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fc_path = os.path.join(output_dir, "failure_cases.png")
    plt.savefig(fc_path, dpi=300)
    plt.close()
    print(f"[Plot] Failure cases saved to {fc_path}")
