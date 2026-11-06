# Deep Learning Final Project — Group 13
**Course:** Deep Learning (Department of Artificial Intelligence, Faculty of AI and Robotics - FAIR, Universitas Pelita Harapan)  
**Deadline:** 1 November 2026, 23:59 GMT+7  
**Platform:** PCL Learning Platform

---

## 👥 Team Information (Group 13)
- **Institution:** Universitas Pelita Harapan (UPH)
- **Faculty & College:** Faculty of AI and Robotics (FAIR), College of Emerging Sciences

| Member Name | Student ID (NIM) | Email | Department / Program | Role & Core Contributions |
| :--- | :---: | :--- | :--- | :--- |
| **Aldrich Taqi Marvel** | `01881250022` | `01881250022@student.uph.edu` | Artificial Intelligence | Data Pipeline, Baseline CNN Architecture, Parameter Progression |
| **Darren Christian Rapang** | `01881250021` | `01881250021@student.uph.edu` | Artificial Intelligence | Pretrained ResNet-18 Transfer Learning (Method A), Evaluation Metrics |
| **Johannes Christian Tahun** | `01881250050` | `01881250050@student.uph.edu` | Artificial Intelligence | Squeeze-and-Excitation Attention (Method B), Multi-Seed Validation, Paper Lead |

---

## 🎯 1. Problem Statement & Hypothesis

### Problem Definition
- **Input:** $28 \times 28 \times 3$ RGB microscopic digital images of peripheral blood cells.
- **Output:** Categorical classification across 8 cytological blood cell types:
  1. Basophil (`0`)
  2. Eosinophil (`1`)
  3. Erythroblast (`2`)
  4. Immature Granulocyte (`3`)
  5. Lymphocyte (`4`)
  6. Monocyte (`5`)
  7. Neutrophil (`6`)
  8. Platelet (`7`)

### Hypothesis
> *We hypothesize that **Method B (ResNet-18 with Squeeze-and-Excitation Channel Attention and Targeted Augmentations)** will achieve significantly higher Macro-F1 and Top-1 Accuracy than both the Baseline CNN and standard Pretrained ResNet-18 (Method A). This is because channel-wise adaptive recalibration enables the network to emphasize subtle morphological characteristics (such as nuclear lobulation and cytoplasmic granulation) that are critical for distinguishing easily confused leukocyte types (e.g., neutrophils vs. immature granulocytes).*

---

## 🔬 2. Dataset Overview: BloodMNIST
The dataset is acquired from the **MedMNIST v2** benchmark (Yang et al., Nature Scientific Data 2023), based on open-access peripheral blood cell microscopy under the CC BY 4.0 license.

- **Total Samples:** 17,092 images ($28 \times 28 \times 3$).
- **Official Splits (Guaranteed Zero Data Leakage):**
  - **Training:** 11,959 samples ($70\%$)
  - **Validation:** 1,712 samples ($10\%$)
  - **Testing:** 3,421 samples ($20\%$)

The full data distribution table across all 8 classes is automatically exported to `./outputs/data_distribution.csv`.

---

## 🏗️ 3. Comparative Models & Architectures

### 1. Baseline Model (Custom 3-Stage CNN)
Built entirely from scratch without pretrained weights:
- **Block 1:** `Conv2d(3, 32, k=3, p=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`
- **Block 2:** `Conv2d(32, 64, k=3, p=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`
- **Block 3:** `Conv2d(64, 128, k=3, p=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `AdaptiveAvgPool2d(1, 1)`
- **Classifier:** `Flatten` $\rightarrow$ `Linear(128, 64)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(64, 8)`
- **Total Parameters:** **102,248**

#### Baseline Layer-by-Layer Output Shape Progression:
| Layer / Stage | Output Tensor Dimensions |
| :--- | :--- |
| **Input Image** | `[Batch, 3, 28, 28]` |
| **After Block 1** | `[Batch, 32, 14, 14]` |
| **After Block 2** | `[Batch, 64, 7, 7]` |
| **After Block 3** | `[Batch, 128, 1, 1]` |
| **After Flatten** | `[Batch, 128]` |
| **Dense (Hidden)**| `[Batch, 64]` |
| **Output Logits** | `[Batch, 8]` |

### 2. Method A: Transfer Learning with ResNet-18
- Pretrained on ImageNet-1K (`torchvision.models.ResNet18_Weights.DEFAULT`).
- `conv1` adapted to $3 \times 3, s=1, p=1$ (without initial maxpool) to preserve $28 \times 28$ spatial resolution.
- Final fully connected layer replaced with `Linear(512, 8)`.
- **Total Parameters:** **11,172,936**

### 3. Method B (Our Proposed Innovation): SE-ResNet-18 + Augmentation
- Enhanced with **Squeeze-and-Excitation (SE)** channel attention blocks attached to all residual units:
  $$s = \sigma\big(W_2 \cdot \text{ReLU}(W_1 \cdot \text{AdaptiveAvgPool2d}(x))\big)$$
  $$x_{\text{out}} = x \cdot s + \text{Identity}(x)$$
- Domain-specific spatial augmentations: Random Horizontal Flip ($p=0.5$), Random Vertical Flip ($p=0.5$), Random Rotation ($\pm 15^\circ$), and subtle Color Jitter.
- **Total Parameters:** **11,261,000** (Only $+0.78\%$ parameter overhead over Method A!).

---

## ⚡ 4. Reproduction & Execution Guide

### Option A: Run on Google Colab (Recommended, Free GPU)
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `Group13_BloodMNIST_Experiments.ipynb`.
3. In Colab, go to **Runtime $\rightarrow$ Change runtime type $\rightarrow$ Select T4 GPU**.
4. Click **Runtime $\rightarrow$ Run all**.
5. The entire benchmark trains across all 3 seeds in **~12 minutes total**. All figures, tables, and `Group13_Code.zip` will be generated in `./outputs/`.

### Option B: Run Locally via Command Line
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training across all 3 seeds and generate visualizations
python train.py
python evaluate.py
```

---

## 📊 5. Official Results Summary (3 Seeds: 42, 123, 999)

All models are trained with identical budget (12 epochs, batch size 128, AdamW optimizer with $\eta = 10^{-3}$, cosine annealing learning rate schedule, early stopping on validation Macro-F1).

| Model Architecture | Trainable Params | Test Accuracy (%) | Test Macro-F1 (%) | Avg Training Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Custom CNN)** | 102,248 | **93.70 ± 0.34** | **93.00 ± 0.39** | 44.6 ± 5.9 s |
| **Method A (ResNet-18)** | 11,172,936 | **96.80 ± 0.12** | **96.52 ± 0.19** | 159.2 ± 1.3 s |
| **Method B (SE-ResNet-18)** | 11,261,000 | **97.23 ± 0.14** | **97.13 ± 0.14** | 225.0 ± 2.6 s |

> **Key Takeaway:** Method B achieves state-of-the-art **97.23% Accuracy** and **97.13% Macro-F1** across 3 seeds on BloodMNIST, confirming our hypothesis that channel-attention effectively resolves subtle leukocyte morphological confusions.

---

## 📁 6. Repository Structure
```text
Group13_Final_Project/
├── Group13_BloodMNIST_Experiments.ipynb  # Self-contained master Colab notebook
├── dataset.py                            # MedMNIST data loaders & distribution check
├── models.py                             # Baseline, Method A, and SE-ResNet18 definitions
├── train.py                              # Multi-seed training engine (seed 42, 123, 999)
├── evaluate.py                           # Plotting & statistical aggregation scripts
├── requirements.txt                      # Python dependencies
├── README.md                             # Reproduction documentation
└── outputs/                              # Generated figures & CSV tables
    ├── data_distribution.csv
    ├── results_table.csv
    ├── learning_curves.png
    ├── confusion_matrix.png
    └── failure_cases.png
```
