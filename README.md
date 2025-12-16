# Histopathological Tissue Classification

Deep learning pipeline for multi-class histopathological image classification using ResNet50 with custom preprocessing and ROI-aware training.

## 🎯 Overview

- **Task**: 8-class tissue type classification from histology images
- **Architecture**: ResNet50 with pretrained ImageNet weights
- **Best F1 Score**: 0.3865 (test set)
- **Key Features**: Artifact removal, tile-based processing, k-fold ensemble, Grad-CAM visualization

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run preprocessing + training + inference
jupyter notebook main.ipynb
```

**Configure experiment** by setting `EXP_NAME` in first cell (e.g., `"resnet50_strongaug_384_new_kfold_finale"`).

## 📁 Project Structure

```
├── main.ipynb                    # Main training pipeline
├── includes/
│   ├── config.py                 # Experiment configurations
│   ├── data_utils.py             # Data loading & augmentation
│   ├── model_utils.py            # Model building & training
│   ├── inference_utils.py        # Test inference & ensemble
│   └── cam_utils.py              # Grad-CAM visualization
├── preprocessing/
│   └── preprocessing.py          # Offline data preprocessing
├── data/
│   ├── train_data/               # Raw training images
│   ├── test_data/                # Raw test images
│   ├── pp_train_data/            # Preprocessed tiles
│   └── pp_test_data/             # Preprocessed test tiles
└── out/                          # Submission files & visualizations
```

## 🔧 Key Components

**Preprocessing Pipeline**:
- Shrek artifact removal (green channel corruption)
- Stain normalization & black rectangle removal
- ROI-based square cropping with padding
- Tile splitting (6-8 tiles per image)
- Offline augmentation (rotation, zoom, color jitter)

**Training Strategy**:
- Stratified Group K-Fold (prevents tile leakage)
- Class-weighted loss for imbalance
- Mixed precision training (AMP)
- Cosine annealing LR scheduler
- Early stopping with patience

**Inference**:
- 4-fold ensemble averaging
- Tile-to-image aggregation (softmax averaging)
- Test-time augmentation optional

## 📊 Results

| Configuration | Val F1 | Test F1 |
|--------------|--------|---------|
| Baseline (no preprocessing) | 0.19 | - |
| + Preprocessing | 0.32 | 0.29 |
| + K-fold ensemble | 0.47 | 0.33 |
| **Final (strong aug + ensemble)** | **0.72** | **0.39** |

## 🎨 Visualization

Grad-CAM heatmaps available in `out/gradcam/` showing model attention on discriminative tissue features.

## ⚙️ Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision, scikit-learn, opencv-python, pandas, matplotlib


---

*Project developed for AN2DL Challenge 2 (2025)*
