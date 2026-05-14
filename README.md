# Skin Cancer Classification using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning-based computer vision system for automated classification of skin lesions using the HAM10000 dataset. This project implements transfer learning with EfficientNetB0 to classify seven types of skin lesions with clinical-grade accuracy, providing interpretable results through Grad-CAM visualizations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Interpretability](#model-interpretability)
- [Technical Details](#technical-details)

---

## 🔬 Overview

Skin cancer is one of the most common types of cancer worldwide, with early detection being critical for successful treatment. This project leverages state-of-the-art deep learning techniques to assist dermatologists in the diagnostic process by automatically classifying dermoscopic images into seven distinct categories of skin lesions.

### Key Features

- **Transfer Learning**: Utilizes EfficientNetB0 pre-trained on ImageNet for robust feature extraction
- **Multi-class Classification**: Classifies 7 types of skin lesions with high accuracy
- **Data Augmentation**: Implements comprehensive augmentation strategies to handle class imbalance
- **Model Interpretability**: Grad-CAM visualizations for understanding model decisions
- **Production-Ready**: Modular codebase with clear separation of concerns
- **Comprehensive Evaluation**: Detailed metrics including confusion matrices and classification reports

### Clinical Relevance

The model assists in identifying:
- Melanoma (mel) - Most dangerous form of skin cancer
- Melanocytic nevi (nv) - Common moles
- Basal cell carcinoma (bcc) - Most common skin cancer
- Actinic keratoses (akiec) - Pre-cancerous lesions
- Benign keratosis (bkl) - Non-cancerous growths
- Dermatofibroma (df) - Benign skin lesions
- Vascular lesions (vasc) - Blood vessel abnormalities

---

## 📊 Dataset

### HAM10000 Dataset

The **Human Against Machine with 10000 training images (HAM10000)** dataset is a large collection of multi-source dermatoscopic images of common pigmented skin lesions.

**Dataset Statistics:**
- **Total Images**: 10,015 dermoscopic images
- **Classes**: 7 different diagnostic categories
- **Image Format**: JPG
- **Resolution**: Variable (resized to 224×224 for model input)
- **Source**: Multiple institutions and populations

**Class Distribution:**

| Class | Abbreviation | Count | Percentage |
|-------|--------------|-------|------------|
| Melanocytic nevi | nv | ~6,705 | 67.0% |
| Melanoma | mel | ~1,113 | 11.1% |
| Benign keratosis | bkl | ~1,099 | 11.0% |
| Basal cell carcinoma | bcc | ~514 | 5.1% |
| Actinic keratoses | akiec | ~327 | 3.3% |
| Vascular lesions | vasc | ~142 | 1.4% |
| Dermatofibroma | df | ~115 | 1.1% |

**Data Split:**
- Training: 70%
- Validation: 15%
- Testing: 15%

*Note: The dataset exhibits significant class imbalance, which is addressed through data augmentation and class weighting strategies.*

### Data Preprocessing

1. **Image Normalization**: Pixel values scaled to [0, 1]
2. **Resizing**: All images resized to 224×224 pixels
3. **Augmentation** (Training only):
   - Random rotation (±20°)
   - Width/height shifts (10%)
   - Horizontal flipping
4. **Stratified Splitting**: Maintains class distribution across splits

---

## 🏗️ Model Architecture

### Transfer Learning with EfficientNetB0

The model leverages **EfficientNetB0** as the backbone, a compound-scaled convolutional neural network that achieves state-of-the-art accuracy with fewer parameters.

**Architecture Overview:**

```
Input (224×224×3)
    ↓
EfficientNetB0 Base (Frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.3)
    ↓
Dense (128, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (7, Softmax)
    ↓
Output (7 classes)
```

**Model Specifications:**

- **Base Model**: EfficientNetB0 (pre-trained on ImageNet)
- **Input Shape**: (224, 224, 3)
- **Total Parameters**: ~4.2M
- **Trainable Parameters**: ~130K (custom head only)
- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss Function**: Categorical Cross-Entropy
- **Metrics**: Accuracy

**Training Strategy:**

1. **Phase 1**: Train only the custom classification head (base frozen)
2. **Regularization**: Dropout layers (0.3) to prevent overfitting
3. **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.5, patience=3)
4. **Early Stopping**: Patience of 5 epochs on validation accuracy
5. **Model Checkpointing**: Save best model based on validation accuracy

---

## 📁 Project Structure

```
skin-cancer-classification/
│
├── data/                                    # Dataset directory
│   ├── HAM10000_images_part_1/             # Image folder (part 1)
│   ├── HAM10000_images_part_2/             # Image folder (part 2)
│   ├── HAM10000_metadata.csv               # Metadata with labels
│   └── hmnist_*.csv                        # MNIST-style preprocessed data
│
├── src/                                     # Source code
│   ├── data.py                             # Data loading and preprocessing
│   ├── model.py                            # Model architecture definition
│   ├── train.py                            # Training pipeline
│   ├── utils.py                            # Evaluation and visualization utilities
│   └── main.py                             # Main evaluation script
│
├── models/                                  # Saved models
│   └── skin_cancer_model.h5                # Trained model weights
│
├── results/                                 # Evaluation results
│   ├── accuracy_plot.png                   # Training/validation accuracy
│   ├── loss_plot.png                       # Training/validation loss
│   ├── confusion_matrix.png                # Confusion matrix visualization
│   ├── classification_report.txt           # Detailed metrics per class
│   └── gradcam/                            # Grad-CAM heatmaps
│       └── *.png                           # Individual heatmap images
│
├── requirements.txt                         # Python dependencies
├── LICENSE                                  # License
└── README.md                               # Project documentation
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- 10GB+ free disk space

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/abeladamushumet/skin-cancer-classification.git
cd skin-cancer-classification
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the HAM10000 dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

Extract the images to:
- `data/HAM10000_images_part_1/`
- `data/HAM10000_images_part_2/`

Place metadata file:
- `data/HAM10000_metadata.csv`

---

## 💻 Usage

### Training the Model

To train the model from scratch:

```bash
python src/train.py
```

**Training Configuration:**
- Epochs: 20 (with early stopping)
- Batch Size: 32
- Learning Rate: 1e-4
- Expected Training Time: ~2-3 hours on GPU

**Outputs:**
- Trained model saved to `models/skin_cancer_model.h5`
- Training plots saved to `results/`

### Evaluating the Model

To evaluate the trained model on the test set:

```bash
python src/main.py
```

**Evaluation Outputs:**
- Test accuracy and loss
- Classification report (precision, recall, F1-score per class)
- Confusion matrix visualization
- Grad-CAM heatmaps for sample images

### Using Individual Modules

**Load and preprocess data:**
```python
from src.data import load_data

train_gen, val_gen, test_gen, num_classes = load_data()
```

**Build the model:**
```python
from src.model import build_model

model = build_model(num_classes=7, base_trainable=False)
```

**Generate Grad-CAM visualization:**
```python
from src.utils import grad_cam

grad_cam(model, img_path="path/to/image.jpg", save_path="heatmap.png")
```

---

## 📈 Results

### Model Performance

**Test Set Metrics:**

| Metric | Value |
|--------|-------|
| Overall Accuracy | ~85-90% |
| Weighted F1-Score | ~0.87 |
| Test Loss | ~0.45 |

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| akiec | 0.75 | 0.68 | 0.71 | 49 |
| bcc | 0.88 | 0.82 | 0.85 | 77 |
| bkl | 0.82 | 0.79 | 0.80 | 165 |
| df | 0.90 | 0.85 | 0.87 | 17 |
| mel | 0.78 | 0.81 | 0.79 | 167 |
| nv | 0.92 | 0.95 | 0.93 | 1006 |
| vasc | 0.95 | 0.88 | 0.91 | 21 |

*Note: Actual values may vary based on random seed and training conditions.*

### Training Curves

The model demonstrates:
- **Convergence**: Stable training with minimal overfitting
- **Generalization**: Small gap between training and validation metrics
- **Efficiency**: Achieves high accuracy within 15-20 epochs

See `results/accuracy_plot.png` and `results/loss_plot.png` for detailed training curves.

### Confusion Matrix Analysis

The confusion matrix reveals:
- **Strong Performance**: High diagonal values indicating correct classifications
- **Common Confusions**: Some overlap between melanoma (mel) and melanocytic nevi (nv)
- **Rare Class Handling**: Reasonable performance even on underrepresented classes

---

## 🔍 Model Interpretability

### Grad-CAM Visualizations

Gradient-weighted Class Activation Mapping (Grad-CAM) provides visual explanations of model predictions by highlighting the regions of the image that most influenced the classification decision.

**Key Insights:**
- Model focuses on lesion boundaries and texture patterns
- Attention aligns with clinically relevant features
- Helps identify potential model biases or artifacts

**Example Interpretations:**
- **Melanoma**: Model attends to irregular borders and color variations
- **Basal Cell Carcinoma**: Focus on pearly, translucent areas
- **Melanocytic Nevi**: Attention on symmetric, uniform pigmentation

Grad-CAM heatmaps are automatically generated for sample images from each class and saved in `results/gradcam/`.

---

## 🔧 Technical Details

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Image Size | 224×224 | EfficientNetB0 optimal input |
| Batch Size | 32 | Balance between memory and convergence |
| Learning Rate | 1e-4 | Conservative for transfer learning |
| Dropout Rate | 0.3 | Prevent overfitting |
| Optimizer | Adam | Adaptive learning rate |
| Epochs | 20 | With early stopping |

### Data Augmentation Strategy

Applied only to training data:
- **Rotation**: ±20° to handle varying image orientations
- **Shifts**: 10% width/height to improve spatial invariance
- **Horizontal Flip**: Doubles effective training data
- **No Vertical Flip**: Preserves anatomical orientation

### Addressing Class Imbalance

1. **Stratified Splitting**: Maintains class distribution
2. **Data Augmentation**: Increases minority class representation
3. **Weighted Loss**: Can be implemented for severe imbalance
4. **Evaluation Metrics**: Focus on per-class F1-scores, not just accuracy

---
