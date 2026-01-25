<div align="center">

# 🍃 Leaffliction - Plant Disease Diagnosis AI

![School](https://img.shields.io/badge/School-42_Paris-000000?style=flat-square&logo=42&logoColor=white&labelColor=24292e)
![Language](https://img.shields.io/badge/Language-Python_3.11-3776AB?style=flat-square&logo=python&logoColor=white&labelColor=24292e)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white&labelColor=24292e)
![Library](https://img.shields.io/badge/Library-OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white&labelColor=24292e)

![Score](https://img.shields.io/badge/Score-96%2F100-32a852?style=flat-square&labelColor=24292e)

<p align="center">
  <strong>A Computer Vision pipeline for diagnosing apple leaf diseases with >97% accuracy.</strong><br>
  Built from scratch: Custom CNN architecture, Data Augmentation, and Analytics pipeline.
</p>

[Report Bug](https://github.com/jaytakahashii/Leaffliction/issues) · [Request Feature](https://github.com/jaytakahashii/Leaffliction/issues)

</div>

---

## 📒 Introduction

**Leaffliction** is a deep learning project designed to classify apple leaf pathologies. [cite_start]The system identifies conditions such as **Apple Scab**, **Black Rot**, and **Cedar Apple Rust** from raw images [cite: 66-68].

Unlike using high-level AutoML APIs, this project implements the entire machine learning lifecycle manually—from statistical data analysis and image processing to designing and training a Convolutional Neural Network (CNN) using **PyTorch**. The model achieves high generalization performance through custom data augmentation and strict validation protocols.

---

## 🚀 Technical Highlights

### 🧠 1. Custom CNN Architecture

Instead of using pre-trained models (like ResNet), I implemented a **VGG-style Convolutional Neural Network** from scratch to demonstrate a fundamental understanding of deep learning operations.

- **Feature Extractor**: Composed of **4 Convolutional Blocks**. Each block follows a `Conv2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d` pattern, progressively expanding feature maps from 32 to 128 channels.
- **Dynamic Layer Sizing**: The model dynamically calculates the input size for the fully connected layer using a dummy forward pass. This makes the architecture robust to changes in input image resolution ($256 \times 256$ by default) without manual hardcoding.
- **Regularization**: Implemented **Dropout (0.5)** in the classifier head to prevent overfitting on the training data.

### ⚡ 2. Optimized Training Loop

The training pipeline is built for performance and flexibility:

- **Hardware Acceleration**: The script automatically detects and utilizes the best available hardware accelerator: **CUDA** (NVIDIA), **MPS** (Apple Silicon/Metal), or CPU.
- **Optimizer**: Utilizes the **Adam** optimizer ($lr=0.001$) for faster convergence compared to standard SGD.
- **Early Stopping**: Includes a monitoring system that stops training once the validation accuracy exceeds the target threshold (97%), saving computational resources.

### 📊 3. Data Engineering Pipeline

To handle real-world data issues, the project includes comprehensive data processing tools:

- **Distribution Analysis**: Visualizes class imbalances using **Matplotlib/Seaborn** to guide the augmentation strategy.
- **Augmentation**: A custom pipeline applies transformations (Flip, Rotate, Skew, Shear, Distortion) to balance the dataset, ensuring the model sees a diverse range of examples.
- **Feature Extraction**: capable of analyzing ROI (Region of Interest) and Color Histograms using **OpenCV**.

---

## 🛠️ Installation & Usage

### Prerequisites

- Python 3.11+
- PyTorch, Torchvision, OpenCV, Matplotlib (See `requirements.txt`)

```bash
git clone https://github.com/jaytakahashii/Leaffliction.git
cd Leaffliction
pip install -r requirements.txt
```

### 1. Data Preparation & Analysis

Analyze the dataset distribution and apply augmentation if necessary.

```bash
# Analyze distribution
python Distribution.py ./images/Apple

# Apply augmentation to a specific image
python Augmentation.py ./images/Apple/apple_healthy/image_01.JPG
```

### 2. Training

Train the CNN model. The system creates a `submission.zip` containing the model and dataset upon completion.

```bash
# Train with default settings (15 epochs)
python train.py ./images/prepared_dataset

# Train with custom epochs
python train.py ./images/prepared_dataset --epochs 30
```

### 3. Prediction

Predict the disease class for a new, unseen image.

```bash
python predict.py ./test_images/unknown_leaf.JPG
# Output: Class predicted: apple_black_rot
```

---

## 📂 Project Structure

```text
Leaffliction/
├── tools/
│   ├── SimpleCNN.py        # CNN Model Definition (PyTorch)
│   ├── ImageAugmentor.py   # Data Augmentation Logic
│   └── ...
├── Distribution.py         # Dataset Visualization Tool
├── Augmentation.py         # Augmentation CLI
├── train.py                # Training Loop & Validation
├── predict.py              # Inference Script
├── requirements.txt        # Project Dependencies
└── README.md
```
