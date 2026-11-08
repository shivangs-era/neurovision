# 🧠 NeuroVision: Deep Learning Vision & Explainable AI Suite

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.15](https://img.shields.io/badge/PyTorch-2.15%2Bcpu-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.53-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Test Accuracy](https://img.shields.io/badge/CNN%20Accuracy-99.36%25-brightgreen)](https://github.com/)
[![ResNet Accuracy](https://img.shields.io/badge/ResNet%20Accuracy-98.34%25-brightgreen)](https://github.com/)

**NeuroVision** is a production-grade, interactive Deep Learning system built with **PyTorch** and **Streamlit**. It bridges real-time Computer Vision classification with state-of-the-art **Explainable AI (Grad-CAM & Saliency)**, **Adversarial Robustness (FGSM)**, and **Generative Latent Manifold Synthesis (ConvVAE)**.

---

## 📑 Table of Contents

- [Key Capabilities](#-key-capabilities)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Benchmark Results](#-benchmark-results)
- [Quickstart & Installation](#-quickstart--installation)
- [Interactive Web Studio (Streamlit)](#-interactive-web-studio-streamlit)
- [Command-Line Interface (CLI)](#-command-line-interface-cli)
- [Python API Examples](#-python-api-examples)
- [Unit Testing & Verification](#-unit-testing--verification)
- [License](#-license)

---

## 🌟 Key Capabilities

### 1. Dual Deep Vision Architectures
- **`ConvNetClassifier`**: Multi-stage Deep Convolutional Neural Network with Batch Normalization, Spatial Regularization (Dropout), and Adaptive Average Pooling (**99.36% test accuracy** across 10,000 samples).
- **`NeuroResNet`**: Deep Residual Network with skip connections ($F(x) + x$) and $1 \times 1$ projection shortcuts to prevent gradient vanishing (**98.34% test accuracy**).

### 2. Generative Latent Space Synthesis (ConvVAE)
- Convolutional Variational Autoencoder that maps visual data into a continuous 2-dimensional Gaussian latent manifold $\mathbf{z} \in \mathbb{R}^2$.
- Utilizes the reparameterization trick ($\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$) for end-to-end backpropagation.
- Real-time latent coordinate sliders ($z_1, z_2$) synthesize novel digits on the fly.

### 3. Explainable AI (XAI) Suite
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Captures gradient flow into the final convolutional feature maps, highlighting the spatial regions that drive decisions. Includes counterfactual explanations.
- **Pixel Saliency & SmoothGrad**: Computes pixel-level loss gradients and applies Gaussian noise sampling to eliminate high-frequency gradient noise.
- **Intermediate Layer Feature Maps**: Visualizes internal channel activations hierarchically (edges $\to$ textures $\to$ semantics).

### 4. Adversarial Robustness Lab
- **Fast Gradient Sign Method (FGSM)**:
  $$\eta = \epsilon \cdot \text{sign}\left(\nabla_x \mathcal{L}(\theta, x, y)\right)$$
  $$x_{\text{adv}} = \text{clip}(x + \eta, 0, 1)$$
- Supports both **untargeted misclassification** and **targeted class spoofing**.
- Measures $L_\infty$ and $L_2$ distortion metrics alongside probability degradation.

### 5. Interactive Streamlit Web Studio
- Full-featured browser UI featuring 6 specialized workspaces:
  - 🎯 **Live Inference & Canvas**: Interactive HTML5 drawing pad, custom file uploads, and sample presets.
  - 🔍 **Explainable AI (XAI)**: Grad-CAM heatmap overlays, SmoothGrad, and feature maps.
  - 🛡️ **Adversarial Robustness**: Real-time $\epsilon$ perturbation slider and attack breakdown.
  - 🌌 **Latent Space & VAE**: Latent coordinate sliders and 2D manifold synthesis grid.
  - 🔬 **Live Training Studio**: Train models live in-browser with streaming loss/accuracy curves.
  - 📊 **Architecture & Metrics**: Layer-by-layer parameter analysis and confusion matrix.

---

## 🏗️ System Architecture

```
                    ┌───────────────────────────────┐
                    │      Input Representation     │
                    │  (Drawn Canvas / Upload / Test)│
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Center-of-Mass Preprocessing│
                    │   (BBox Crop, 20x20, Centered)│
                    └───────────────┬───────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│ConvNetClassifier          │  NeuroResNet  │           │    ConvVAE    │
│(5 Conv + BN)  │           │(Skip Conns)   │           │(Latent Space) │
└───────┬───────┘           └───────┬───────┘           └───────┬───────┘
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│Logits & Probs │           │Logits & Probs │           │Decoded Sample │
│Top-1: 99.36%  │           │Top-1: 98.34%  │           │z1, z2 in R^2  │
└───────┬───────┘           └───────┬───────┘           └───────────────┘
        │                           │
        ├───────────────────────────┤
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│Explainable AI │           │  Adversarial  │
│(Grad-CAM / XAI│           │ (FGSM Attack) │
└───────────────┘           └───────────────┘
```

---

## 📁 Repository Structure

```
.
├── .gitignore                  # Git exclusions for venv, caches, and datasets
├── app.py                      # Interactive Streamlit Web Application
├── cli.py                      # Command-Line Interface (CLI)
├── requirements.txt            # Frozen Python dependencies
├── train_defaults.py           # Local training pipeline for baseline weights
├── README.md                   # Complete repository documentation
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py              # ConvNetClassifier architecture
│   │   ├── resnet.py           # NeuroResNet residual network
│   │   └── vae.py              # ConvVAE variational autoencoder
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── gradcam.py          # Grad-CAM hook-based activation mapping
│   │   ├── saliency.py         # Vanilla gradient saliency & SmoothGrad
│   │   └── feature_maps.py     # Layer-wise activation extraction & grid maker
│   ├── adversarial/
│   │   ├── __init__.py
│   │   └── fgsm.py             # Fast Gradient Sign Method (FGSM) attack engine
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Offline dataset loader & synthetic fallback
│   │   └── presets.py          # Curated test samples for classes 0-9
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Supervised ModelTrainer & Unsupervised VAETrainer
│   │   └── metrics.py          # Pure vector accuracy, precision, recall, F1, confusion matrix
│   └── utils/
│       ├── __init__.py
│       ├── image_ops.py        # Center-of-mass centering & 28x28 aspect normalization
│       └── visualization.py    # Matplotlib plotting routines (Grad-CAM, FGSM, Manifold)
├── weights/
│   ├── best_cnn.pt             # Pre-trained ConvNetClassifier weights (99.36% test acc)
│   ├── best_resnet.pt          # Pre-trained NeuroResNet weights (98.34% test acc)
│   └── best_vae.pt             # Pre-trained ConvVAE latent generator weights
└── tests/
    ├── test_models.py          # Unit tests for CNN, ResNet, VAE forward/backward pass
    ├── test_explainability.py  # Unit tests for Grad-CAM, Saliency, Feature Maps
    ├── test_adversarial.py     # Unit tests for FGSM perturbation & bounds
    └── test_pipeline.py        # Unit tests for preprocessing & classification metrics
```

---

## 📊 Benchmark Results

Evaluated across the full **10,000 sample test partition**:

| Model Architecture | Parameters | Top-1 Accuracy | Macro F1 | Weighted F1 | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ConvNetClassifier** | **312,874** | **99.36%** | **0.9936** | **0.9936** | **0.0205** |
| **NeuroResNet** | **227,626** | **98.34%** | **0.9833** | **0.9834** | **0.0642** |
| **ConvVAE** | **823,297** | Continuous $\mathbb{R}^2$ Latent Manifold | N/A | N/A | Recon: 104.2 |

### Per-Class Performance Breakdown (`ConvNetClassifier`):

| Class | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 99.5% | 99.7% | 0.996 | 980 |
| **1** | 99.5% | 99.6% | 0.995 | 1135 |
| **2** | 99.4% | 99.4% | 0.994 | 1032 |
| **3** | 99.2% | 99.7% | 0.995 | 1010 |
| **4** | 99.3% | 99.5% | 0.994 | 982 |
| **5** | 99.9% | 98.4% | 0.992 | 892 |
| **6** | 99.4% | 99.2% | 0.993 | 958 |
| **7** | 98.7% | 99.6% | 0.992 | 1028 |
| **8** | 99.2% | 99.5% | 0.993 | 974 |
| **9** | 99.6% | 98.9% | 0.993 | 1009 |

---

## 🚀 Quickstart & Installation

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/shivangs-era/neurovision.git
cd neurovision

# Create Python virtual environment
python -m venv .venv

# Activate on Windows
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Web Studio

```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 💻 Command-Line Interface (CLI)

The repository provides a full CLI via `cli.py`:

### Predict Class of an Image
```bash
python cli.py predict --image test_digit_7.png --model cnn
```
```
[*] Loaded CNN weights from: ./weights/best_cnn.pt
=============================================
NeuroVision Prediction Result
=============================================
Predicted Class: Digit 7
Confidence:      100.00%

Top Candidates:
  1. Digit 7: 100.00%
  2. Digit 3: 0.00%
  3. Digit 9: 0.00%
=============================================
```

### Generate Grad-CAM Explanation
```bash
python cli.py explain --image test_digit_7.png --method gradcam --output explanation_gradcam.png
```

### Run FGSM Adversarial Attack
```bash
python cli.py attack --image test_digit_7.png --epsilon 0.25 --output adversarial_result.png
```
```
[*] Loaded CNN weights from: ./weights/best_cnn.pt
==================================================
FGSM Adversarial Attack Evaluation
==================================================
Perturbation Magnitude (epsilon): 0.25
Clean Prediction:            Digit 7 (100.00%)
Adversarial Prediction:      Digit 3 (96.93%)
Model Fooled:                YES
L_inf Max Distortion:        0.2500
L2 Euclidean Distortion:     5.3123
==================================================
```

### Evaluate on Test Set
```bash
python cli.py evaluate --model cnn
python cli.py evaluate --model resnet
```

### Train from Scratch
```bash
python cli.py train --model cnn --epochs 5 --lr 0.001 --optimizer adam
```

---

## 🔬 Python API Examples

### 1. Forward Pass and Inference
```python
import torch
from PIL import Image
from src.models import ConvNetClassifier
from src.utils import preprocess_drawn_image

# Load model
model = ConvNetClassifier()
ckpt = torch.load("./weights/best_cnn.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Preprocess image
img = Image.open("test_digit_7.png")
tensor, norm_np = preprocess_drawn_image(img)

# Inference
with torch.no_grad():
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    prediction = int(torch.argmax(probs, dim=1).item())

print(f"Predicted Class: {prediction} with {probs[0, prediction]*100:.1f}% confidence")
```

### 2. Generate Grad-CAM Heatmap
```python
from src.explainability import GradCAM

cam = GradCAM(model)
heatmap, pred_class, conf = cam.generate(tensor)
overlay = cam.overlay_on_image(norm_np, heatmap, colormap="jet", alpha=0.5)
```

### 3. Generate Adversarial Examples with FGSM
```python
from src.adversarial import FGSMAttack

attacker = FGSMAttack(model)
results = attacker.attack(tensor, epsilon=0.20)
print(f"Fooled: {results['is_fooled']} -> Adv Class: {results['adv_prediction']}")
```

---

## 🧪 Unit Testing & Verification

The test suite covers model architectures, forward/backward passes, layer activation hooks, Grad-CAM attribution, FGSM bounds, and data transforms:

```bash
python -m unittest discover tests
```

Output:
```
..............
----------------------------------------------------------------------
Ran 14 tests in 0.336s

OK
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
