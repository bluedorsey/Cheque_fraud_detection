<div align="center">

# 🏦 Cheque Fraud Detection (ML Backend)
**An AI-driven solution to identify counterfeit and altered bank cheques.**

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)

*Securing financial transactions with Deep Learning and Computer Vision.*

[My Contribution](#%EF%B8%8F-my-contribution-ml-backend) • [UI Showcase](#-ui-showcase) • [Tech Stack](#-tech-stack) • [Installation](#%EF%B8%8F-installation--setup)
</div>

---

## 📖 Introduction
This repository showcases the **Machine Learning Backend** for a collaborative Cheque Fraud Detection system. Cheque fraud costs financial institutions billions annually, and this project automates the verification process by analyzing scanned cheque images for anomalies, alterations, and structural inconsistencies.

While this was a team project with a dedicated frontend, this repository focuses on the core PyTorch and OpenCV inference engine that powers the application.

---

## 🛠️ My Contribution (ML Backend)
As the Machine Learning Developer on this team, I was responsible for the entire AI pipeline, from data processing to model inference:
* **Image Preprocessing:** Utilized `cv2` (OpenCV) and `PIL` to handle grayscale conversion, thresholding, and cropping of Regions of Interest (ROIs).
* **Data Augmentation:** Implemented `torchvision.transforms` to standardize cheque images and prepare tensors for the neural network.
* **Deep Learning Model:** Built and trained the classification architecture using `torch.nn` and pre-trained networks via `torchvision.models`.
* **Inference Pipeline:** Created the backend logic to accept an image, pass it through the PyTorch model, and return a confidence score regarding its authenticity.
* **Visualization:** Used `matplotlib.pyplot` and `matplotlib.patches` for bounding box visualization and debugging during the model training phase.

---

## 📱 Output
<p align="center">
  <img src="https://github.com/user-attachments/assets/b26c945f-43ef-4b25-a34f-751b4ac0cb80" width="45%"  />
  <img src="https://github.com/user-attachments/assets/69559d6a-08d5-44cc-86d8-ab423335ba8c" width="45%" />
</p>

---

## 🧰 Tech Stack
**Core AI & Vision Libraries:**
* `torch`, `torch.nn` - Deep Learning framework and neural network layers.
* `torchvision` (`models`, `transforms`) - Pre-trained model architectures and image transformations.
* `cv2` (OpenCV) - Core computer vision and image manipulation.
* `numpy` - High-performance matrix and tensor operations.

**Data Handling & Visualization:**
* `PIL.Image` - Image loading and formatting.
* `matplotlib` (`pyplot`, `patches`) - Generating plots, graphs, and visual bounding boxes.

**Standard Python Libraries:**
* `os`, `io`, `sys`, `pathlib` - File and system path management.
* `json`, `re` - Data parsing and regular expressions.
* `argparse` - CLI argument parsing for model training scripts.
* `time`, `random` - Performance tracking and stochastic operations.

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.8+
* A CUDA-enabled GPU is highly recommended for training/inference.

### Step-by-Step Guide
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/bluedorsey/Cheque_fraud_detection.git](https://github.com/bluedorsey/Cheque_fraud_detection.git)
   cd Cheque_fraud_detection
