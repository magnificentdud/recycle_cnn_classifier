<p align="center">
  <img src="https://svg-banners.vercel.app/api?type=rainbow&text=Recycle+Classification+CNN&width=1000&height=250" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-MobileNetV2-rainbow" />
  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange" />
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen" />
</p>

🌈━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━🌈

# 🌈✨ Recycle Classification using CNNs & Transfer Learning

A vibrant, end‑to‑end machine learning project for classifying recyclable materials into **10 categories** using **Convolutional Neural Networks** and **MobileNetV2 transfer learning**.

🌈━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━🌈

## 🌈📘 Overview

This project builds an image‑classification model that identifies different types of recyclable waste.  
It uses:

- 🧠 Deep Learning (CNNs)  
- 🚀 Transfer Learning (MobileNetV2)  
- 🗂️ ImageDataGenerator pipelines  
- 🧪 Train / Validation / Test splits  
- 📊 Evaluation tools (confusion matrix, classification report)  

The goal is to create a **clean, modular, portfolio‑ready ML project** with reproducible code and a professional structure.

🌈━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━🌈

## 🌈📦 Dataset

The dataset contains **10 classes**, each with ~1000 images:

- 🔋 battery  
- 🧫 biological  
- 🟤 brown‑glass  
- 📦 cardboard  
- 🟢 green‑glass  
- 🛠️ metal  
- 📄 paper  
- 🧴 plastic  
- 🗑️ trash  
- ⚪ white‑glass  

Split into:

- **Train:** ~800 images/class  
- **Validation:** ~100 images/class  
- **Test:** ~100 images/class  

🌈━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━🌈

## 🌈🧠 Model Architecture

This project uses **MobileNetV2** with:

- Frozen base layers (for fast training)  
- Global Average Pooling  
- Dropout regularization  
- Dense softmax output layer  

This approach provides:

- ⚡ Faster training  
- 🎯 Higher accuracy  
- 🧩 Better generalization  

🌈━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━🌈

## 🌈📁 Project Structure

recycle-cnn-classification/
├─ src/
│  ├─ dataset.py          # Data loading utilities
│  ├─ model.py            # CNN / MobileNetV2 model
│  ├─ train.py            # Training script
│  └─ eval.py             # Evaluation utilities
├─ notebooks/
│  └─ recycle_classification_training.ipynb
├─ saved_models/
├─ data/                  # (empty placeholder)
└─ README.md

Code

🌈━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━🌈

## 🌈🚀 How to Run the Code

### 1. Install dependencies
pip install tensorflow matplotlib seaborn scikit-learn

### 2. Load dataset
Place your dataset in:
data/train/
data/val/
data/test/

### 3. Train the model
python src/train.py

### 4. Evaluate
python src/eval.py

🌈━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━🌈

## 🌈📊 Results (from original paper)

*(Replace these with your actual numbers)*

- **Accuracy:** 92–95%  
- **Loss:** Stable convergence  
- **Confusion Matrix:** Strong performance across all 10 classes  
- **Observations:** Transfer learning significantly outperformed custom CNN  

🌈━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━🌈

## 🌈🎯 Key Features

- 🌈 Clean, modular code  
- 🧠 Transfer learning for high accuracy  
- 🧪 Reproducible training pipeline  
- 📊 Built‑in evaluation tools  
- 📁 Professional project structure  
- 🧵 Ready for GitHub & portfolio use  

🌈━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━🌈

## 🌈💡 Future Improvements

- Add Grad‑CAM visualizations  
- Deploy as a web app  
- Add ONNX export  
- Improve dataset augmentation  

🌈━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━🌈

## 🌈👤 Author

**Woochan Jung**  
Machine Learning & Computer Vision Enthusiast  

<p align="center">
  🌈 Made with passion, Python, and a lot of colors 🌈
</p>