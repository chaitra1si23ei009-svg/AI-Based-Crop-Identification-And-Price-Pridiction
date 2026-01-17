# 🌾 AI-Based Crop Identification and Harvest-Time Price Prediction

This repository contains the implementation of an **AI-based system for automated crop identification and harvest-time price prediction** using deep learning techniques. The project integrates **image-based crop classification** with **time-series market price forecasting** to support smart and data-driven agricultural decision-making.

---

## 📌 Project Overview

Agriculture plays a crucial role in the Indian economy, but farmers often face challenges such as:
- Inaccurate crop identification
- Uncertainty in market prices at harvest time

To address these issues, this project proposes an integrated AI solution that:
- Identifies crops (Maize, Rice, Wheat) from images using **ResNet50**
- Predicts harvest-time crop prices using **LSTM (Long Short-Term Memory)** networks

The system helps farmers and stakeholders make informed decisions regarding crop planning, harvest timing, and market strategies.

---

## 🎯 Objectives

- Automate crop identification using deep learning
- Predict future market prices at harvest time
- Reduce dependency on manual inspection and expert judgment
- Provide a scalable and farmer-friendly smart agriculture solution

---

## 🧠 System Architecture

### 1. Crop Identification Module
- Model: **ResNet50 (CNN)**
- Technique: Transfer Learning
- Input: Crop leaf images
- Output: Crop type with confidence score

### 2. Price Prediction Module
- Model: **LSTM**
- Input: Historical crop price data
- Output: Predicted harvest-time prices

### 3. Integrated Interface
- Image upload / webcam input
- Crop identification result
- Historical and predicted price trends

---

## 🛠️ Technologies Used

### Programming & Frameworks
- Python 3.10
- PyTorch
- Torchvision
- NumPy
- Pandas
- OpenCV
- Pillow

### Tools & Platforms
- Streamlit (Web Interface)
- Google Colab / Kaggle (GPU support)

---

## 📂 Dataset Details

### Image Dataset
- Total Images: 3000
- Crops: Maize, Rice, Wheat
- Image Size: 224 × 224
- Train/Test Split: 80% / 20%

### Price Dataset
- Historical crop market prices
- Time-series structured data
- Used for LSTM-based forecasting

---

## 📈 Performance Metrics

### Crop Identification
- Accuracy: **96%**
- Precision: **95.8%**
- Recall: **95.9%**
- F1-Score: **95.9%**

### Price Prediction
- RMSE: **12.34**
- MAE: **9.87**

---

## ▶️ How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-crop-identification-price-prediction.git
cd ai-crop-identification-price-prediction
Install Dependencies
pip install -r requirements.txt

3. Run the Application
streamlit run app.py
