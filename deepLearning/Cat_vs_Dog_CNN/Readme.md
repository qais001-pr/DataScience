# 🐶🐱 Cats vs Dogs Image Classification using CNN

This repository contains a **Convolutional Neural Network (CNN)** implementation in **TensorFlow/Keras** for classifying images of **cats and dogs**.  
The project is built to run smoothly on **Google Colab** with GPU support.

---

## 📌 Project Overview
The goal of this project is to build and train a deep learning model that can classify images into **2 categories**:
- Cat 🐱
- Dog 🐶

The dataset consists of **25,000 training images** and **12,500 testing images** (JPG format).  

---

## ⚙️ Features
- Image preprocessing and normalization  
- CNN model with Conv2D, MaxPooling, Flatten, and Dense layers  
- Training with accuracy/loss monitoring  
- Evaluation on the test dataset  
- Visualization of training vs validation performance  

---

## 🚀 Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Google Colab (GPU recommended)  

---

## 🧩 Model Architecture
- **Conv2D + MaxPooling layers** for feature extraction  
- **Flatten + Dense layers** for classification  
- **Sigmoid output layer** for binary classification (Cat vs Dog)  

---

## 📊 Results
- Model trained for multiple epochs  
- Achieved **~90% accuracy** on test dataset  
- Accuracy and loss curves plotted for training/validation  

---

## 🔮 Future Improvements
- Add **Data Augmentation** for more robust learning  
- Introduce **Dropout layers** to reduce overfitting  
- Hyperparameter tuning (learning rate, batch size, number of epochs)  

---

## 📊 Presentation
- [Download Presentation](./docs/Cats-vs-Dogs-Classification.pptx)

---

## ▶️ How to Run (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/).  
2. Upload the file:  
   - `cats_vs_dogs_colab.py` (or `.ipynb`)  
3. Install required dependencies:
   ```bash
   !pip install tensorflow numpy matplotlib kaggle
