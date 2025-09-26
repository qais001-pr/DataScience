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

## ⚙️ Requirements
Make sure you have the following installed:
- Python 3.7+  
- PySpark  
- TensorFlow 2.x  
- NumPy  

Install dependencies:
```bash
pip install pyspark tensorflow numpy
spark-submit --driver-python python3 cats_vs_dogs.py
