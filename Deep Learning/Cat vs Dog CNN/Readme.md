# 🐶🐱 Cats vs Dogs Classification (PySpark + TensorFlow)

This project trains a **Convolutional Neural Network (CNN)** to classify images of cats and dogs.  
It uses **PySpark** for distributed data handling and **TensorFlow/Keras** for deep learning model training.

---

## 📂 Dataset
- **Source**: [Kaggle Dogs vs Cats](https://www.kaggle.com/datasets/biaiscience/dogs-vs-cats)  
- Training images: `/kaggle/input/dogs-vs-cats/train/train`  
- Testing images: `/kaggle/input/dogs-vs-cats/test/test`  
- File format: `.jpg`  
- Filenames are of the form:  
  - `cat.1234.jpg` → label = `cat` (0)  
  - `dog.5678.jpg` → label = `dog` (1)  

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
