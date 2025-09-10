# 🎬 Movie Recommendation System using PySpark (ALS)

This repository contains a **Movie Recommendation System** built with **PySpark’s Alternating Least Squares (ALS)** algorithm.  
It uses the [MovieLens dataset](https://www.kaggle.com/datasets/aigamer/movie-lens-dataset) to demonstrate how collaborative filtering can be applied to recommend movies to users based on their historical ratings.

---

## 📌 Project Overview
- Loads and processes the **MovieLens dataset **
- Trains a **collaborative filtering model** using ALS from Spark MLlib  
- Evaluates the model with **Root Mean Square Error (RMSE)**  
- Generates:
  - Top-N movie recommendations for each user  
  - Top-N user recommendations for each movie  

---

## 🛠️ Tech Stack
- **Python 3.x**  
- **PySpark** (MLlib, SQL)  
- **MovieLens Dataset (100k ratings)**  

---

## 📂 Dataset
The project uses the **MovieLens small dataset (100k ratings)** which contains:  

- `ratings.csv` → userId, movieId, rating, timestamp  
- `movies.csv` → movieId, title, genres  

👉 Download here: [MovieLens Datasets](https://www.kaggle.com/datasets/aigamer/movie-lens-dataset)

---

## ⚡ Getting Started
### 📊 Presentation
[Download Presentation (PPTX) ](docs/slides.pptx)

### 1. Clone the Repository
```bash
git clone https://github.com/qais001-pr/Data-Science-Projects-Movie-Recommendation-System.git
```
### 2. Run this Command
```bash
python /home/usr/<name>/script.py hdfs:///<filePath> hdfs:///<filePath>