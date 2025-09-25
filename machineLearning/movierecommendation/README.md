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
- **Python 3.6.8**  
- **PySpark** (MLlib, SQL)  
- **MovieLens Dataset (100k ratings)**  

---
## Libraries 
- PySpark
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
### 2. Locate this folder
```bash
cd movierecommendation
```
### 2. Run this Command

```bash
spark-submit \
--master yarn \
--deploy-mode client\ 
#/home/qais/Desktop/datascience/DataScience/machineLearning/movierecommendation/script.py
srcipt.py\
#/user/qais/MovieLens/ratings.csv
hdfs:///<ratingsFilePath> \
#/user/qais/MovieLens/movies.csv
hdfs:///<moviesfilePath>
```
