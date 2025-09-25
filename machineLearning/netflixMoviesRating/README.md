# Netflix Movie Recommendation System using PySpark ALS

This project implements a **Netflix-style movie recommendation system** using **PySpark's ALS (Alternating Least Squares) collaborative filtering** algorithm.

---

## **Project Overview**

The goal of this project is to recommend movies to users based on historical ratings. It includes:

- Reading user ratings and movie information from CSV files.
- Preprocessing data (casting types, handling nulls).
- Training an ALS model for collaborative filtering.
- Evaluating predictions with **RMSE**.
- Generating top-10 recommendations for users and movies.
- Saving results to HDFS.

---

## **DataSet**
  [Netflix Movie Rating](https://www.kaggle.com/datasets/evanschreiner/netflix-movie-ratings)
## **Presentation** 
  [Download Here](/netflix_movies_rating/docs/PySpark-ALS-Netflix-Movie-Recommendation-System.pptx)

---
## **Requirements**

- Python 3.6.8
- PySpark
- HDFS (for output storage)
- CSV datasets:
  - `ratings.csv` → Columns: `CustId, MovieId, Rating`
  - `movies.csv` → Columns: `MovieId, MovieTitle`

---
## **Setup & Usage**

### 1. Clone the repository:

```bash
git clone https://github.com/qais001-pr/Data_Science_Projects.git
```
## 2. Locate this folder
```bash
cd netflixMoviesRating
```


## ⚙️ Environment Setup

### 1. Create a Virtual Environment

```bash
python3.6 -m venv env
```

### 2.Activate the Virtual Environment
- Windows
```bash
  env\Scripts\activate
```
- Linux
```bash
source env/bin/activate
```
### 3. Install Dependencies
- Download Requirement File

[Download](https://github.com/qais001-pr/DataScience/tree/main/machineLearning/netflixMoviesRating/docs)


```bash

# Update the file Path of requirement file  to setup the virtual env
# File Path Like
# /workspaces/DataScience/machineLearning/movierecommendation/docs/requirements.txt
pip install -r <reuiqrementFilePath>

```

### 4. Run this Command in Terminal:

```bash
spark-submit \
--master yarn  \
   --deploy-mode client  \
    --driver-memory 4G \
    --executor-memory 4G  \
    # Update the file path where you file are located in your cluster or workspace
    #/home/qais/Desktop/dataScience/machinelearning/netflixMovieRating/script.py
    "<pythonScript>"  \
    #hdfs:///user/qais/MovieLens/Netflix_Prizes_Data.csv
    "hdfs://<userRatingData>" \
    #hdfs:///user/qais/MovieLens/movies.csv
    "hdfs://<moviesFilePath>" \
```

### 5. Create Database
```bash
CREATE DATABASE NETFLIX_DATA;
```

### 6. Use DATABASE and Create Table
```bash
Use NETFLIX_DATA
CREATE TABLE netflix_predictions (
    CustId INT,
    MovieId INT,
    MovieTitle STRING,
    Rating FLOAT,
    prediction FLOAT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
TBLPROPERTIES ("skip.header.line.count"="1");
```

### 7. Store Output Data in hive
```bash
  # update the file path of output csv file where you stored the results.
  LOAD DATA INPATH '<Path_CSV_File_From_HDFS>' OVERWRITE INTO TABLE netflix_predictions;
```
