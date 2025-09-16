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

- Python 3.x
- PySpark
- HDFS (for output storage)
- CSV datasets:
  - `ratings.csv` → Columns: `CustId, MovieId, Rating`
  - `movies.csv` → Columns: `MovieId, MovieTitle`

---
## **Setup & Usage**

1. Clone the repository:

```bash
git clone https://github.com/qais001-pr/netflix-recommendation.git
```
2. Run this Command in Terminal:

```bash
spark-submit \
--master yarn  \
   --deploy-mode client  \
    --driver-memory 4G \
    --executor-memory 4G  \
    "<pythonScript>"  \
    "hdfs://<userRatingData>" \
    "hdfs://<moviesFilePath>" \
```
3. Create Database
```bash
CREATE DATABASE NETFLIX_DATA;
```
4. Use DATABASE and Create Table
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
5. Store Output Data in hive
```bash
  LOAD DATA INPATH '<Path_CSV_File_From_HDFS>' OVERWRITE INTO TABLE netflix_predictions;
```
