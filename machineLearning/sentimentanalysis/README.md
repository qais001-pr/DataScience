# 🧠 Sentiment Analysis on Tweets using PySpark (TF-IDF + Logistic Regression)

This project implements a sentiment analysis pipeline using PySpark on the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140). It classifies tweets as *positive* or *negative* based on their textual content using *TF-IDF features* and *Logistic Regression*.

---

## 📌 Project Overview

- **Language:** Python (PySpark)  
- **Dataset:** Sentiment140 (1.6M labeled tweets)  
- **Goal:** Classify tweets into positive (1) or negative (0) sentiments  
- **Techniques Used:**  
  - Tokenization  
  - Stopword Removal  
  - TF-IDF Vectorization  
  - Logistic Regression Classification  

---

## 🔧 Tech Stack

- Apache Spark (PySpark)  
- Python 3.6.8  
- `.py` script  

---

## 📂 Dataset Format

The Sentiment140 dataset is a CSV file with no headers. Relevant columns:

- `_c0`: Sentiment label (0 = negative, 4 = positive)  
- `_c5`: Tweet text  

> Only these two columns are used in this project.

**Download Dataset:** [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)  

---

## 📦 Required Libraries

Make sure the following Python libraries are installed in your environment:

```text
pyspark
pandas
numpy
matplotlib   # optional, for visualizations
seaborn      # optional, for visualizations
```
## Presentation 
- [Download Presentation (PPTX)](data/Sentiment-Analysis-of-Tweets-with-PySpark.pptx)
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

[Download](https://github.com/Faizi0952112/DataScience-Projects/blob/main/machineLearning/sentimentanalysis/data/) 


```bash

# Update the file Path of requirement file  to setup the virtual env
# File Path Like
# /workspaces/DataScience/machineLearning/movierecommendation/docs/requirements.txt
pip install -r <reuiqrementFilePath>

```

### 1. Clone the Repository

```bash
git clone [https://github.com/yourusername/sentiment-analysis-pyspark.git](https://github.com/Faizi0952112/DataScience-Projects.git)
```
### 2. How to run in virtual enviroment
```bash
spark-submit
--master yarn
--deploy-mode cluster  
--archives hdfs:///user/<name>/myenv.tar.gz#environment
--conf spark.pyspark.python=environment/bin/python
--conf spark.pyspark.driver.python=environment/bin/python
script.py hdfs:///<filepath>
# /user/faiz/input/Sentiment_Tweets

