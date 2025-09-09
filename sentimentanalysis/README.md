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
- Python 3.x  
- Jupyter Notebook or `.py` script  

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
- [Download Presentation (PPTX)](data/slides.pptx)
## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone [https://github.com/yourusername/sentiment-analysis-pyspark.git](https://github.com/Faizi0952112/Sentiment-Analysis-on-Tweets.git)
```
### 2. How to run in virtual enviroment
```bash
spark-submit
--master yarn
--deploy-mode cluster  
--archives hdfs:///user/faiz/myenv.tar.gz#environment
--conf spark.pyspark.python=environment/bin/python
--conf spark.pyspark.driver.python=environment/bin/python
script.py hdfs:///<filepath>
