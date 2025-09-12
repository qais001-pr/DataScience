# 📈 GDP Growth Prediction using PySpark (Linear Regression)

This repository contains a **GDP Prediction System** built using **PySpark’s Machine Learning library**.  
It utilizes historical GDP data (2020–2024) of different countries to predict the **GDP for 2025** using **Linear Regression**.

---

## 📌 Project Overview

- Loads and processes a CSV file containing country-wise GDP data from **2020 to 2025**
- Uses **feature engineering** to prepare training data
- Trains a **Linear Regression model** with PySpark MLlib
- Evaluates model performance using **Root Mean Square Error (RMSE)**
- Predicts **GDP for 2025** based on 2020–2024 trends

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Apache Spark (PySpark)**
- **HDFS (Hadoop Distributed File System)**
- **YARN (Yet Another Resource Negotiator)**

---

## 📂 Dataset

The project uses a custom CSV dataset with the following structure:
```
|---------|--------|-----------|----------|---------|---------|---------|
| Country |  2020  |   2021    |   2022   |   2023  |  2024   |  2025   |
|---------|--------|-----------|----------|---------|---------|---------|
| Albania | 15271  | 18086.0   | 19185.0  | 23388.0 | 27259.0 | 28372.0 |
|---------|--------|-----------|----------|---------|---------|---------|

```

👉 Example dataset: [2020-2025.csv](/GDP\Growth/dataset/2020-2025.csv)

---

## 🧰 Libraries Used

- `pyspark.sql`
- `pyspark.ml.feature`
- `pyspark.ml.regression`
- `pyspark.ml.evaluation`

---

## ⚡ Getting Started

### 1. Upload Dataset to HDFS

```bash
hdfs dfs -mkdir -p /user/<username>/input
hdfs dfs -put "/local/path/2020-2025.csv" /user/<username>/input/
```

### 2. Run Pyspark in Terminal

``` bash
pyspark
```

### 3. Run Spark-Submit With Yarn
```bash
spark-submit "<pythonFilePath>" "<dataSetPath>"
```





