# LinkedIn Jobs Classification using PySpark

This project predicts the **job type** (e.g., Full-time, Part-time, Contract) from LinkedIn job postings using **Apache Spark** and **Machine Learning (Logistic Regression)**.

---

## 🚀 Project Overview

- Reads job postings data from **CSV stored on HDFS**
- Cleans and prepares the data
- Uses **StringIndexer** and **VectorAssembler**
- Trains a **Logistic Regression** model using Spark MLlib
- Evaluates model accuracy on a test set

---
## DataSet

[1.3M Linkedin Jobs & Skills](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024?select=linkedin_job_postings.csv) 

## Presentation 

[Download](/linkedInJobs/docs/slides.pptx)

## 📂 Input Data

- Format: CSV
- Location: HDFS  
  Example path: `hdfs:///user/qais001/input/linkedInJobs/data.csv`
- Required Columns:
  - `job_title`
  - `company`
  - `job_location`
  - `job_level`
  - `job_type` (target)

---

## 🛠️ How to Run

Use the following `spark-submit` command:

```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --driver-memory 3g \
  --executor-memory 4g \
  --num-executors 2 \
  --executor-cores 2 \
  /path/to/linkedInJobs.py \
  hdfs:///<path>
```
