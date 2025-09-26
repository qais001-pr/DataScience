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


## How to Run

---
## **Setup & Usage**

### 1. Clone the repository:

```bash
git clone https://github.com/qais001-pr/Data_Science_Projects.git
```
## 2. Locate this folder
```bash
cd linkedInJobs
```

### 🛠 Requirements

- Python 3.6.8
- Apache Spark (with PySpark)
- Java 8+

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

[Download](https://github.com/qais001-pr/DataScience/tree/main/machineLearning/linkedInJobs/docs)


### Use the following `spark-submit` command in terminal

```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --driver-memory 3g \
  --executor-memory 4g \
  --num-executors 2 \
  --executor-cores 2 \
  #/home/qais/Downloads/DataScience/machineLearning/linkedInJobs/script.py
  /path/to/linkedInJobs.py \
  #hdfs///user/qais/linkedInJobs/linked.csv
  hdfs:///<path>
```
