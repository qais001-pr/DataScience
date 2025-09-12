# 🛒 Next Event Prediction in eCommerce Sessions (PySpark)

This project uses **PySpark** and an **eCommerce multi-category store dataset** (Oct 2019) to predict the **next user event** (e.g., view, cart, purchase) in a shopping session.  
The goal is to model user behavior in an eCommerce store and improve understanding of event sequences.

---

## 📂 Dataset
- **File:** `2019-Oct.csv`  
- **Source:** Public eCommerce behavior dataset (multi-category store)  
- **Columns used:**
  - `event_time`: Timestamp of the user action  
  - `event_type`: Type of event (`view`, `cart`, `purchase`)  
  - `price`: Price of the item  
  - `user_session`: Session identifier  

---

## ⚙️ Data Preprocessing
1. Load dataset with PySpark.  
2. Convert `event_time` to timestamp format.  
3. Sort by `user_session` and `event_time` using PySpark window functions.  
4. Generate `next_event` column (shifted event in the same session).  
5. Remove rows where `next_event` is null.  
6. Feature engineering:  
   - Extract `hour` and `dayofweek` from `event_time`.  
   - Encode categorical features (`event_type`, `next_event`).  

---

## 🧠 Model
- **Algorithm:** Random Forest Classifier  
- **Features used:**
  - `event_type_index` (encoded current event)  
  - `price`  
  - `hour` (time of day)  
  - `dayofweek`  
- **Label:** `next_event` (the upcoming event in the session).  

---

## 📊 Pipeline
1. `StringIndexer` for `event_type` and `next_event`  
2. `VectorAssembler` to combine features  
3. `RandomForestClassifier` (50 trees)  
4. Split into **train (80%)** and **test (20%)**  

---

## ✅ Evaluation
- **Metric:** Accuracy (MulticlassClassificationEvaluator)  
- Example output:  
  ```text
  Test Accuracy = 0.72
