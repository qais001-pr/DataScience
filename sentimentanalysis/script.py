#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Analysis with PySpark
# ## Using TF-IDF and Logistic Regression

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
        sys.exit(1)

    sentiment_tweets = sys.argv[1]

    # 1. Initialize Spark Session
    spark = SparkSession.builder \
        .appName("SentimentAnalysisTFIDF") \
        .getOrCreate()

    # 2. Load and Preprocess Data
    df_raw = spark.read.csv(sentiment_tweets, header=False)
    df_raw.show(5)

    # Select label and tweet text
    df = df_raw.select(col("_c0").alias("label"), col("_c5").alias("tweet"))

    # Convert labels: 0 -> 0 (negative), 4 -> 1 (positive)
    df = df.withColumn("label", when(col("label") == 4, 1).otherwise(0))

    # 3. Text Processing Pipeline
    from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml import Pipeline

    tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label")

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

    # 4. Train-Test Split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # 5. Train the Model
    model = pipeline.fit(train_df)

    # 6. Make Predictions
    predictions = model.transform(test_df)

    # 7. Evaluate the Model
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="label"
    )
    accuracy = evaluator.evaluate(predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

    # 8. Predict on Sample Tweets
    predictions.select("tweet", "label", "prediction").show(truncate=False)

    spark.stop()


if __name__ == '__main__':
    main()
