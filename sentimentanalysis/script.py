#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = "hdfs:///user/faiz/sentiment_output"

    # 1. Spark session
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

    # 2. Read dataset (no header in Sentiment140)
    df_raw = spark.read.csv(input_path, header=False)

    # Select relevant columns: _c0 = label, _c5 = tweet
    df = df_raw.select(
        col("_c0").cast("int").alias("label"),
        col("_c5").alias("tweet")
    )

    # Convert labels: 0 ? 0 (negative), 4 ? 1 (positive)
    df = df.withColumn("label", when(col("label") == 4, 1).otherwise(0))

    # Drop null/empty tweets
    df = df.filter(col("tweet").isNotNull())

    print("Sample data:")
    df.show(5, truncate=False)

    # 3. Text preprocessing pipeline
    tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

    # 4. Train-test split
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    # 5. Model training
    model = pipeline.fit(train)

    # 6. Prediction
    predictions = model.transform(test)

    # 7. Evaluation (Accuracy + more metrics)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # 8. Save results (prediction + actual label)
    predictions.select("tweet", "label", "prediction") \
        .write.mode("overwrite").csv(output_path, header=True)

    print(f"\nResults saved to: {output_path}")

    spark.stop()


if __name__ == "__main__":
    main()
