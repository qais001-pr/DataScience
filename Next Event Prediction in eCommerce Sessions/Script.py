#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour, dayofweek
from pyspark.sql.window import Window
from pyspark.sql.functions import lead
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = "hdfs:///user/faiz/next_event_output"

    # 1. Spark session
    spark = SparkSession.builder.appName("NextEventPrediction").getOrCreate()

    # 2. Read dataset
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Convert event_time to timestamp
    df = df.withColumn("event_time", to_timestamp(col("event_time")))

    # 3. Create next_event column using window function
    window_spec = Window.partitionBy("user_session").orderBy("event_time")
    df = df.withColumn("next_event", lead("event_type").over(window_spec))
    df = df.filter(col("next_event").isNotNull())

    # Extract hour and day of week
    df = df.withColumn("hour", hour("event_time")) \
           .withColumn("dayofweek", dayofweek("event_time"))

    # 4. Feature Engineering
    indexers = [
        StringIndexer(inputCol="event_type", outputCol="event_type_index"),
        StringIndexer(inputCol="next_event", outputCol="label")
    ]

    assembler = VectorAssembler(
        inputCols=["event_type_index", "price", "hour", "dayofweek"],
        outputCol="features"
    )

    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)

    pipeline = Pipeline(stages=indexers + [assembler, rf])

    # 5. Train-test split
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    # 6. Model training
    model = pipeline.fit(train)

    # 7. Prediction
    predictions = model.transform(test)

    # 8. Evaluation
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

    # 9. Save results
    predictions.select("user_session", "event_type", "next_event", "label", "prediction") \
        .write.mode("overwrite").csv(output_path, header=True)

    print(f"\nResults saved to: {output_path}")

    spark.stop()


if __name__ == "__main__":
    main()
