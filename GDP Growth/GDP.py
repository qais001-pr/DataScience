import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: GDP.py <input_csv_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    spark = SparkSession.builder.appName("Country_Income_Prediction").getOrCreate()

    df = spark.read.csv(file_path, header=True, inferSchema=True)
    df = df.na.drop()
    indexer = StringIndexer(inputCol="Country", outputCol="Country_index")
    df = indexer.fit(df).transform(df)

    feature_cols = ['2020', '2021', '2022', '2023', '2024']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    train, test = df.randomSplit([0.8, 0.2], seed=42)

    lr = LinearRegression(featuresCol="features", labelCol="2025")
    model = lr.fit(train)

    predictions = model.transform(test)
    predictions.select("Country", "2025", "prediction").show(5)

    evaluator = RegressionEvaluator(labelCol="2025", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"RMSE = {rmse}")

    spark.stop()

