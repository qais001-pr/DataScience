import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, dayofweek, unix_timestamp, abs as abs_diff
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

def main():
    if len(sys.argv) != 2:
        print("Usage: script.py <file_path>")
        sys.exit(1)
        
    filePath = sys.argv[1]
    
    spark = SparkSession.builder \
        .appName('NYC Taxi Fare Predictions') \
        .getOrCreate()

    df = spark.read.csv(filePath, header=True, inferSchema=True)

    df = df.withColumn("pickup_hour", hour("tpep_pickup_datetime"))
    df = df.withColumn("pickup_day", dayofweek("tpep_pickup_datetime"))
    df = df.withColumn("duration_minutes", 
        (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60)

    df = df.na.drop()

    feature_cols = ["passenger_count", "trip_distance", "pickup_hour", "pickup_day", "duration_minutes"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    final_data = assembler.transform(df)

    train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

    lr = LinearRegression(featuresCol="features", labelCol="total_amount")
    lr_model = lr.fit(train_data)

    test_results = lr_model.evaluate(test_data)
    print("RMSE:", round(test_results.rootMeanSquaredError, 2))
    print("R2:", round(test_results.r2, 2))

    predictions = lr_model.transform(test_data)

    predictions.select(
        "passenger_count", 
        "trip_distance", 
        "duration_minutes", 
        "total_amount",     
        "prediction"        
    ).show(20, truncate=False)

    predictions = predictions.withColumn("error", abs_diff(predictions["total_amount"] - predictions["prediction"]))
    predictions.select("total_amount", "prediction", "error") \
        .orderBy("error", ascending=False) \
        .show(5, truncate=False)

    spark.stop()

if __name__ == '__main__':
    main()
