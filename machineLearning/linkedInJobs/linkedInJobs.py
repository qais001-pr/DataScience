import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if len(sys.argv) > 1:
    input_path = sys.argv[1]

spark = SparkSession.builder \
    .appName("JobTypePrediction") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

print("Reading data...")
df = spark.read.csv(input_path, header=True, inferSchema=True)

df = df.select('job_title', 'company', 'job_location', 'job_level', 'job_type') \
       .na.drop()

# Optional sampling (for testing only)
# df = df.sample(withReplacement=False, fraction=0.01, seed=42)

indexers = [
    StringIndexer(inputCol="job_title", outputCol="job_title_index", handleInvalid="skip"),
    StringIndexer(inputCol="company", outputCol="company_index", handleInvalid="skip"),
    StringIndexer(inputCol="job_location", outputCol="job_location_index", handleInvalid="skip"),
    StringIndexer(inputCol="job_level", outputCol="job_level_index", handleInvalid="skip"),
    StringIndexer(inputCol="job_type", outputCol="job_type_index", handleInvalid="skip")
]

assembler = VectorAssembler(
    inputCols=["job_title_index", "company_index", "job_location_index", "job_level_index"],
    outputCol="features"
)

train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

lr = LogisticRegression(
    labelCol="job_type_index",
    featuresCol="features",
    maxIter=20,
    regParam=0.1,
    elasticNetParam=0
)

pipeline = Pipeline(stages=indexers + [assembler, lr])

print("Training model...")
model = pipeline.fit(train_data)

print("Running predictions...")
predictions = model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(
    labelCol="job_type_index",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"\nModel Test Accuracy: {accuracy:.2f}\n")

# OPTIONAL: Only run if df_missing_label is properly defined
df_missing_label = df.filter(df["job_type"].isNull())
predicted_missing = model.transform(df_missing_label)

label_converter = IndexToString(
    inputCol="prediction",
    outputCol="predicted_job_type",
    labels=model.stages[-1].labels
)
final_predictions = label_converter.transform(predicted_missing)
final_predictions.select("job_title", "company", "job_location", "job_level", "predicted_job_type") \
                 .show(50, truncate=False)

spark.stop()

