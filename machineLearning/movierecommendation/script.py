#!/usr/bin/env python
# coding: utf-8

"""
MovieLens Recommender with ALS (Spark MLlib) - Ratings Only + Save Results
"""

import sys
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import os

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <ratings_path> <movies_path>")
        sys.exit(1)

    ratings_path = sys.argv[1]
    movies_path = sys.argv[2]

    # Optional: Output directory

    
    # update the file path where you want to save the results
    output_dir = "hdfs:///user/qais/MovieLens/output"
    os.makedirs(output_dir, exist_ok=True)

    spark = (SparkSession.builder
             .appName("MovieLens-ALS")
             .getOrCreate())

    df = spark.read.option("header", "true").csv(ratings_path)
    ratings = df.select("userId", "movieId", "rating")

    # Split
    train, test = ratings.randomSplit([0.8, 0.2], seed=42)

    # ALS
    als = ALS(
        maxIter=10,
        regParam=0.1,
        rank=10,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop"
    )

    # Cast types
    train = train.withColumn("userId", col("userId").cast("integer")) \
                 .withColumn("movieId", col("movieId").cast("integer")) \
                 .withColumn("rating", col("rating").cast("float"))

    test = test.withColumn("userId", col("userId").cast("integer")) \
               .withColumn("movieId", col("movieId").cast("integer")) \
               .withColumn("rating", col("rating").cast("float"))

    model = als.fit(train)

    # Predictions
    predictions = model.transform(test)
    predictions.coalesce(1).write.mode("overwrite").option("header", "true").csv(os.path.join(output_dir, "predictions"))

    # Evaluation
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"Root-mean-square error = {rmse:.3f}")

    # Save RMSE to text file
    with open(os.path.join(output_dir, "rmse.txt"), "w") as f:
        f.write(f"Root-mean-square error = {rmse:.3f}\n")

    # Recommendations
    movies = spark.read.csv(movies_path, header=True, inferSchema=True)

    # User Recommendations
    user_recs = model.recommendForAllUsers(10)
    user_recs_exploded = user_recs.withColumn("rec", explode("recommendations")) \
                                  .select("userId", "rec.movieId", "rec.rating")
    user_recs_with_movies = user_recs_exploded.join(movies, on="movieId")
    print("Sample user recommendations (with movie titles):")
    user_recs_with_movies.show(10, truncate=False)
    user_recs_with_movies.coalesce(1).write.mode("overwrite").option("header", "true") \
                          .csv(os.path.join(output_dir, "user_recommendations"))

    # Movie Recommendations
    movie_recs = model.recommendForAllItems(2)
    movie_recs_exploded = movie_recs.withColumn("rec", explode("recommendations")) \
                                    .select("movieId", "rec.userId", "rec.rating")
    movie_recs_with_movies = movie_recs_exploded.join(movies, on="movieId")
    print("Sample movie recommendations (with userIds):")
    movie_recs_with_movies.show(10, truncate=False)
    movie_recs_with_movies.coalesce(1).write.mode("overwrite").option("header", "true") \
                            .csv(os.path.join(output_dir, "movie_recommendations"))

    spark.stop()

if __name__ == "__main__":
    main()
