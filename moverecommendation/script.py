#!/usr/bin/env python
# coding: utf-8

"""
MovieLens Recommender with ALS (Spark MLlib) — Ratings Only

Usage:
    python script.py <ratings_path> <movies_path>

Example:
    python script.py hdfs:///user/<name>/ratings.csv hdfs:///user/<name>/movies.csv
"""

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


def main():
    # Ensure correct arguments
    if len(sys.argv) != 3:
        print("Usage: python script.py <ratings_path> <movies_path>")
        sys.exit(1)

    ratings_path = sys.argv[1]
    movies_path = sys.argv[2]

    # 1) Spark Session
    spark = (SparkSession.builder
             .appName("MovieLens-ALS")
             .getOrCreate())

    # 2) Load Ratings Data
    df = (spark.read
          .option("header", "true")
          .csv(ratings_path))
    print("Ratings data sample:")
    df.show(5)

    # 3) Select required columns
    ratings = df.select("userId", "movieId", "rating")
    ratings.show(10, truncate=False)

    # 4) Split Data
    train, test = ratings.randomSplit([0.8, 0.2], seed=42)

    # 5) Build ALS Model
    als = ALS(
        maxIter=10,
        regParam=0.1,
        rank=10,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop"
    )

    # 6) Cast datatypes
    train = train.withColumn("userId", col("userId").cast("integer")) \
                 .withColumn("movieId", col("movieId").cast("integer")) \
                 .withColumn("rating", col("rating").cast("float"))

    test = test.withColumn("userId", col("userId").cast("integer")) \
               .withColumn("movieId", col("movieId").cast("integer")) \
               .withColumn("rating", col("rating").cast("float"))

    # Fit ALS
    model = als.fit(train)

    # 7) Evaluate Model
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"Root-mean-square error = {rmse:.3f}")

    # 8) Recommendations
    movies = spark.read.csv(movies_path, header=True, inferSchema=True)

    # User Recommendations
    user_recs = model.recommendForAllUsers(10)
    user_recs_exploded = user_recs.withColumn("rec", explode("recommendations")) \
                                  .select("userId", "rec.movieId", "rec.rating")
    user_recs_with_movies = user_recs_exploded.join(movies, on="movieId")
    print("Sample user recommendations (with movie titles):")
    user_recs_with_movies.show(10, truncate=False)

    # Movie Recommendations
    movie_recs = model.recommendForAllItems(2)
    movie_recs_exploded = movie_recs.withColumn("rec", explode("recommendations")) \
                                    .select("movieId", "rec.userId", "rec.rating")
    movie_recs_with_movies = movie_recs_exploded.join(movies, on="movieId")
    print("Sample movie recommendations (with userIds):")
    movie_recs_with_movies.show(10, truncate=False)

    # 9) Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()
