import sys
import os
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import explode, col

def save_to_hdfs(local_path, hdfs_path):
    """Save local file to HDFS, overwrite if exists."""
    os.system(f"hdfs dfs -put -f {local_path} {hdfs_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <ratings_file_hdfs> <movies_file_hdfs>")
        sys.exit(1)

    ratings_file = sys.argv[1]  # Example: hdfs:///user/qais001/input/userRatingData.csv
    movies_file = sys.argv[2]   # Example: hdfs:///user/qais001/input/movies.csv

    # ------------------- Start Spark Session -------------------
    spark = SparkSession.builder \
        .appName("Netflix ALS Recommendation System with Graphs") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    # ------------------- Load Data -------------------
    ratings_df = spark.read.csv(ratings_file, header=True, inferSchema=True).dropna()
    ratings_df = ratings_df.withColumn("CustId", ratings_df["CustId"].cast("integer")) \
                           .withColumn("MovieId", ratings_df["MovieId"].cast("integer")) \
                           .withColumn("Rating", ratings_df["Rating"].cast("float"))

    movies_df = spark.read.csv(movies_file, header=True, inferSchema=True)

    # ------------------- Train-Test Split -------------------
    train, test = ratings_df.randomSplit([0.8, 0.2], seed=42)

    # ------------------- ALS Model -------------------
    als = ALS(userCol="CustId", itemCol="MovieId", ratingCol="Rating", coldStartStrategy="drop")
    model = als.fit(train)

    # ------------------- Predictions -------------------
    predictions = model.transform(test)
    predictions_with_movies = predictions.join(movies_df, on="MovieId", how="left")
    # Save test predictions
    # Update the file path where you want to save the results
    predictions_with_movies.write.mode("overwrite").option("header", True) \
        .csv("hdfs:///user/qais/moviesrating/output/predictions_test")

    # ------------------- Evaluation -------------------
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="Rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"Root-Mean-Square Error (RMSE): {rmse}")

    # ------------------- User Recommendations -------------------
    userRecs = model.recommendForAllUsers(10)
    userRecsExploded = userRecs.withColumn("rec", explode(col("recommendations"))) \
                               .select("CustId", col("rec.MovieId"), col("rec.rating"))
    userRecsWithMovies = userRecsExploded.join(movies_df, on="MovieId", how="left")
    # Save top 10 user recommendations
    # Update the file path where you want to save the results
    userRecsWithMovies.write.mode("overwrite").option("header", True) \
        .csv("hdfs:///user/qais/moviesrating/output/user_recommendations")

    # ------------------- Movie Recommendations -------------------
    movieRecs = model.recommendForAllItems(10)
    movieRecsExploded = movieRecs.withColumn("rec", explode(col("recommendations"))) \
                                 .select("MovieId", col("rec.CustId"), col("rec.rating"))
    movieRecsWithTitles = movieRecsExploded.join(movies_df, on="MovieId", how="left")
    # Save top 10 movie recommendations
     # Update the file path where you want to save the results
    movieRecsWithTitles.write.mode("overwrite").option("header", True) \
        .csv("hdfs:///user/qais/moviesrating/output/movie_recommendations")

    # ------------------- Visualizations -------------------
    # Sample small dataset to Pandas for plotting
    ratings_pd = ratings_df.limit(100000).toPandas()
    user_recs_pd = userRecsWithMovies.filter(userRecsWithMovies["CustId"] == 1).limit(10).toPandas()
    popular_movies_pd = ratings_df.groupBy("MovieId").count() \
                                  .orderBy("count", ascending=False).limit(10) \
                                  .join(movies_df, "MovieId") \
                                  .toPandas()

    tmp_dir = "/tmp"

    # 1. Ratings Distribution
    plt.figure(figsize=(6,4))
    plt.hist(ratings_pd["Rating"], bins=10, color="skyblue", edgecolor="black")
    plt.title("Ratings Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    local_path = os.path.join(tmp_dir, "ratingDistribution.png")
    plt.savefig(local_path)
     # Update the file path where you want to save the graph
    save_to_hdfs(local_path, "hdfs:///user/qais/moviesrating/graphs/ratingDistribution.png")
    plt.close()

    # 2. RMSE Bar Plot
    plt.figure(figsize=(4,4))
    plt.bar(["ALS Model"], [rmse], color="green")
    plt.ylabel("RMSE")
    plt.title("Model Performance (RMSE)")
    local_path = os.path.join(tmp_dir, "modelPerformance.png")
    plt.savefig(local_path)
     # Update the file path where you want to save the graph
    save_to_hdfs(local_path, "hdfs:///user/qais/moviesrating/graphs/modelPerformance.png")
    plt.close()

    # 3. Top 10 Recommendations for User 1
    if not user_recs_pd.empty:
        plt.figure(figsize=(10,5))
        plt.barh(user_recs_pd["MovieTitle"], user_recs_pd["rating"], color="orange")
        plt.xlabel("Predicted Rating")
        plt.ylabel("Movies")
        plt.title("Top 10 Recommendations for User 1")
        plt.gca().invert_yaxis()
        local_path = os.path.join(tmp_dir, "top10User1Recommendations.png")
        plt.savefig(local_path)
        # Update the file path where you want to save the graph
        save_to_hdfs(local_path, "hdfs:///user/qais/moviesrating/graphs/top10User1Recommendations.png")
        plt.close()

    # 4. Top 10 Most Rated Movies
    plt.figure(figsize=(10,5))
    plt.barh(popular_movies_pd["MovieTitle"], popular_movies_pd["count"], color="purple")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Movies")
    plt.title("Top 10 Most Rated Movies")
    plt.gca().invert_yaxis()
    local_path = os.path.join(tmp_dir, "top10MostRatedMovies.png")
    plt.savefig(local_path)
     # Update the file path where you want to save the graph
    save_to_hdfs(local_path, "hdfs:///user/qais/moviesrating/graphs/top10MostRatedMovies.png")
    plt.close()

    print("Graphs saved to HDFS.")
    spark.stop()

if __name__ == "__main__":
    main()
