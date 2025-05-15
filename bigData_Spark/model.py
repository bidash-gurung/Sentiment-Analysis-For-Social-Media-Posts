import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.sql.functions import when

os.environ['JAVA_HOME'] = '/opt/bitnami/java'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'

spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

def load_data(csv_path):
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    df.printSchema()  
    return df

def preprocess_data(df):
    if 'lemmatized_text' not in df.columns:
        raise ValueError("Column 'lemmatized_text' not found in the dataset.")

    df = df.filter(df['lemmatized_text'].isNotNull())
    df = df.withColumn("lemmatized_text", col("lemmatized_text").cast("string"))

    tokenizer = Tokenizer(inputCol="lemmatized_text", outputCol="words")
    hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="features")

    pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])
    model = pipeline.fit(df)
    processed_data = model.transform(df)

    processed_data.show(5)  
    return processed_data

def train_and_save_model(csv_path, hdfs_model_path):
    df = load_data(csv_path)
    processed_data = preprocess_data(df)

    processed_data = processed_data.withColumn(
        "label", 
        when(processed_data["sentiment"] == "Negative", 0)
        .when(processed_data["sentiment"] == "Neutral", 1)
        .when(processed_data["sentiment"] == "Positive", 2)
        .otherwise(3)  
    )

    assembler = VectorAssembler(inputCols=["features"], outputCol="final_features")
    final_data = assembler.transform(processed_data)

    train_df, test_df = final_data.randomSplit([0.8, 0.2], seed=42)

    lr = LogisticRegression(featuresCol="final_features", labelCol="label", family="multinomial")
    model = lr.fit(train_df)

    predictions = model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Model accuracy: {accuracy:.2f}")

    model.write().overwrite().save(hdfs_model_path)

    print(f"Model saved to HDFS at {hdfs_model_path}")

if __name__ == "__main__":
    csv_path = "hdfs://namenode:8020/cleaned_output/final_cleaned_output/"
    hdfs_model_path = "hdfs://namenode:8020/output/sentiment_model"
    train_and_save_model(csv_path, hdfs_model_path)
