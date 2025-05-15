import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
import subprocess

# Initialize Spark session
spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Load the CSV data into a Spark DataFrame
def load_data(csv_path):
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    return df

# Preprocess text data: Tokenize and vectorize text columns
def preprocess_data(df):
    # Ensure lemmatized_text exists and is valid
    if 'lemmatized_text' not in df.columns:
        raise ValueError("Column 'lemmatized_text' not found in the dataset.")

    # Handle missing or null values
    df = df.filter(df['lemmatized_text'].isNotNull())
    df = df.withColumn("lemmatized_text", col("lemmatized_text").cast("string"))

    # Tokenizer to break text into words
    tokenizer = Tokenizer(inputCol="lemmatized_text", outputCol="words")
    # HashingTF to convert words into feature vectors
    hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=10000)
    # IDF to scale the feature vectors
    idf = IDF(inputCol="raw_features", outputCol="features")

    pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])
    model = pipeline.fit(df)

    processed_data = model.transform(df)
    return processed_data

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=128, input_length=input_shape))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def prepare_data_for_lstm(processed_data):
    pandas_df = processed_data.select('features', 'sentiment').toPandas()

    X = pad_sequences(
        pandas_df['features'].apply(lambda x: x.toArray()).values, 
        padding='post', 
        maxlen=100
    )
    y = pandas_df['sentiment'].apply(lambda x: 1 if x == 'Positive' else 0).values
    return X, y

def save_model_to_hdfs(local_model_path, hdfs_model_path):
    subprocess.run(["hdfs", "dfs", "-put", local_model_path, hdfs_model_path], check=True)
    print(f"Model saved to HDFS at {hdfs_model_path}")

def train_and_save_model(csv_path, hdfs_model_path, local_model_path="sentiment_model.keras"):
    df = load_data(csv_path)
    processed_data = preprocess_data(df)

    X, y = prepare_data_for_lstm(processed_data)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_lstm_model(X_train.shape[1])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64)

    model.save(local_model_path)

    save_model_to_hdfs(local_model_path, hdfs_model_path)

    print(f"Model saved locally at {local_model_path} and uploaded to HDFS at {hdfs_model_path}")

if __name__ == "__main__":
    csv_path = "hdfs://namenode:8020/cleaned_output/final_cleaned_output/"
    hdfs_model_path = "hdfs://namenode:8020/output/sentiment_model.keras"
    train_and_save_model(csv_path, hdfs_model_path)
