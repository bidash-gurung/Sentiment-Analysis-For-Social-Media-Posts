import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.functions import col
import os
from pyspark.ml import Pipeline


os.environ['JAVA_HOME'] = '/opt/bitnami/java'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'

spark = SparkSession.builder \
    .appName("SentimentAnalysisDashboard") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

def load_model_from_hdfs(model_path):
    model = LogisticRegressionModel.load(model_path)
    return model

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

    return processed_data

def make_predictions(model, input_text):
    input_data = spark.createDataFrame([(input_text,)], ["lemmatized_text"])

    processed_data = preprocess_data(input_data)

    assembler = VectorAssembler(inputCols=["features"], outputCol="final_features")
    final_data = assembler.transform(processed_data)

    predictions = model.transform(final_data)

    predictions = predictions.select("prediction").collect()
    sentiment = predictions[0][0]

    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_labels.get(sentiment, "Unknown")

def run_dashboard():
    st.title("Sentiment Analysis Dashboard")
    st.write("""
        This is a sentiment analysis tool that predicts the sentiment of the given text.
        You can enter a sentence, and the model will predict if the sentiment is Negative, Neutral, or Positive.
    """)

    model_path = "hdfs://namenode:8020/output/sentiment_model"
    model = load_model_from_hdfs(model_path)

    input_text = st.text_area("Enter Text for Sentiment Analysis", "Type your text here...")

    if st.button("Predict Sentiment"):
        if input_text.strip() != "":
            with st.spinner('Making Prediction...'):
                sentiment = make_predictions(model, input_text)
                st.write(f"Sentiment Prediction: **{sentiment}**")
        else:
            st.write("Please enter some text to analyze.")

if __name__ == "__main__":
    run_dashboard()
