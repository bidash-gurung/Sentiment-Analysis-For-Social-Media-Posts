import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, count, desc, when
import matplotlib.pyplot as plt
import pandas as pd 

spark = SparkSession.builder \
    .appName("DateRangeAnalysis") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

st.title("Sentiment Analysis and Post Distribution by Date Range")

file_path = "hdfs://namenode:8020/cleaned_output/"  # Replace with your HDFS path
df = spark.read.csv(file_path, header=True, inferSchema=True)

df_with_date = df.withColumn("date", to_date(col("date"), "EEE MMM dd HH:mm:ss z yyyy"))

date_distribution = df_with_date.groupBy("date").agg(count("*").alias("count")).orderBy(desc("count"))

top_date_row = date_distribution.first()
top_date = top_date_row["date"]

days_range = 3
start_date = pd.to_datetime(top_date) - pd.Timedelta(days=days_range)
end_date = pd.to_datetime(top_date) + pd.Timedelta(days=days_range)

filtered_df = df_with_date.filter((col("date") >= start_date) & (col("date") <= end_date))

sentiment_distribution = filtered_df.groupBy("sentiment").agg(count("*").alias("count"))

sentiment_data = sentiment_distribution.collect()
sentiment_labels = [row["sentiment"] for row in sentiment_data]
sentiment_values = [row["count"] for row in sentiment_data]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(sentiment_labels, sentiment_values, color="skyblue")
ax.set_title(f"Sentiment Distribution from {start_date.date()} to {end_date.date()}")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")

st.pyplot(fig)

for sentiment, color in [("Positive", "green"), ("Negative", "red")]:
    sentiment_df = filtered_df.filter(col("sentiment") == sentiment).groupBy("date").agg(count("*").alias("count")).orderBy("date")
    sentiment_data_pd = sentiment_df.toPandas()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sentiment_data_pd["date"], sentiment_data_pd["count"], marker="o", linestyle="-", color=color)
    ax.set_title(f"Distribution of {sentiment} Sentiment from {start_date.date()} to {end_date.date()}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.grid(True)

    st.pyplot(fig)

fig2, ax2 = plt.subplots(figsize=(10, 6))
date_data = date_distribution.toPandas()
ax2.plot(date_data["date"], date_data["count"], marker="o", linestyle="-", color="blue")
ax2.set_title("Spike in Posts Over Dates")
ax2.set_xlabel("Date")
ax2.set_ylabel("Number of Posts")
ax2.grid(True)

st.pyplot(fig2)

total_sentiments = date_distribution.agg(count("*").alias("total")).collect()[0]["total"]
st.write(f"Total Number of Sentiments: **{total_sentiments}**")

content_type_distribution = df.groupBy("content_type").agg(count("*").alias("count"))
content_type_data = content_type_distribution.collect()
content_type_labels = [row["content_type"] for row in content_type_data]
content_type_values = [row["count"] for row in content_type_data]

fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3.pie(content_type_values, labels=content_type_labels, autopct='%1.1f%%', startangle=140)
ax3.set_title("Content Type Distribution")

st.pyplot(fig3)

filtered_content_type = filtered_df.filter(col("content_type") != "personal")
content_type_date_distribution = filtered_content_type.groupBy("content_type", "date").agg(count("*").alias("count")).orderBy("date")

content_type_date_pd = content_type_date_distribution.toPandas()
fig4, ax4 = plt.subplots(figsize=(12, 8))
for content_type in content_type_date_pd["content_type"].unique():
    subset = content_type_date_pd[content_type_date_pd["content_type"] == content_type]
    ax4.plot(subset["date"], subset["count"], marker="o", linestyle="-", label=content_type)

ax4.set_title("Content Type Distribution (Excluding Personal) Over Date Range")
ax4.set_xlabel("Date")
ax4.set_ylabel("Count")
ax4.legend()
ax4.grid(True)

st.pyplot(fig4)

spark.stop()
