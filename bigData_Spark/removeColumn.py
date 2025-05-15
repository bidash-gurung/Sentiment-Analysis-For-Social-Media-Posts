from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("RemoveColumns") \
    .getOrCreate()

input_path = "hdfs://namenode:8020/finalpreprocessing/"  
df_with_labels = spark.read.csv(input_path, header=True, inferSchema=True)

columns_to_drop = ["id", "user", "flag"]
df_cleaned = df_with_labels.drop(*columns_to_drop)

output_path = "hdfs://namenode:8020/cleaned_output/"  
df_cleaned.coalesce(1).write.csv(output_path, header=True, mode="overwrite")

spark.stop()
