from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
import spacy

spark = SparkSession.builder \
    .appName("LemmatizeAndAnalyze") \
    .getOrCreate()

file_path = "hdfs://namenode:8020/input/newData.csv" 
df = spark.read.csv(file_path, header=True, inferSchema=True)

nlp = spacy.load("en_core_web_sm")

def lemmatize_text(text):
    if text:
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])
    return None

lemmatize_udf = udf(lemmatize_text, StringType())

df_with_lemmas = df.withColumn("lemmatized_text", lemmatize_udf(df["filtered_text"]))

positive_keywords = {
    "love", "great", "awesome", "amazing", "fantastic", "like", "enjoy",
    "happy", "joy", "wonderful", "best", "lit", "blessed", "excited",
    "vibes", "winning", "yass", "woot", "cheers", "sweet", "goals",
    "fave", "smile", "adorable", "exciting", "grateful"
}

negative_keywords = {
    "hate", "terrible", "bad", "awful", "worst", "dislike", "annoying",
    "sad", "angry", "disappointed", "frustrated", "miserable", "sucks",
    "garbage", "trash", "fail", "meh", "yikes", "nope", "rage", "cringe"
}

neutral_keywords = {
    "okay", "fine", "average", "meh", "normal", "neutral", "indifferent",
    "whatever", "just", "kinda", "sorta", "not bad"
}

informational_keywords = {
    "news", "update", "learn", "tips", "guide", "info", "report", "thread",
    "breaking", "analysis", "research", "insight", "fact", "data", "watch",
    "check", "details", "what", "how"
}

promotional_keywords = {
    "discount", "sale", "offer", "buy", "deal", "free", "promo", "limited",
    "exclusive", "giveaway", "event", "contest", "bargain", "clearance", 
    "shop", "steal", "price", "coupon", "save"
}

personal_keywords = {
    "i", "me", "my", "we", "us", "mine", "our", "self", "personal",
    "feel", "thought", "opinion", "story", "experience", "share", "vent"
}

humor_keywords = {
    "joke", "funny", "meme", "lol", "lmao", "rofl", "hilarious",
    "laugh", "comedy", "silly", "humorous", "witty", "puns", "sarcastic",
    "quirky", "banter", "giggle"
}

def detect_sentiment(text):
    if text is None:
        return "Neutral"  
    text = text.lower()
    if any(word in text for word in positive_keywords):
        return "Positive"
    elif any(word in text for word in negative_keywords):
        return "Negative"
    elif any(word in text for word in neutral_keywords):
        return "Neutral"
    else:
        return "Neutral"  

def detect_content_type(text):
    if text is None:
        return "General"  
    text = text.lower()
    if any(word in text for word in informational_keywords):
        return "Informational"
    elif any(word in text for word in promotional_keywords):
        return "Promotional"
    elif any(word in text for word in personal_keywords):
        return "Personal"
    elif any(word in text for word in humor_keywords):
        return "Humor"
    else:
        return "General"  

sentiment_udf = udf(detect_sentiment, StringType())
content_type_udf = udf(detect_content_type, StringType())

df_with_labels = df_with_lemmas \
    .withColumn("sentiment", sentiment_udf(col("lemmatized_text"))) \
    .withColumn("content_type", content_type_udf(col("lemmatized_text")))

output_path = "hdfs://namenode:8020/finalpreprocessing/"  # Replace with your desired HDFS output path
df_with_labels.coalesce(1).write.csv(output_path, header=True, mode="overwrite")

spark.stop()
