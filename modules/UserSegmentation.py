import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json

# -------------------------------
# 🔥 FEATURE ENGINEERING
# -------------------------------
def create_features(df):
    # Behavior feature: text length
    df["text_length"] = df["text"].astype(str).apply(len)

    # Behavior feature: sentiment encode
    le = LabelEncoder()
    df["sentiment_encoded"] = le.fit_transform(df["sentiment"])

    # Demographic simulation (topic as category)
    df["topic_encoded"] = LabelEncoder().fit_transform(df["topic"].astype(str))

    return df


# -------------------------------
# 🔥 CLUSTERING
# -------------------------------
def perform_clustering(df, k=3):
    features = df[["text_length", "sentiment_encoded", "topic_encoded"]]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df["cluster"] = kmeans.fit_predict(scaled)

    return df


# -------------------------------
# 🔥 MAIN RUN
# -------------------------------
if __name__ == "__main__":
    print("Running User Segmentation...")

    df = pd.read_csv("twitter_training.csv")

    # Fix columns
    df.columns = ["id", "topic", "sentiment", "text"]

    # 🔥 CLEANING
    df = df.dropna(subset=["text"])
    df = df.sample(2000)
    df["text"] = df["text"].astype(str)

    # Features
    df = create_features(df)

    # Clustering
    df = perform_clustering(df, k=3)

    # Output
    output = df[["text", "cluster"]].head(10).to_dict(orient="records")

    print("OUTPUT:\n", output)

    