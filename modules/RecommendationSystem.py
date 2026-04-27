import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# -------------------------------
# 🔥 CONTENT-BASED FILTERING
# -------------------------------
def content_based_recommendation(df, index=0, top_n=5):
    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(df["text"])

    # only compare with selected index (memory efficient)
    similarity = cosine_similarity(tfidf_matrix[index:index+1], tfidf_matrix).flatten()

    scores = list(enumerate(similarity))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_posts = [df.iloc[i[0]]["text"] for i in scores[1:top_n+1]]

    return top_posts


# -------------------------------
# 🔥 COLLABORATIVE FILTERING
# -------------------------------
def collaborative_filtering(df, top_n=5):
    df["user_id"] = df.index % 10  # simulate users

    pivot = df.pivot_table(index="user_id", values="text", aggfunc="count").fillna(0)

    similarity = cosine_similarity(pivot)

    similar_users = similarity[0].argsort()[::-1][1:top_n+1]

    recommended_posts = df[df["user_id"].isin(similar_users)]["text"].head(top_n).tolist()

    return recommended_posts


# -------------------------------
# 🔥 MAIN FUNCTION
# -------------------------------
def get_recommendations(df):
    content_rec = content_based_recommendation(df)
    collab_rec = collaborative_filtering(df)

    return {
        "content_based": content_rec,
        "collaborative": collab_rec
    }


# -------------------------------
# 🔥 RUN
# -------------------------------
if __name__ == "__main__":
    print("Running Recommendation System...")

    df = pd.read_csv("twitter_training.csv")

    # Fix columns
    df.columns = ["id", "topic", "sentiment", "text"]

    # 🔥 FIX 1: Remove missing values
    df = df.dropna(subset=["text"])

    # 🔥 FIX 2: Reduce dataset size (VERY IMPORTANT)
    df = df.sample(2000)

    df["text"] = df["text"].astype(str)

    result = get_recommendations(df)

    print("OUTPUT:\n", result)

    

    