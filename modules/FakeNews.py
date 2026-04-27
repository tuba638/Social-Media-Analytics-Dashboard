import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json

# -------------------------------
# 🔥 DATA PREPARATION
# -------------------------------
def prepare_data(df):
    # Fake = Negative, Real = Positive/Neutral
    df["label"] = df["sentiment"].apply(lambda x: 1 if x == "Negative" else 0)
    return df


# -------------------------------
# 🔥 TRAIN MODEL
# -------------------------------
def train_model(df):
    X = df["text"]
    y = df["label"]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, vectorizer, accuracy


# -------------------------------
# 🔥 PREDICTION FUNCTION
# -------------------------------
def predict_fake_news(model, vectorizer, texts):
    X_new = vectorizer.transform(texts)
    preds = model.predict(X_new)

    return ["Fake" if p == 1 else "Real" for p in preds]


# -------------------------------
# 🔥 MAIN RUN
# -------------------------------
if __name__ == "__main__":
    print("Running Fake News Detection...")

    df = pd.read_csv("twitter_training.csv")

    # Fix columns
    df.columns = ["id", "topic", "sentiment", "text"]

    # 🔥 IMPORTANT FIXES
    df = df.dropna(subset=["text"])   # remove NaN
    df = df.sample(2000)             # reduce size (memory safe)
    df["text"] = df["text"].astype(str)

    df = prepare_data(df)

    model, vectorizer, accuracy = train_model(df)

    # sample predictions
    sample_texts = df["text"].head(5).tolist()
    predictions = predict_fake_news(model, vectorizer, sample_texts)

    output = {
        "accuracy": float(accuracy),
        "results": list(zip(sample_texts, predictions))
    }

    print("OUTPUT:\n", output)
