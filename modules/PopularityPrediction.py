import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import json

# -------------------------------
# 🔥 FEATURE ENGINEERING
# -------------------------------
def create_features(df):
    df["text_length"] = df["text"].astype(str).apply(len)

    df["sentiment_encoded"] = LabelEncoder().fit_transform(df["sentiment"])
    df["topic_encoded"] = LabelEncoder().fit_transform(df["topic"].astype(str))

    # simulate engagement (since real nahi hai)
    df["engagement"] = df["text_length"] * 0.5 + df["sentiment_encoded"] * 10

    return df


# -------------------------------
# 🔥 TRAIN MODEL
# -------------------------------
def train_model(df):
    X = df[["text_length", "sentiment_encoded", "topic_encoded"]]
    y = df["engagement"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    return model, score


# -------------------------------
# 🔥 PREDICT
# -------------------------------
def predict(model, df):
    X = df[["text_length", "sentiment_encoded", "topic_encoded"]]
    preds = model.predict(X)

    return preds.tolist()


# -------------------------------
# 🔥 MAIN RUN
# -------------------------------
if __name__ == "__main__":
    print("Running Popularity Prediction...")

    df = pd.read_csv("twitter_training.csv")

    df.columns = ["id", "topic", "sentiment", "text"]

    # 🔥 CLEANING
    df = df.dropna(subset=["text"])
    df = df.sample(2000)
    df["text"] = df["text"].astype(str)

    df = create_features(df)

    model, score = train_model(df)

    predictions = predict(model, df.head(5))

    output = {
        "model_score": score,
        "predictions": predictions
    }

    print("OUTPUT:\n", output)

    