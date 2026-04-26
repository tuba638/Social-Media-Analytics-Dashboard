from flask import Flask, render_template
import pandas as pd

from modules.sentiment import analyze_sentiment, analyze_sentiment_by_topic
from modules.trending import get_trends
from modules.visualization import prepare_chart_data

app = Flask(__name__)

# ✅ LOAD FULL DATASET (no head, no sample)
df = pd.read_csv("twitter_training.csv", encoding="latin-1")

df.columns = ["id", "topic", "sentiment", "text"]

# Clean
df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str)

# Convert once
posts = df.to_dict(orient="records")

# ✅ PRECOMPUTE EVERYTHING ONCE
sentiments, accuracy = analyze_sentiment(posts)
topic_sentiments = analyze_sentiment_by_topic(posts)
trends = get_trends(posts)
labels, values = prepare_chart_data(sentiments)


@app.route("/")
def home():
    return render_template(
        "index.html",
        sentiments=sentiments,
        topic_sentiments=topic_sentiments,
        trends=trends,
        accuracy=accuracy,
        labels=labels,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)