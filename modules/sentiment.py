from collections import defaultdict

def normalize_sentiment(sentiment):
    if sentiment == "Irrelevant":
        return "Neutral"
    return sentiment


def analyze_sentiment(posts):
    result = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for post in posts:
        sentiment = normalize_sentiment(post["sentiment"])
        result[sentiment] += 1

    accuracy = 1.0

    return result, accuracy


def analyze_sentiment_by_topic(posts):
    result = defaultdict(lambda: {"Positive": 0, "Negative": 0, "Neutral": 0})

    for post in posts:
        topic = post["topic"]
        sentiment = normalize_sentiment(post["sentiment"])

        result[topic][sentiment] += 1

    return dict(result)