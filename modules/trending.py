from collections import Counter

def get_trends(posts):
    topics = []

    for post in posts:
        topic = str(post["topic"]).lower()
        topics.append(topic)

    return Counter(topics).most_common(5)