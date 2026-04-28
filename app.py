from flask import Flask, render_template
import pandas as pd

# ===== BASIC MODULES =====
from modules.sentiment import analyze_sentiment, analyze_sentiment_by_topic
from modules.trending import get_trends
from modules.visualization import prepare_chart_data

# ===== ADVANCED MODULES =====
from modules.NetworkAnalysis import build_graph, detect_communities, get_influencers
from modules.recommendation_systems import get_recommendations
from modules.fake_news import prepare_data, train_model as fake_news_model
from modules.UserSegmentation import create_features as seg_features, perform_clustering
from modules.AdCampaign import calculate_metrics as ad_metrics
from modules.influencer_detection import detect_influencers
from modules.competitor_analysis import calculate_metrics as comp_metrics, analyze_strategy
from modules.PopularityPrediction import create_features as pop_features, train_model as pop_model, predict
from modules.realtime_monitoring import monitor_keyword

app = Flask(__name__)

# =========================
# 📊 LOAD DATA
# =========================
df = pd.read_csv("twitter_training.csv", encoding="latin-1")
df.columns = ["id", "topic", "sentiment", "text"]

df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str)

# ⚡ performance safe
df = df.sample(3000)

posts = df.to_dict(orient="records")

# =========================
# 🔥 MODULE 1: SENTIMENT
# =========================
sentiments, accuracy = analyze_sentiment(posts)
topic_sentiments = analyze_sentiment_by_topic(posts)

# =========================
# 🔥 MODULE 2: TRENDING
# =========================
trends = get_trends(posts)

# =========================
# 🔥 MODULE 3: VISUALIZATION
# =========================
labels, values = prepare_chart_data(sentiments)

# =========================
# 🔥 MODULE 4: NETWORK
# =========================
# ⚠️ dataset me user/mentions nahi hai → dummy bana rahe
network_posts = [{"user": f"user{i}", "mentions": [f"user{i+1}"]} for i in range(100)]
G = build_graph(network_posts)
communities = detect_communities(G)
influencers_net = get_influencers(G)

# =========================
# 🔥 MODULE 5: RECOMMENDATION
# =========================
recommendations = get_recommendations(df.head(500))

# =========================
# 🔥 MODULE 6: FAKE NEWS
# =========================
df_fake = prepare_data(df.head(2000))
fake_model, vectorizer, fake_acc = fake_news_model(df_fake)

# =========================
# 🔥 MODULE 7: USER SEGMENTATION
# =========================
df_seg = seg_features(df.head(1000))
df_seg = perform_clustering(df_seg)
segments = df_seg[["text", "cluster"]].head(10).to_dict(orient="records")

# =========================
# 🔥 MODULE 8: AD CAMPAIGN
# =========================
# ⚠️ dataset me clicks/impressions nahi → dummy data
ad_dummy = pd.DataFrame({
    "campaign": ["A", "B", "C"],
    "impressions": [1000, 2000, 1500],
    "clicks": [100, 250, 120],
    "conversions": [10, 40, 15],
    "cost": [500, 800, 600],
    "revenue": [1500, 2000, 1200]
})
ad_data = ad_metrics(ad_dummy).to_dict(orient="records")

# =========================
# 🔥 MODULE 9: INFLUENCER (Eigen)
# =========================
influencers_eigen = detect_influencers(G)

# =========================
# 🔥 MODULE 10: REAL-TIME
# =========================
realtime_data = monitor_keyword("technology", iterations=1, delay=1)

# =========================
# 🔥 MODULE 11: COMPETITOR
# =========================
comp_dummy = pd.DataFrame({
    "competitor": ["A", "A", "B", "B"],
    "followers_start": [1000, 1000, 2000, 2000],
    "followers_end": [1200, 1200, 2500, 2500],
    "likes": [100, 150, 200, 180],
    "comments": [20, 30, 40, 35],
    "shares": [10, 15, 25, 20],
    "content_type": ["video", "image", "video", "text"]
})
comp_df = comp_metrics(comp_dummy)
strategy = analyze_strategy(comp_df).to_dict(orient="records")

# =========================
# 🔥 MODULE 12: POPULARITY
# =========================
df_pop = pop_features(df.head(2000))
model, score = pop_model(df_pop)
predictions = predict(model, df_pop.head(5))

# =========================
# 🌐 ROUTE
# =========================
@app.route("/")
def home():
    return render_template(
        "index.html",

        sentiments=sentiments,
        accuracy=accuracy,
        topic_sentiments=topic_sentiments,
        trends=trends,
        labels=labels,
        values=values,

        communities=communities,
        influencers_net=influencers_net,
        recommendations=recommendations,
        fake_acc=fake_acc,
        segments=segments,
        ad_data=ad_data,
        influencers_eigen=influencers_eigen,
        realtime_data=realtime_data,
        strategy=strategy,
        predictions=predictions,
        model_score=score
    )


if __name__ == "__main__":
    app.run(debug=True)
