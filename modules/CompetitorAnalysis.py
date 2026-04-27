import pandas as pd
import json

# -------------------------------
# 🔥 CALCULATE METRICS
# -------------------------------
def calculate_metrics(df):
    # Growth rate
    df["growth_rate"] = ((df["followers_end"] - df["followers_start"]) / df["followers_start"]) * 100

    # Engagement rate
    df["engagement_rate"] = ((df["likes"] + df["comments"] + df["shares"]) / df["followers_end"]) * 100

    return df


# -------------------------------
# 🔥 STRATEGY ANALYSIS
# -------------------------------
def analyze_strategy(df):
    # find most frequent content type per competitor
    strategy = df.groupby("competitor")["content_type"] \
                 .agg(lambda x: x.value_counts().idxmax()) \
                 .reset_index()

    strategy.columns = ["competitor", "top_content_type"]

    return strategy


# -------------------------------
# 🔥 MAIN RUN
# -------------------------------
if __name__ == "__main__":
    print("Running Competitor Analysis...")

    # Sample data
    data = {
        "competitor": ["A", "A", "B", "B", "C", "C"],
        "followers_start": [1000, 1000, 2000, 2000, 1500, 1500],
        "followers_end": [1200, 1200, 2500, 2500, 1800, 1800],
        "likes": [100, 150, 200, 180, 130, 160],
        "comments": [20, 30, 40, 35, 25, 30],
        "shares": [10, 15, 25, 20, 12, 18],
        "content_type": ["video", "image", "video", "text", "image", "video"]
    }

    df = pd.DataFrame(data)

    df = calculate_metrics(df)
    strategy_df = analyze_strategy(df)

    # Merge results
    final_df = pd.merge(df, strategy_df, on="competitor")

    output = final_df.to_dict(orient="records")

    print("OUTPUT:\n", output)

    