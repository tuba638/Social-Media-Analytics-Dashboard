import pandas as pd
import json

# -------------------------------
# 🔥 METRICS CALCULATION
# -------------------------------
def calculate_metrics(df):
    df["CTR"] = (df["clicks"] / df["impressions"]) * 100
    df["conversion_rate"] = (df["conversions"] / df["clicks"]) * 100
    df["ROI"] = ((df["revenue"] - df["cost"]) / df["cost"]) * 100

    return df


# -------------------------------
# 🔥 MAIN RUN
# -------------------------------
if __name__ == "__main__":
    print("Running Ad Campaign Optimization...")

    # Sample data (agar real nahi hai)
    data = {
        "campaign": ["A", "B", "C", "D"],
        "impressions": [1000, 2000, 1500, 1800],
        "clicks": [100, 250, 120, 200],
        "conversions": [10, 40, 15, 25],
        "cost": [500, 800, 600, 700],
        "revenue": [1500, 2000, 1200, 1800]
    }

    df = pd.DataFrame(data)

    df = calculate_metrics(df)

    output = df.to_dict(orient="records")

    print("OUTPUT:\n", output)
