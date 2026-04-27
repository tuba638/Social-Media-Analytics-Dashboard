import networkx as nx
import json

# -------------------------------
# 🔥 BUILD GRAPH
# -------------------------------
def build_graph(posts):
    G = nx.DiGraph()

    for post in posts:
        user = post["user"]
        mentions = post.get("mentions", [])

        for m in mentions:
            G.add_edge(user, m)

    return G


# -------------------------------
# 🔥 INFLUENCER DETECTION
# -------------------------------
def detect_influencers(G, top_n=5):
    # Eigenvector Centrality
    centrality = nx.eigenvector_centrality(G, max_iter=1000)

    sorted_users = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    return sorted_users[:top_n]


# -------------------------------
# 🔥 MAIN RUN
# -------------------------------
if __name__ == "__main__":
    print("Running Influencer Detection...")

    # Sample data
    posts = [
        {"user": "A", "mentions": ["B", "C"]},
        {"user": "B", "mentions": ["C", "D"]},
        {"user": "C", "mentions": ["D"]},
        {"user": "D", "mentions": ["A"]},
        {"user": "E", "mentions": ["A", "B"]}
    ]

    G = build_graph(posts)

    influencers = detect_influencers(G)

    output = {
        "top_influencers": influencers
    }

    print("OUTPUT:\n", output)

   