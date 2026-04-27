import networkx as nx
from collections import defaultdict


def build_graph(posts):
    G = nx.Graph()

    for post in posts:
        user = post.get("user", "unknown")
        mentions = post.get("mentions", [])

        for m in mentions:
            G.add_edge(user, m)

    return G



def get_influencers(G, top_n=5):
    centrality = nx.degree_centrality(G)
    sorted_users = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    return sorted_users[:top_n]



def detect_communities(G):
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))

    result = []
    for i, community in enumerate(communities):
        result.append({
            "community_id": i,
            "members": list(community)
        })

    return result



def get_connectors(G):
    connectors = list(nx.articulation_points(G))
    return connectors



def analyze_network(posts):
    G = build_graph(posts)

    influencers = get_influencers(G)
    communities = detect_communities(G)
    connectors = get_connectors(G)

    return {
        "influencers": influencers,
        "communities": communities,
        "connectors": connectors
    }


if __name__ == "__main__":
    print("Running Network Analysis...")

    posts = [
        {"user": "A", "mentions": ["B", "C"]},
        {"user": "B", "mentions": ["C"]},
        {"user": "C", "mentions": ["A"]}
    ]

    result = analyze_network(posts)
    print("OUTPUT:\n", result)