import pymongo
import networkx as nx
from db.mongo_setup import get_collection

# Initialize MongoDB collections
tweets_collection = get_collection("tweets")
network_collection = get_collection("user_network")

def build_user_network():
    """Builds and returns a NetworkX graph based on user mentions in tweets."""
    G = nx.Graph()
    
    tweets = tweets_collection.find({}, {"user_name": 1, "mentions": 1})

    for tweet in tweets:
        user = tweet['user_name']
        mentions = tweet.get("mentions", [])
        if not G.has_node(user):
            G.add_node(user)

        # Add mentions as nodes and create edges between user and mentions
        for mention in mentions:
            if not G.has_node(mention):
                G.add_node(mention)
            if G.has_edge(user, mention):
                G[user][mention]['weight'] += 1
            else:
                G.add_edge(user, mention, weight=1)

    return G

def calculate_centrality(G):
    """Calculates and sets centrality scores for nodes in the graph."""
    centrality_scores = nx.degree_centrality(G)
    nx.set_node_attributes(G, centrality_scores, "centrality")
    return centrality_scores

def store_network_data(G):
    """Stores network data in MongoDB."""
    network_data = []
    for node in G.nodes(data=True):
        user = node[0]
        attributes = node[1]
        connections = list(G.neighbors(user))

        network_data.append({
            "user_name": user,
            "centrality": attributes.get("centrality", 0),
            "connections": connections
        })

    network_collection.delete_many({})
    network_collection.insert_many(network_data)
    print(f"Stored {len(network_data)} user nodes in the network collection.")

def analyze_network():
    G = build_user_network()
    calculate_centrality(G)
    store_network_data(G)
    print("Network analysis complete and stored in MongoDB.")

if __name__ == "__main__":
    analyze_network()
