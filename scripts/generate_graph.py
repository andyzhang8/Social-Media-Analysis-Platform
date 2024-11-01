import networkx as nx
from db.mongo_setup import get_collection

def export_network_to_gexf(graph, filename="user_network.gexf"):
    """Exports the NetworkX graph to a GEXF file for visualization."""
    nx.write_gexf(graph, filename)
    print(f"Network exported to {filename}.")

def main():
    # Build
    tweets_collection = get_collection("tweets")
    G = nx.Graph()
    
    # Re-create user network
    for tweet in tweets_collection.find({}, {"user_name": 1, "text": 1}):
        user = tweet["user_name"]
        mentions = [word[1:] for word in tweet["text"].split() if word.startswith("@")]

        if not G.has_node(user):
            G.add_node(user)

        for mention in mentions:
            if not G.has_node(mention):
                G.add_node(mention)
            if G.has_edge(user, mention):
                G[user][mention]["weight"] += 1
            else:
                G.add_edge(user, mention, weight=1)

    pagerank = nx.pagerank(G)
    nx.set_node_attributes(G, pagerank, "pagerank")

    # Export to GEXF format
    export_network_to_gexf(G)

if __name__ == "__main__":
    main()
