from db.mongo_setup import get_collection

tweets_collection = get_collection("tweets")

def verify_sentiment_labels():
    """Fetch tweets from MongoDB and verify their sentiment labels."""
    tweets = tweets_collection.find()
    for tweet in tweets:
        print(f"Tweet: {tweet['text'][:50]}... | Sentiment: {tweet['sentiment']}")

if __name__ == "__main__":
    verify_sentiment_labels()
    print("Sentiment verification based on pre-labeled dataset complete.")
