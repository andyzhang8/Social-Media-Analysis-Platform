from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, request, jsonify
import torch
import pymongo

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "social_media_analysis"
COLLECTION_NAME = "tweets"

app = Flask(__name__)

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Constants
MODEL_PATH = "./sentiment_model"
MAX_LENGTH = 128

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # Set model to evaluation mode

@app.route("/network", methods=["GET"])
def get_network_data():
    network_data = list(network_collection.find({}, {"_id": 0, "user_name": 1, "centrality": 1, "retweet_count": 1, "connections": 1}))
    return jsonify(network_data)

def predict_sentiment(text):
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # Run model inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Map predicted class to sentiment label
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return label_map[predicted_class]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    sentiment = predict_sentiment(text)
    return jsonify({"text": text, "sentiment": sentiment})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

