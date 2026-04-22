import json
import os
import pandas as pd
import torch
import mlflow.pyfunc
from kafka import KafkaConsumer
import psycopg2
from transformers import AutoTokenizer

# 1. Configuration - Pointing to the specific model folder found in your 'ls -R'
MODEL_PATH = "/home/chaitanya/reddit-sentiment-analysis/mlruns/581710459020374166/models/m-b4b8dda99b6d488988a7bd355e6fb4fd/artifacts"
DB_CONFIG = {
    "host": "localhost",
    "database": "sentiment_db",
    "user": "mlops_user",
    "password": "mlops_password",
    "port": "5433"
}

# 2. Load Model and Tokenizer
print("📦 Loading DistilBERT Champion and Tokenizer...")
# We load the tokenizer specifically to handle raw text from Kafka
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = mlflow.pytorch.load_model(MODEL_PATH)
model.eval()

# 3. Setup Kafka Consumer
consumer = KafkaConsumer(
    'reddit_stream',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

def save_to_db(post_id, title, sentiment, subreddit):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        query = """
            INSERT INTO reddit_sentiment (post_id, title, sentiment, subreddit) 
            VALUES (%s, %s, %s, %s) 
            ON CONFLICT (post_id) DO NOTHING;
        """
        cur.execute(query, (post_id, title, sentiment, subreddit))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ DB Insert Error: {e}")

print("🎧 Listening to Kafka... (Press Ctrl+C to stop)")

for message in consumer:
    data = message.value
    text = data['title']
    
    # 4. PRE-PROCESSING: Tokenize text before passing to the PyTorch model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # 5. INFERENCE
    with torch.no_grad():
        outputs = model(**inputs)
        # For DistilBERT SST-2: 0 is Negative, 1 is Positive
        # Note: Your training data had 3 classes, so we use argmax
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    # Map back to human-readable sentiment
    # Adjust mapping if your training used 0=Neg, 1=Neu, 2=Pos
    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = mapping.get(prediction, "Unknown")
    
    print(f"[{sentiment}] → {text[:60]}...")
    
    # 6. SAVE TO POSTGRES
    save_to_db(data['post_id'], text, sentiment, data.get('subreddit', 'unknown'))
