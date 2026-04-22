import json
import torch
import psycopg2
from kafka import KafkaConsumer
from transformers import pipeline
from prometheus_client import start_http_server, Counter

# 1. PROMETHEUS METRICS SETUP
# This must be defined at the top level
SENTIMENT_COUNT = Counter('sentiment_predictions_total', 'Total predictions by label', ['label'])

# Start Prometheus exporter on port 8000
# This allows Prometheus to scrape data from http://localhost:8000
start_http_server(8000)
print("📈 Prometheus metrics exporter started on port 8000")

# 2. SETUP MODEL (RoBERTa)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
print(f"📦 Loading Production Transformer: {MODEL_NAME}...")
sentiment_task = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME, device=-1)

# 3. CONFIG
DB_CONFIG = {"host": "localhost", "database": "sentiment_db", "user": "mlops_user", "password": "mlops_password", "port": "5433"}

# 4. KAFKA SETUP
consumer = KafkaConsumer('reddit_stream', bootstrap_servers=['localhost:9092'], value_deserializer=lambda x: json.loads(x.decode('utf-8')))

def save_to_db(post_id, title, sentiment, subreddit):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("INSERT INTO reddit_sentiment (post_id, title, sentiment, subreddit) VALUES (%s, %s, %s, %s) ON CONFLICT (post_id) DO NOTHING;", (post_id, title, sentiment, subreddit))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e: print(f"⚠️ DB Error: {e}")

print("🎧 Worker Live & Monitoring Kafka...")

# 5. INFERENCE LOOP
for message in consumer:
    data = message.value
    text = data['title']
    
    # Accurate Inference
    result = sentiment_task(text[:512])[0]
    
    # CRITICAL: Define the label first
    label = result['label'].capitalize() 
    
    # NOW increment the Prometheus counter
    SENTIMENT_COUNT.labels(label=label).inc()
    
    print(f"[{label}] → {text[:70]}...")
    save_to_db(data['post_id'], text, label, data.get('subreddit', 'unknown'))
