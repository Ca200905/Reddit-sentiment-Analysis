import json
import time
from kafka import KafkaConsumer
from transformers import pipeline
from prometheus_client import start_http_server, Counter, Summary, Gauge
import psycopg2

# 1. ADVANCED PROMETHEUS METRICS
# Counter: Total count of predictions (Good for throughput)
SENTIMENT_COUNT = Counter('sentiment_predictions_total', 'Total predictions', ['label'])

# Summary: Tracks how long inference takes (Good for performance/latency)
INFERENCE_LATENCY = Summary('inference_latency_seconds', 'Time taken for model inference')

# Gauge: Tracks the length of incoming text (Good for detecting Data Drift in input size)
INPUT_TEXT_LENGTH = Gauge('input_text_char_length', 'Length of the input text in characters')

# Start Prometheus exporter
start_http_server(8000, addr='0.0.0.0')
print("✅ Prometheus metrics exporter started on port 8000")

# 2. SETUP MODEL (RoBERTa)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME, device=-1)

# 3. DATABASE CONFIG
DB_CONFIG = {
    "host": "localhost",
    "database": "sentiment_db",
    "user": "mlops_user",
    "password": "mlops_password",
    "port": "5432"
}

# 4. KAFKA SETUP
consumer = KafkaConsumer(
    'reddit_stream',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

def save_to_db(post_id, title, sentiment, subreddit):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO reddit_sentiment (post_id, title, sentiment, subreddit) VALUES (%s, %s, %s, %s) ON CONFLICT (post_id) DO NOTHING;",
            (post_id, title, sentiment, subreddit)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ DB Error: {e}")

print("🎧 Worker Live & Monitoring Kafka...")

# 5. INFERENCE LOOP WITH MONITORING
for message in consumer:
    data = message.value
    text = data['title']
    
    # Track Input Drift: Measure text length
    INPUT_TEXT_LENGTH.set(len(text))

    # Track Latency: Measure how long the model takes
    start_time = time.time()
    result = sentiment_task(text[:512])[0]
    latency = time.time() - start_time
    INFERENCE_LATENCY.observe(latency)

    label = result['label'].capitalize()
    
    # Increment Sentiment Counter
    SENTIMENT_COUNT.labels(label=label).inc()

    print(f"[{label}] ({latency:.3f}s) → {text[:70]}...")
    
    save_to_db(data['post_id'], text, label, data.get('subreddit', 'unknown'))
