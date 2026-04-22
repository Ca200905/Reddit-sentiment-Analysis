import requests
import json
import time
import psycopg2
from kafka import KafkaProducer

# 1. Setup Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    api_version=(3, 7, 0),
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

headers = {"User-Agent": "python:mlops.project:v1.0 (by /u/chaitanya)"}
subreddits = ["technology", "worldnews", "politics", "gaming", "news", "science"]

def is_already_in_db(post_id):
    try:
        conn = psycopg2.connect(
            host="localhost", database="sentiment_db", 
            user="mlops_user", password="mlops_password", port="5433"
        )
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM reddit_sentiment WHERE post_id = %s", (post_id,))
        exists = cur.fetchone() is not None
        cur.close()
        conn.close()
        return exists
    except:
        return False

def fetch_and_stream(subreddit):
    url = f"https://api.reddit.com/r/{subreddit}/new.json?limit=25"
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            posts = res.json()["data"]["children"]
            new_count = 0
            for p in posts:
                data = p["data"]
                post_id = data["id"]
                if not is_already_in_db(post_id):
                    message = {
                        "post_id": post_id,
                        "title": data["title"],
                        "subreddit": subreddit,
                        "created_utc": data["created_utc"]
                    }
                    producer.send('reddit_stream', message)
                    new_count += 1
            print(f"🚀 r/{subreddit}: Sent {new_count} posts.")
        else:
            print(f"⚠️ Reddit API Error {res.status_code}")
    except Exception as e:
        print(f"❌ Producer Error: {e}")

if __name__ == "__main__":
    print("🛰️ Starting Real-Time Ingestion...")
    while True:
        for sub in subreddits:
            fetch_and_stream(sub)
            time.sleep(2)
        print("\n⏳ Sleeping 60s...\n")
        time.sleep(60)
