import requests
import time

def safe_request(url, headers):
    """Helper function to handle network timeouts safely."""
    try:
        response = requests.get(url, headers=headers, timeout=10)
        return response
    except Exception as e:
        print(f"📡 Network Error: {e}")
        return None

def fetch_comments(post_id):
    url = f"https://api.reddit.com/comments/{post_id}?limit=50"
    headers = {
        "User-Agent": "python:bigdata.project:v1.0 (by /u/temporary_user)"
    }

    try:
        # Fixed: passing headers and using the now-defined safe_request
        response = safe_request(url, headers)

        if response is None:
            return []

        print(f"💬 Comments {post_id} → Status:", response.status_code)

        if response.status_code != 200:
            return []

        data = response.json()

    except Exception as e:
        print(f"⚠️ Error fetching comments for {post_id}: {e}")
        time.sleep(2)
        return []

    comments = []
    try:
        # data[1] contains the actual comments tree in Reddit's JSON
        for comment in data[1]["data"]["children"]:
            if comment["kind"] != "t1":
                continue

            d = comment["data"]
            body = d.get("body", "")

            # skip low quality or bot comments
            if len(body.split()) < 5:
                continue

            comments.append({
                "comment_id": d.get("id"),
                "comment": body,
                "comment_score": d.get("score"),
                "author": d.get("author"),
                "parent_id": d.get("parent_id"),
                "root_id": d.get("link_id"),
                "comment_time": d.get("created_utc")
            })

    except Exception as e:
        print("⚠️ Parsing error:", e)

    return comments
