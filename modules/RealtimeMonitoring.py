import requests
import time
import json

def fetch_data(keyword):
    url = f"https://api.datamuse.com/words?ml={keyword}"

    try:
        response = requests.get(url)
        data = response.json()

        # extract words as fake "posts"
        posts = [item["word"] for item in data[:5]]

        return posts

    except Exception as e:
        print("Error fetching data:", e)
        return []

def monitor_keyword(keyword, iterations=3, delay=5):
    results = []

    for i in range(iterations):
        print(f"\nFetching data for '{keyword}'...")

        posts = fetch_data(keyword)

        print("New Data:", posts)

        results.append({
            "iteration": i+1,
            "keyword": keyword,
            "data": posts
        })

        time.sleep(delay)  # wait before next fetch

    return results

if __name__ == "__main__":
    print("Running Real-Time Monitoring...")

    keyword = "technology"

    output = monitor_keyword(keyword)

    print("\nFINAL OUTPUT:\n", output)

    
