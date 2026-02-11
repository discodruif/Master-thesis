import requests
import json
import os
from datetime import datetime

# Load NYT API Key
with open("../config/nyt-api-key.txt") as f:
    API_KEY = f.read().strip()

# Base URL
BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

def fetch_articles(query, begin_date, end_date):
    """Fetch articles from NYT API for a given query and date range"""
    params = {
        'q': query,
        'begin_date': begin_date,
        'end_date': end_date,
        'api-key': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

# Define search queries and date range
queries = [
    "Machine learning",
    "Algorithmic trading",
    "Options pricing",
    "Derivatives markets",
    "Quantitative finance"
]
begin_date = "20210101"
end_date = "20251231"

# Create data directory if it doesn't exist
os.makedirs("../data/nyt", exist_ok=True)

# Fetch articles for each query
for query in queries:
    print(f"Fetching articles for query: {query}")
    try:
        articles = fetch_articles(query, begin_date, end_date)
        with open(f"../data/nyt/{query.replace(' ', '_').lower()}_articles.json", "w") as f:
            json.dump(articles, f, indent=4)
        print(f"Saved articles for {query}")
    except Exception as e:
        print(f"Failed to fetch articles for {query}: {str(e)}")