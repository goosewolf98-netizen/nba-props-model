import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from db.manager import DBManager
from collectors.reddit import RedditScraper
from db.ingest_lines import ingest_lines
# We can import the download function directly if we refactor download_nba_api_data to be importable
# It currently has a `if __name__ == "__main__": download_data()` so importing is safe.
from download_nba_api_data import download_data

def update_all():
    print("=== Starting Database Update ===")

    # 1. NBA API Data
    print("\n--- 1. Updating NBA API Data ---")
    try:
        download_data() # This now handles DB upsert internally
    except Exception as e:
        print(f"Error downloading NBA API data: {e}")

    # 2. Prop Lines
    print("\n--- 2. Ingesting Prop Lines ---")
    ingest_lines()

    # 3. Reddit Sentiment
    print("\n--- 3. Updating Reddit Sentiment ---")
    scraper = RedditScraper()
    db = DBManager()

    subreddits = ['nba', 'sportsbook', 'dfs']
    for sub in subreddits:
        print(f"Scraping r/{sub}...")
        posts = scraper.fetch_subreddit(sub, limit=25)
        if posts:
            db.insert_sentiment(posts)
        else:
            print(f"No posts found for r/{sub} (or access denied).")

    print("\n=== Database Update Complete ===")

if __name__ == "__main__":
    update_all()
