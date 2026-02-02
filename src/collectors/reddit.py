import requests
import time
from datetime import datetime

class RedditScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def fetch_subreddit(self, subreddit, limit=25, mock=False):
        if mock:
            # Return fake data for testing
            return [{
                'id': f't3_fake_{subreddit}',
                'subreddit': subreddit,
                'title': f'Test Post in {subreddit}',
                'text': 'This is a test post body.',
                'score': 100,
                'num_comments': 10,
                'url': f'https://reddit.com/r/{subreddit}/fake',
                'created_utc': time.time()
            }]

        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
        try:
            print(f"Fetching r/{subreddit}...")
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                print(f"Error fetching {subreddit}: {response.status_code} (Reddit often blocks scraping without API auth)")
                return []

            data = response.json()
            posts = []
            for child in data.get('data', {}).get('children', []):
                post_data = child.get('data', {})
                posts.append({
                    'id': post_data.get('name'), # t3_... ID
                    'subreddit': subreddit,
                    'title': post_data.get('title'),
                    'text': post_data.get('selftext'),
                    'score': post_data.get('score'),
                    'num_comments': post_data.get('num_comments'),
                    'url': post_data.get('url'),
                    'created_utc': post_data.get('created_utc')
                })
            return posts
        except Exception as e:
            print(f"Exception fetching {subreddit}: {e}")
            return []

if __name__ == "__main__":
    scraper = RedditScraper()
    nba_posts = scraper.fetch_subreddit("nba", limit=5)
    print(f"Fetched {len(nba_posts)} posts from r/nba")
    if nba_posts:
        print("Sample post:", nba_posts[0]['title'])

    sb_posts = scraper.fetch_subreddit("sportsbook", limit=5)
    print(f"Fetched {len(sb_posts)} posts from r/sportsbook")
