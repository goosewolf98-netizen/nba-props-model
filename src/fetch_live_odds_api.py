import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

API_KEY = os.getenv("THE_ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
OUT_PATH = Path("data/lines/props_lines.csv")

def fetch_live_props():
    if not API_KEY:
        print("THE_ODDS_API_KEY missing. Skipping live odds fetch.")
        return

    print("Fetching live NBA events...")
    events_url = f"{BASE_URL}/?apiKey={API_KEY}"
    events = requests.get(events_url).json()

    all_lines = []

    # We'll focus on the first few games to save rate limits
    for event in events[:5]:
        event_id = event['id']
        print(f"Fetching props for {event['home_team']} vs {event['away_team']}...")

        # Props endpoint
        # Market types: player_points, player_rebounds, player_assists, etc.
        markets = "player_points,player_rebounds,player_assists,player_threes,player_blocks,player_steals"
        props_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/bookmakers?apiKey={API_KEY}&regions=us&markets={markets}"

        props_data = requests.get(props_url).json()

        for bookmaker in props_data.get('bookmakers', []):
            book_name = bookmaker['title']
            for market in bookmaker.get('markets', []):
                stat_map = {
                    "player_points": "pts",
                    "player_rebounds": "reb",
                    "player_assists": "ast",
                    "player_threes": "tpm",
                    "player_blocks": "blk",
                    "player_steals": "stl"
                }
                stat = stat_map.get(market['key'])
                if not stat: continue

                for outcome in market.get('outcomes', []):
                    player = outcome['description']
                    line = outcome.get('point')
                    price = outcome.get('price')
                    side = outcome.get('name').lower() # 'over' or 'under'

                    all_lines.append({
                        "game_date": datetime.now().strftime("%Y-%m-%d"),
                        "player": player,
                        "stat": stat,
                        "line": line,
                        "odds": price,
                        "side": side,
                        "book": book_name
                    })

    if all_lines:
        df = pd.DataFrame(all_lines)
        # Pivot to match the required format: game_date, player, stat, line, over_odds, under_odds, book
        pivoted = df.pivot_table(
            index=["game_date", "player", "stat", "line", "book"],
            columns="side",
            values="odds"
        ).reset_index()

        pivoted = pivoted.rename(columns={"over": "over_odds", "under": "under_odds"})

        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        if OUT_PATH.exists():
            existing = pd.read_csv(OUT_PATH)
            combined = pd.concat([existing, pivoted]).drop_duplicates(subset=["game_date", "player", "stat", "book"])
            combined.to_csv(OUT_PATH, index=False)
        else:
            pivoted.to_csv(OUT_PATH, index=False)
        print(f"Saved {len(pivoted)} real prop lines to {OUT_PATH}")

if __name__ == "__main__":
    fetch_live_props()
