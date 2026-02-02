from nba_api.stats.endpoints import leaguegamelog
import pandas as pd
from pathlib import Path
import time
import sys
import os

# Add src to path to import db manager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.db.manager import DBManager

def download_data():
    RAW_DIR = Path("data/raw")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    seasons = ['2024-25', '2025-26']

    all_p = []
    all_t = []

    for season in seasons:
        print(f"Downloading {season} Player Boxscores...")
        try:
            log_p = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='P')
            df_p = log_p.get_data_frames()[0]
            all_p.append(df_p)
            print(f"Fetched {len(df_p)} player boxscores for {season}.")
        except Exception as e:
            print(f"Failed to download player boxscores for {season}: {e}")

        time.sleep(1) # respect rate limits

        print(f"Downloading {season} Team Boxscores...")
        try:
            log_t = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='T')
            df_t = log_t.get_data_frames()[0]
            all_t.append(df_t)
            print(f"Fetched {len(df_t)} team boxscores for {season}.")
        except Exception as e:
            print(f"Failed to download team boxscores for {season}: {e}")

        time.sleep(1)

    if all_p:
        df_all_p = pd.concat(all_p, ignore_index=True)
        df_all_p.to_csv(RAW_DIR / "nba_player_box.csv", index=False)
        print(f"Saved total {len(df_all_p)} player boxscores.")

        # Upsert to DB
        print("Upserting player boxscores to database...")
        try:
            db = DBManager()
            db.upsert_boxscores(df_all_p)
        except Exception as e:
            print(f"Error updating database: {e}")

    if all_t:
        df_all_t = pd.concat(all_t, ignore_index=True)
        df_all_t.to_csv(RAW_DIR / "nba_team_box.csv", index=False)
        print(f"Saved total {len(df_all_t)} team boxscores.")

if __name__ == "__main__":
    download_data()
