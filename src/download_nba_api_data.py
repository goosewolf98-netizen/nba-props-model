from nba_api.stats.endpoints import leaguegamelog
import pandas as pd
from pathlib import Path
import time

def download_data():
    RAW_DIR = Path("data/raw")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading 2024-25 Player Boxscores...")
    try:
        log_p = leaguegamelog.LeagueGameLog(season='2024-25', player_or_team_abbreviation='P')
        df_p = log_p.get_data_frames()[0]
        df_p.to_csv(RAW_DIR / "nba_player_box.csv", index=False)
        print(f"Saved {len(df_p)} player boxscores.")
    except Exception as e:
        print(f"Failed to download player boxscores: {e}")

    time.sleep(1) # respect rate limits

    print("Downloading 2024-25 Team Boxscores...")
    try:
        log_t = leaguegamelog.LeagueGameLog(season='2024-25', player_or_team_abbreviation='T')
        df_t = log_t.get_data_frames()[0]
        df_t.to_csv(RAW_DIR / "nba_team_box.csv", index=False)
        print(f"Saved {len(df_t)} team boxscores.")
    except Exception as e:
        print(f"Failed to download team boxscores: {e}")

if __name__ == "__main__":
    download_data()
