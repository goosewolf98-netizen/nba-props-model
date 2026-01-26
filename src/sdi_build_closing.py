from __future__ import annotations

from pathlib import Path
import pandas as pd

MASTER_PATH = Path("data/lines/sdi_props_master.csv")
CLOSING_PATH = Path("data/lines/sdi_props_closing.csv")


def main():
    CLOSING_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MASTER_PATH.exists():
        pd.DataFrame(columns=[
            "game_date", "game_id", "player", "player_id", "team_abbr", "opp_abbr",
            "stat", "line", "over_odds", "under_odds", "book", "snapshot_ts",
        ]).to_csv(CLOSING_PATH, index=False)
        print("Missing master props file; wrote empty closing file.")
        return

    try:
        df = pd.read_csv(MASTER_PATH)
    except Exception:
        pd.DataFrame(columns=[
            "game_date", "game_id", "player", "player_id", "team_abbr", "opp_abbr",
            "stat", "line", "over_odds", "under_odds", "book", "snapshot_ts",
        ]).to_csv(CLOSING_PATH, index=False)
        print("Failed to read master props file; wrote empty closing file.")
        return

    if df.empty:
        df.to_csv(CLOSING_PATH, index=False)
        print("Master props file empty; wrote empty closing file.")
        return

    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce")
    df = df.sort_values("snapshot_ts")
    closing = df.dropna(subset=["snapshot_ts"]).groupby(
        ["game_id", "player", "stat", "book"], as_index=False
    ).tail(1)
    closing.to_csv(CLOSING_PATH, index=False)
    print("Saved closing props file:", CLOSING_PATH)


if __name__ == "__main__":
    main()
