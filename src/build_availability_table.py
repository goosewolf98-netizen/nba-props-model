from __future__ import annotations

from pathlib import Path
import pandas as pd


MASTER_PATH = Path("data/injuries/nba_injuries_master.csv")
OUT_PATH = Path("data/injuries/availability_by_game.csv")


def _flag(status: str, label: str) -> int:
    return int(str(status).lower() == label)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MASTER_PATH.exists():
        pd.DataFrame(columns=[
            "game_date", "team_abbr", "player",
            "is_out", "is_doubt", "is_q", "is_prob", "is_active",
        ]).to_csv(OUT_PATH, index=False)
        print("Missing injuries master; wrote empty availability table.")
        return

    try:
        df = pd.read_csv(MASTER_PATH)
    except Exception:
        pd.DataFrame(columns=[
            "game_date", "team_abbr", "player",
            "is_out", "is_doubt", "is_q", "is_prob", "is_active",
        ]).to_csv(OUT_PATH, index=False)
        print("Failed to read injuries master; wrote empty availability table.")
        return

    if df.empty:
        df.reindex(columns=[
            "game_date", "team_abbr", "player",
            "is_out", "is_doubt", "is_q", "is_prob", "is_active",
        ]).to_csv(OUT_PATH, index=False)
        print("Injuries master empty; wrote empty availability table.")
        return

    for col in ["game_date", "team_abbr", "player", "status"]:
        if col not in df.columns:
            df[col] = ""

    df["status"] = df["status"].astype(str).str.lower()
    df["is_out"] = df["status"].apply(lambda s: _flag(s, "out"))
    df["is_doubt"] = df["status"].apply(lambda s: _flag(s, "doubtful"))
    df["is_q"] = df["status"].apply(lambda s: _flag(s, "questionable"))
    df["is_prob"] = df["status"].apply(lambda s: _flag(s, "probable"))
    df["is_active"] = df["status"].apply(lambda s: _flag(s, "active"))

    out = df[[
        "game_date", "team_abbr", "player",
        "is_out", "is_doubt", "is_q", "is_prob", "is_active",
    ]].drop_duplicates()
    out.to_csv(OUT_PATH, index=False)
    print("Saved availability table to", OUT_PATH)


if __name__ == "__main__":
    main()
