from __future__ import annotations

from pathlib import Path
import pandas as pd


def _normalize_player(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _normalize_market(market: str) -> str:
    market = str(market).strip().lower()
    mapping = {
        "points": "pts",
        "point": "pts",
        "pts": "pts",
        "rebounds": "reb",
        "rebound": "reb",
        "reb": "reb",
        "assists": "ast",
        "assist": "ast",
        "ast": "ast",
    }
    return mapping.get(market, market)


def load_prop_lines(path: Path | str = "data/lines/props_lines.csv") -> pd.DataFrame:
    path = Path(path)
    columns = ["game_date", "player", "stat", "line", "over_odds", "under_odds", "book"]
    if not path.exists():
        print(f"Prop lines file missing: {path}")
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path)
    except Exception:
        print(f"Prop lines failed to read: {path}")
        return pd.DataFrame(columns=columns)

    required = ["game_date", "player", "stat", "line"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        print(f"Prop lines missing required columns: {missing_required}")
        return pd.DataFrame(columns=columns)

    for col in ["over_odds", "under_odds", "book"]:
        if col not in df.columns:
            df[col] = pd.NA

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype(str)
    df["player"] = df["player"].astype(str)
    df["player_norm"] = df["player"].apply(_normalize_player)
    df["stat"] = df["stat"].astype(str)
    df["stat_norm"] = df["stat"].apply(_normalize_market)
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["over_odds"] = pd.to_numeric(df["over_odds"], errors="coerce").fillna(-110)
    df["under_odds"] = pd.to_numeric(df["under_odds"], errors="coerce").fillna(-110)
    df["book"] = df["book"].astype(str)
    return df
