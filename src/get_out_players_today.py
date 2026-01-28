from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd


def _normalize_date(value: str) -> str:
    return str(pd.to_datetime(value, errors="coerce").date())


def _load_availability(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Target game date (YYYY-MM-DD)")
    args = parser.parse_args()

    target_date = args.date or str(date.today())

    availability_path = Path("data/injuries/availability_by_game.csv")
    injuries_latest_path = Path("artifacts/injuries_latest.csv")

    out_map: dict[str, list[str]] = {}
    availability = _load_availability(availability_path)
    if not availability.empty:
        for col in ["game_date", "team_abbr", "player", "is_out"]:
            if col not in availability.columns:
                availability[col] = "" if col != "is_out" else 0
        availability["game_date"] = availability["game_date"].apply(_normalize_date)
        availability["is_out"] = pd.to_numeric(availability["is_out"], errors="coerce").fillna(0).astype(int)
        subset = availability[(availability["game_date"] == target_date) & (availability["is_out"] == 1)]
        for team, rows in subset.groupby("team_abbr"):
            out_map[str(team)] = sorted(rows["player"].dropna().astype(str).unique().tolist())
    else:
        injuries = _load_availability(injuries_latest_path)
        if not injuries.empty:
            for col in ["team_abbr", "player", "status"]:
                if col not in injuries.columns:
                    injuries[col] = ""
            injuries["status"] = injuries["status"].astype(str).str.upper()
            subset = injuries[injuries["status"] == "OUT"]
            for team, rows in subset.groupby("team_abbr"):
                out_map[str(team)] = sorted(rows["player"].dropna().astype(str).unique().tolist())

    out_path = Path("artifacts/out_players_today.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_map, indent=2))
    print("Saved", out_path)


if __name__ == "__main__":
    main()
