from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import os
from urllib.request import Request, urlopen

import pandas as pd

BASE_URL = "https://api.sportsdata.io/v3/nba/scores/json/Injuries"
OUT_PATH = Path("data/injuries/nba_injuries_master.csv")


def _http_get_json(url: str, api_key: str):
    req = Request(url, headers={"Ocp-Apim-Subscription-Key": api_key, "User-Agent": "nba-props-model"})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _date_range(center: datetime, backfill_days: int, forward_days: int) -> list[str]:
    start = (center - timedelta(days=backfill_days)).date()
    end = (center + timedelta(days=forward_days)).date()
    dates = []
    d = start
    while d <= end:
        dates.append(d.isoformat())
        d += timedelta(days=1)
    return dates


def _normalize_status(raw: str) -> str:
    status = str(raw or "").strip().lower()
    mapping = {
        "out": "out",
        "doubtful": "doubtful",
        "questionable": "questionable",
        "probable": "probable",
        "active": "active",
    }
    return mapping.get(status, status or "active")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    api_key = os.getenv("SPORTSDATAIO_API_KEY", "")
    now = datetime.now(timezone.utc)
    dates = _date_range(now, backfill_days=14, forward_days=3)

    rows = []
    if api_key:
        try:
            payload = _http_get_json(BASE_URL, api_key)
        except Exception as exc:
            print(f"SportsDataIO injuries fetch failed: {exc}")
            payload = []
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                team = item.get("Team") or item.get("TeamAbbr") or item.get("TeamAbbreviation") or ""
                player = item.get("Name") or item.get("Player") or item.get("PlayerName") or ""
                status = _normalize_status(item.get("Status"))
                injury = item.get("Injury") or item.get("InjuryDescription") or ""
                updated = item.get("Updated") or item.get("LastUpdated")
                updated_ts = None
                if updated:
                    try:
                        updated_ts = pd.to_datetime(updated, errors="coerce").isoformat()
                    except Exception:
                        updated_ts = None
                for game_date in dates:
                    rows.append({
                        "game_date": game_date,
                        "team_abbr": team,
                        "player": player,
                        "status": status,
                        "injury": injury,
                        "updated_ts": updated_ts or now.isoformat(),
                        "source": "sportsdataio",
                    })

    df = pd.DataFrame(rows, columns=[
        "game_date", "team_abbr", "player", "status", "injury", "updated_ts", "source"
    ])
    if OUT_PATH.exists():
        try:
            existing = pd.read_csv(OUT_PATH)
        except Exception:
            existing = pd.DataFrame(columns=df.columns)
    else:
        existing = pd.DataFrame(columns=df.columns)
    combined = pd.concat([existing, df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_date", "team_abbr", "player", "status", "updated_ts"])
    combined.to_csv(OUT_PATH, index=False)
    if api_key:
        print("Saved SportsDataIO injuries to", OUT_PATH)
    else:
        print("SPORTSDATAIO_API_KEY missing; wrote empty injuries file to", OUT_PATH)


if __name__ == "__main__":
    main()
