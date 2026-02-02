from __future__ import annotations

from pathlib import Path
import json
import os
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen, Request

import pandas as pd

BASE_URL = "https://api.sportsdata.io/v3/nba/odds/json"
SNAPSHOT_DIR = Path("data/lines/sdi_props_snapshots")
MASTER_PATH = Path("data/lines/sdi_props_master.csv")

HEADERS = [
    "game_date",
    "player",
    "stat",
    "line",
    "over_odds",
    "under_odds",
    "book",
    "snapshot_ts",
]


def _http_get_json(url: str, api_key: str) -> object:
    req = Request(url, headers={"Ocp-Apim-Subscription-Key": api_key, "User-Agent": "nba-props-model"})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _normalize_stat(stat: str) -> str:
    stat = str(stat).strip().lower()
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
        "pra": "pra",
    }
    return mapping.get(stat, stat)


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except Exception:
        return None


def _extract_rows(payload: object, snapshot_ts: str) -> list[dict]:
    rows = []
    if not isinstance(payload, list):
        return rows
    for item in payload:
        if not isinstance(item, dict):
            continue
        game_date = str(item.get("Date") or item.get("GameDate") or "")[:10]
        markets = item.get("BettingMarkets") or item.get("Markets") or item.get("PlayerProps") or []
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict):
                continue
            player = market.get("PlayerName") or market.get("Player") or item.get("PlayerName")
            stat = market.get("BetName") or market.get("Market") or market.get("Stat") or ""
            stat = _normalize_stat(stat)
            if stat not in {"pts", "reb", "ast", "pra", "stl", "blk", "tpm"}:
                continue
            line = _safe_float(market.get("Handicap") or market.get("Line") or market.get("Total"))
            over_odds = _safe_float(market.get("OverOdds") or market.get("Over") or market.get("OverPayout"))
            under_odds = _safe_float(market.get("UnderOdds") or market.get("Under") or market.get("UnderPayout"))
            book = market.get("Sportsbook") or market.get("Book") or market.get("Provider") or ""
            rows.append({
                "game_date": game_date,
                "player": player,
                "stat": stat,
                "line": line,
                "over_odds": over_odds,
                "under_odds": under_odds,
                "book": book,
                "snapshot_ts": snapshot_ts,
            })
    return rows


def fetch_props_snapshot(date_str: str, api_key: str) -> pd.DataFrame:
    endpoints = [
        f"{BASE_URL}/PlayerPropsByDate/{date_str}",
        f"{BASE_URL}/PlayerProps/{date_str}",
    ]
    for url in endpoints:
        try:
            payload = _http_get_json(url, api_key)
            rows = _extract_rows(payload, snapshot_ts=datetime.now(timezone.utc).isoformat())
            return pd.DataFrame(rows)
        except Exception as exc:
            print(f"SportsDataIO fetch failed for {url}: {exc}")
            continue
    return pd.DataFrame(columns=HEADERS)

def _date_range(days: int) -> list[str]:
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    dates = []
    d = start
    while d <= end:
        dates.append(d.isoformat())
        d += timedelta(days=1)
    return dates

def main():
    api_key = os.getenv("SPORTSDATAIO_API_KEY", "")
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    lookback_days = 30
    if not api_key:
        df = pd.DataFrame(columns=HEADERS)
        snapshot_path = SNAPSHOT_DIR / f"{datetime.utcnow().date().isoformat()}.csv"
        df.to_csv(snapshot_path, index=False)
        df.to_csv(MASTER_PATH, index=False)
        print("SPORTSDATAIO_API_KEY missing; wrote empty snapshots.")
        return

    snapshots = []
    for date_str in _date_range(lookback_days):
        snapshot = fetch_props_snapshot(date_str, api_key)
        if snapshot.empty:
            snapshot = pd.DataFrame(columns=HEADERS)
        snapshot_path = SNAPSHOT_DIR / f"{date_str}.csv"
        snapshot.to_csv(snapshot_path, index=False)
        snapshots.append(snapshot)

    if MASTER_PATH.exists():
        master = pd.read_csv(MASTER_PATH)
    else:
        master = pd.DataFrame(columns=HEADERS)
    combined = pd.concat([master] + snapshots, ignore_index=True)
    combined = combined.reindex(columns=HEADERS)
    combined = combined.drop_duplicates(subset=["game_date", "player", "stat", "book", "snapshot_ts"])
    combined.to_csv(MASTER_PATH, index=False)
    print("Saved SportsDataIO props snapshots to", SNAPSHOT_DIR, "and master to", MASTER_PATH)


if __name__ == "__main__":
    main()
