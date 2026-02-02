from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
from datetime import datetime, timedelta, timezone

import pandas as pd

DATA_DIR = Path("data/derived")

PLAYER_SCHEMA = [
    "player",
    "team_abbr",
    "player_on_off_net",
    "player_on_off_pace",
    "player_minutes_on_recent",
    "minutes_on",
    "minutes_off",
    "cache_timestamp",
]

TEAM_SCHEMA = [
    "team_abbr",
    "team_def_net_recent",
    "cache_timestamp",
]


def _cache_paths():
    return (
        DATA_DIR / "player_onoff_cache.parquet",
        DATA_DIR / "team_onoff_cache.parquet",
    )


def _write_empty_cache(cache_timestamp: str) -> None:
    player_path, team_path = _cache_paths()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=PLAYER_SCHEMA).assign(cache_timestamp=cache_timestamp).to_parquet(player_path, index=False)
    pd.DataFrame(columns=TEAM_SCHEMA).assign(cache_timestamp=cache_timestamp).to_parquet(team_path, index=False)


def _cache_fresh(max_age_hours: int = 24) -> bool:
    player_path, team_path = _cache_paths()
    if not player_path.exists() or not team_path.exists():
        return False
    now = datetime.now(timezone.utc)
    player_mtime = datetime.fromtimestamp(player_path.stat().st_mtime, tz=timezone.utc)
    team_mtime = datetime.fromtimestamp(team_path.stat().st_mtime, tz=timezone.utc)
    return (now - player_mtime) < timedelta(hours=max_age_hours) and (now - team_mtime) < timedelta(hours=max_age_hours)


def ensure_lineup_cache(days: int = 60, season: str | None = None, force: bool = False) -> None:
    if not force and _cache_fresh():
        print("Lineup cache fresh; skipping rebuild.")
        return
    build_lineup_cache(days=days, season=season)


def build_lineup_cache(days: int = 60, season: str | None = None) -> None:
    cache_timestamp = datetime.now(timezone.utc).isoformat()
    if importlib.util.find_spec("pbpstats") is None:
        _write_empty_cache(cache_timestamp)
        print("pbpstats not available; wrote empty lineup cache.")
        return

    try:
        from pbpstats.client import Client  # type: ignore
    except Exception:
        _write_empty_cache(cache_timestamp)
        print("pbpstats import failed; wrote empty lineup cache.")
        return

    try:
        settings = {"Games": {"Season": season or "2025-26"}}
        client = Client(settings)
        _ = client  # placeholder for future pbpstats extraction
        _write_empty_cache(cache_timestamp)
        print("pbpstats available but data extraction not implemented; wrote empty lineup cache.")
    except Exception as exc:
        _write_empty_cache(cache_timestamp)
        print(f"pbpstats fetch failed; wrote empty lineup cache: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Build lineup/on-off cache with pbpstats (optional).")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--season", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force:
        build_lineup_cache(days=args.days, season=args.season)
    else:
        ensure_lineup_cache(days=args.days, season=args.season)


if __name__ == "__main__":
    main()
