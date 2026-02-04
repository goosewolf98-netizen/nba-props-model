import os
import re
import requests
import pandas as pd
from pathlib import Path
from io import BytesIO

OWNER = "sportsdataverse"
REPO = "sportsdataverse-data"

DATA_TYPES = {
    "player_box": ["espn_nba_player_boxscores", "nba_player_boxscores"],
    "team_box": ["espn_nba_team_boxscores", "nba_team_boxscores"],
    "schedule": ["espn_nba_schedules", "nba_schedules"],
}

SEASONS = [2024, 2025]

def gh_get(url: str, token: str | None = None) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"GitHub API error {r.status_code}: {r.text[:300]}")
    return r.json()

def find_release_by_tag(token: str | None, tag: str):
    try:
        return gh_get(f"https://api.github.com/repos/{OWNER}/{REPO}/releases/tags/{tag}", token)
    except Exception:
        return None

def pick_asset(release: dict, year: int, patterns: list[str]) -> dict | None:
    assets = release.get("assets") or []
    # DEBUG: Print assets
    print(f"DEBUG: Assets in release {release.get('tag_name')}: {[a['name'] for a in assets]}")

    exts = ["parquet", "csv"]

    for ext in exts:
        for p in patterns:
            # Regex to match pattern + year + ext
            rx = re.compile(rf".*{year}.*\.{ext}$", re.I)
            for a in assets:
                name = a.get("name", "")
                if rx.match(name) and "wnba" not in name.lower():
                    return a, ext

    # Fallback: if patterns is empty (which I passed in main), just match year and ext
    if not patterns:
        for ext in exts:
             rx = re.compile(rf".*{year}.*\.{ext}$", re.I)
             for a in assets:
                name = a.get("name", "")
                if rx.match(name) and "wnba" not in name.lower():
                    return a, ext

    return None, None

def download_df(url: str, ext: str) -> pd.DataFrame:
    print(f"Downloading {url}...")
    with requests.get(url, stream=True, timeout=180) as r:
        if r.status_code >= 400:
            raise RuntimeError(f"Download failed {r.status_code}")
        data = BytesIO(r.content)
        if ext == "parquet":
            return pd.read_parquet(data)
        else:
            return pd.read_csv(data)

def main():
    token = os.getenv("GITHUB_TOKEN")
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_store = {k: [] for k in DATA_TYPES}

    for dtype, tags in DATA_TYPES.items():
        print(f"--- Processing {dtype} ---")
        release = None
        for tag in tags:
            release = find_release_by_tag(token, tag)
            if release:
                print(f"Found release: {tag}")
                break

        if not release:
            print(f"Warning: No release found for {dtype} (checked {tags})")
            continue

        for year in SEASONS:
            asset, ext = pick_asset(release, year, [])
            if not asset:
                print(f"Warning: No asset found for {dtype} {year} in {release.get('tag_name')}")
                continue

            print(f"Found asset for {dtype} {year}: {asset['name']}")
            try:
                df = download_df(asset["browser_download_url"], ext)
                data_store[dtype].append(df)
            except Exception as e:
                print(f"Error downloading/reading {dtype} {year}: {e}")

    for dtype, dfs in data_store.items():
        if not dfs:
            print(f"No data for {dtype}, skipping save.")
            continue

        full_df = pd.concat(dfs, ignore_index=True)

        if dtype == "player_box":
            out_name = "nba_player_box.csv"
        elif dtype == "team_box":
            out_name = "nba_team_box.csv"
        elif dtype == "schedule":
            out_name = "nba_schedule.csv"
        else:
            out_name = f"{dtype}.csv"

        out_path = out_dir / out_name
        full_df.to_csv(out_path, index=False)
        print(f"Saved {out_path} with {len(full_df)} rows.")

if __name__ == "__main__":
    main()
