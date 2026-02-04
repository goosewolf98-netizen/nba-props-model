import os
import re
import requests
import pandas as pd
from pathlib import Path
from io import BytesIO

OWNER = "sportsdataverse"
REPO = "sportsdataverse-data"

DATA_TYPES = {
    "betting": ["nba_betting", "nba_odds", "nba_lines", "nba_props"],
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

def find_release_by_keyword(token: str | None, keywords: list[str]):
    try:
        releases = gh_get(f"https://api.github.com/repos/{OWNER}/{REPO}/releases?per_page=100", token)
        for rel in releases:
            t = (rel.get("tag_name") or "").lower()
            name = (rel.get("name") or "").lower()
            if any(k in t for k in keywords) or any(k in name for k in keywords):
                return rel
    except Exception as e:
        print(f"Error searching releases: {e}")
    return None

def main():
    token = os.getenv("GITHUB_TOKEN")
    out_dir = Path("data/lines")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("--- Searching for Betting Data ---")
    keywords = DATA_TYPES["betting"]
    release = find_release_by_keyword(token, keywords)

    if not release:
        print("No betting/odds release found in sportsdataverse-data.")
        return

    print(f"Found release: {release.get('tag_name')}")
    assets = release.get("assets") or []

    for a in assets:
        print(f"Asset: {a['name']}")
        # Try to download if it looks like betting data
        if "csv" in a['name'] or "parquet" in a['name']:
            print(f"Downloading potentially relevant asset: {a['name']}")
            try:
                with requests.get(a["browser_download_url"], stream=True, timeout=180) as r:
                    if r.status_code == 200:
                        with open(out_dir / a['name'], "wb") as f:
                             for chunk in r.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                        print(f"Saved to {out_dir / a['name']}")
                    else:
                        print(f"Failed to download {a['name']}")
            except Exception as e:
                print(f"Error downloading {a['name']}: {e}")

if __name__ == "__main__":
    main()
