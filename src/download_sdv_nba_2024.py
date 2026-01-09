import os
import re
import requests
from pathlib import Path

OWNER = "sportsdataverse"
REPO = "sportsdataverse-data"

TAG_CANDIDATES = [
    "espn_nba_player_boxscores",
    "nba_player_boxscores",
    "espn_nba_player_box",
]

def gh_get(url: str, token: str | None = None) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"GitHub API error {r.status_code}: {r.text[:300]}")
    return r.json()

def find_release(token: str | None):
    for tag in TAG_CANDIDATES:
        try:
            return gh_get(f"https://api.github.com/repos/{OWNER}/{REPO}/releases/tags/{tag}", token), tag
        except Exception:
            pass

    releases = gh_get(f"https://api.github.com/repos/{OWNER}/{REPO}/releases?per_page=100", token)
    for rel in releases:
        t = (rel.get("tag_name") or "").lower()
        name = (rel.get("name") or "").lower()
        if "nba" in t and ("player" in t) and ("box" in t):
            return rel, rel.get("tag_name")
        if "nba" in name and ("player" in name) and ("box" in name):
            return rel, rel.get("tag_name")
    raise RuntimeError("Could not find an NBA player boxscore release in sportsdataverse-data.")

def pick_asset(release: dict, year: int) -> dict:
    assets = release.get("assets") or []
    if not assets:
        raise RuntimeError("Release has no assets listed (or GitHub API returned none).")

    patterns = [
        (re.compile(rf".*{year}.*\.parquet$", re.I), "parquet"),
        (re.compile(rf".*{year}.*\.csv$", re.I), "csv"),
    ]

    for rx, _ in patterns:
        for a in assets:
            if rx.match(a.get("name", "")):
                return a

    for ext in (".parquet", ".csv"):
        for a in assets:
            if a.get("name", "").lower().endswith(ext):
                return a

    raise RuntimeError("No suitable asset found (parquet/csv) for the requested year.")

def download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as r:
        if r.status_code >= 400:
            raise RuntimeError(f"Download failed {r.status_code}: {r.text[:200]}")
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

if __name__ == "__main__":
    year = int(os.getenv("NBA_SEASON_START_YEAR", "2024"))
    token = os.getenv("GITHUB_TOKEN")

    release, tag = find_release(token)
    asset = pick_asset(release, year)

    out_dir = Path("data/raw")
    out_path = out_dir / asset["name"]

    print(f"Using release tag: {tag}")
    print(f"Downloading asset: {asset['name']}")
    download(asset["browser_download_url"], out_path)
    print(f"Saved: {out_path}")
