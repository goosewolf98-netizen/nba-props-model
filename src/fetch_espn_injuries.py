import pandas as pd
import requests
from pathlib import Path

OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ESPN_INJ_URL = "https://www.espn.com/nba/injuries"

def main():
    # ESPN injuries page has tables; pandas can read them
    html = requests.get(ESPN_INJ_URL, timeout=30).text
    tables = pd.read_html(html)

    # ESPN typically returns one table per team.
    # We'll stack them and keep common columns.
    all_rows = []
    for t in tables:
        # normalize column names
        t.columns = [str(c).strip() for c in t.columns]
        # expected: Player, Date, Status, Comment (varies)
        all_rows.append(t)

    df = pd.concat(all_rows, ignore_index=True)

    # Clean up a bit
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    out_csv = OUT_DIR / "espn_injuries.csv"
    df.to_csv(out_csv, index=False)

    print(f"Saved {out_csv} with {len(df)} rows and {len(df.columns)} cols")
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()
