from pathlib import Path
import pandas as pd

OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ESPN_INJ_URL = "https://www.espn.com/nba/injuries"

def write_error(msg: str):
    (OUT_DIR / "espn_injuries_error.txt").write_text(msg)
    # also write an empty CSV so downstream steps still have a file
    (OUT_DIR / "espn_injuries.csv").write_text("error\n" + msg + "\n")

def main():
    try:
        # Import requests inside try so missing deps doesn't crash the workflow
        import requests
    except Exception as e:
        write_error(f"requests import failed: {e}")
        print("ESPN injuries fetch skipped (requests missing).")
        return

    try:
        html = requests.get(ESPN_INJ_URL, timeout=30).text
    except Exception as e:
        write_error(f"requests.get failed: {e}")
        print("ESPN injuries fetch skipped (network error).")
        return

    try:
        tables = pd.read_html(html)
        if not tables:
            write_error("pandas.read_html returned 0 tables (page layout may have changed).")
            print("No injury tables found.")
            return

        df = pd.concat(tables, ignore_index=True)

        # Light cleanup
        df.columns = [str(c).strip() for c in df.columns]
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip()

        out_csv = OUT_DIR / "espn_injuries.csv"
        df.to_csv(out_csv, index=False)

        print(f"Saved {out_csv} with {len(df)} rows and {len(df.columns)} cols")
        print("Columns:", list(df.columns))

    except Exception as e:
        # Most common: missing lxml/html5lib OR ESPN HTML changed
        write_error(f"pandas.read_html failed: {e}")
        print("ESPN injuries parsing failed, but workflow will continue.")

if __name__ == "__main__":
    main()
