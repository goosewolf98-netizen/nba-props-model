import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.db.manager import DBManager

LINES_PATH = Path("data/lines/props_lines.csv")

def ingest_lines():
    if not LINES_PATH.exists():
        print(f"No props file found at {LINES_PATH}")
        return

    print(f"Reading props from {LINES_PATH}...")
    try:
        df = pd.read_csv(LINES_PATH)
        if df.empty:
            print("Props file is empty.")
            return

        print(f"Found {len(df)} lines. Upserting to database...")
        db = DBManager()
        db.upsert_prop_lines(df)

    except Exception as e:
        print(f"Error ingesting lines: {e}")

if __name__ == "__main__":
    ingest_lines()
