from __future__ import annotations

from pathlib import Path
import pandas as pd

MASTER_PATH = Path("data/lines/sdi_props_master.csv")
OUTPUT_PATH = Path("data/lines/props_line_movement.csv")

OUTPUT_COLUMNS = [
    "game_date",
    "player",
    "stat",
    "book",
    "open_line",
    "open_over_odds",
    "open_under_odds",
    "open_ts",
    "close_line",
    "close_over_odds",
    "close_under_odds",
    "close_ts",
    "line_move",
    "over_price_move",
    "under_price_move",
]


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MASTER_PATH.exists():
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(OUTPUT_PATH, index=False)
        print("Missing props master; wrote empty line movement file.")
        return

    try:
        df = pd.read_csv(MASTER_PATH)
    except Exception:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(OUTPUT_PATH, index=False)
        print("Failed to read props master; wrote empty line movement file.")
        return

    if df.empty:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(OUTPUT_PATH, index=False)
        print("Props master empty; wrote empty line movement file.")
        return

    for col in ["game_date", "player", "stat", "book", "line", "over_odds", "under_odds", "snapshot_ts"]:
        if col not in df.columns:
            df[col] = pd.NA

    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce")
    df = df.dropna(subset=["snapshot_ts", "line"])
    if df.empty:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(OUTPUT_PATH, index=False)
        print("No valid snapshot rows; wrote empty line movement file.")
        return

    df = df.sort_values("snapshot_ts")
    groups = df.groupby(["game_date", "player", "stat", "book"], dropna=False)
    rows = []
    for (game_date, player, stat, book), grp in groups:
        if grp.empty:
            continue
        open_row = grp.iloc[0]
        close_row = grp.iloc[-1]
        open_line = open_row.get("line")
        close_line = close_row.get("line")
        if pd.isna(open_line) or pd.isna(close_line):
            continue

        open_over = open_row.get("over_odds")
        open_under = open_row.get("under_odds")
        close_over = close_row.get("over_odds")
        close_under = close_row.get("under_odds")
        rows.append({
            "game_date": str(game_date)[:10],
            "player": player,
            "stat": stat,
            "book": book,
            "open_line": open_line,
            "open_over_odds": open_over,
            "open_under_odds": open_under,
            "open_ts": open_row.get("snapshot_ts"),
            "close_line": close_line,
            "close_over_odds": close_over,
            "close_under_odds": close_under,
            "close_ts": close_row.get("snapshot_ts"),
            "line_move": close_line - open_line,
            "over_price_move": (close_over - open_over) if pd.notna(close_over) and pd.notna(open_over) else pd.NA,
            "under_price_move": (close_under - open_under) if pd.notna(close_under) and pd.notna(open_under) else pd.NA,
        })

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out.to_csv(OUTPUT_PATH, index=False)
    print("Saved line movement file:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
