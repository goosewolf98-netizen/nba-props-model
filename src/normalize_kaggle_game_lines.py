from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_kaggle_files(base_dir: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in base_dir.rglob("*"):
        if path.suffix.lower() == ".csv":
            try:
                frames.append(pd.read_csv(path))
            except Exception:
                continue
        elif path.suffix.lower() == ".parquet":
            try:
                frames.append(pd.read_parquet(path))
            except Exception:
                continue
    return frames


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["game_date", "date", "Date", "GAME_DATE"]:
        if col in df.columns:
            df = df.rename(columns={col: "game_date"})
            break
    for col in ["home_abbr", "home_team", "Home Team", "home", "team_home"]:
        if col in df.columns:
            df = df.rename(columns={col: "home_abbr"})
            break
    for col in ["away_abbr", "away_team", "Away Team", "away", "team_away"]:
        if col in df.columns:
            df = df.rename(columns={col: "away_abbr"})
            break
    for col in ["spread_close", "closing_spread", "spread", "Close", "closing_line"]:
        if col in df.columns:
            df = df.rename(columns={col: "spread_close"})
            break
    for col in ["total_close", "closing_total", "total", "Total", "closing_total_line"]:
        if col in df.columns:
            df = df.rename(columns={col: "total_close"})
            break
    return df


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype(str)
    return df


def main() -> None:
    datasets = {
        "cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024": Path(
            "data/kaggle/cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024"
        ),
        "christophertreasure/nba-odds-data": Path("data/kaggle/christophertreasure/nba-odds-data"),
    }

    output_frames: list[pd.DataFrame] = []
    for source, base_dir in datasets.items():
        if not base_dir.exists():
            continue
        for frame in _load_kaggle_files(base_dir):
            if frame.empty:
                continue
            frame = _normalize_columns(frame)
            frame = _normalize_dates(frame)
            if "game_date" not in frame.columns or "home_abbr" not in frame.columns or "away_abbr" not in frame.columns:
                continue
            if "spread_close" not in frame.columns and "total_close" not in frame.columns:
                continue
            out = frame[[c for c in ["game_date", "home_abbr", "away_abbr", "spread_close", "total_close"] if c in frame.columns]].copy()
            out["source"] = source
            output_frames.append(out)

    if not output_frames:
        print("Kaggle game lines not found, skipping")
        return

    lines = pd.concat(output_frames, ignore_index=True)
    lines = lines.dropna(subset=["game_date", "home_abbr", "away_abbr"], how="any")
    if "spread_close" in lines.columns:
        lines["spread_close"] = pd.to_numeric(lines["spread_close"], errors="coerce")
    if "total_close" in lines.columns:
        lines["total_close"] = pd.to_numeric(lines["total_close"], errors="coerce")

    output_path = Path("data/raw/kaggle_game_lines.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines.to_parquet(output_path, index=False)
    print("Saved", output_path)


if __name__ == "__main__":
    main()
